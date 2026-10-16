// CUDA kernels for route A (plan 4.7): atomic slot build + global-softmax merge.
//
// Semantics identical to the Triton reference (src/moba_opt/ops/merge_triton.py):
//   build: slot = atomic ticket per global row p; stores masked by slot < kpad;
//          overflow is counted and reported (never written out of bounds).
//   merge: m = max_j l_j; w_j = exp(l_j - m); O = sum_j w_j o_j / z;
//          z = sum_j w_j; L = m + log(z)   (plan 4.3)
// One warp per query/head row; lanes cover D contiguously (coalesced).
// All math in fp32 with accurate expf/logf (no fast-math); O cast back to the
// input dtype; global LSE stays fp32. Launches run on the INPUT's device
// (CUDAGuard), not whatever device happens to be current.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moba_opt {

constexpr int kBuildThreads = 256;
constexpr int kWarpsPerBlock = 4;

__global__ void build_slots_atomic_kernel(
    const int64_t* __restrict__ mapping,
    int* __restrict__ slots,
    int* __restrict__ counts,
    int* __restrict__ overflow,
    const int64_t M,
    const int kpad) {
  const int64_t m = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (m >= M) return;
  const int64_t p = mapping[m];
  const int slot = atomicAdd(&counts[p], 1);
  if (slot < kpad) {
    slots[p * kpad + slot] = (int)m;
  } else if (overflow != nullptr) {
    // capacity overflow: never write out of bounds; flag it so the launcher
    // can raise instead of letting merge read uninitialized slots.
    atomicAdd(overflow, 1);
  }
}

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }

template <typename scalar_t, int D>
__global__ void merge_kernel(
    const scalar_t* __restrict__ self_out,
    const float* __restrict__ self_lse,
    const scalar_t* __restrict__ remote_out,
    const float* __restrict__ remote_lse,
    const int* __restrict__ slots,
    const int* __restrict__ counts,
    scalar_t* __restrict__ out,
    float* __restrict__ lse,
    const int64_t P,
    const int kpad) {
  constexpr int LPT = D / 32;  // elements per lane
  const int lane = threadIdx.x & 31;
  const int64_t gwarp = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int64_t stride = (int64_t)gridDim.x * (blockDim.x >> 5);

  for (int64_t row = gwarp; row < P; row += stride) {
    const float l_self = self_lse[row];
    // defensive: with check_capacity (or the structural bound of the official
    // routing, counts <= topk-1 = kmax <= kpad) this never triggers; it only
    // keeps an unchecked misuse from reading out of bounds.
    const int cnt = min(counts[row], kpad);

    // lane j (< cnt, cnt <= kpad <= 8 <= 32) carries the j-th remote contribution
    float l_j = -INFINITY;
    int slot_j = 0;
    if (lane < cnt) {
      slot_j = slots[row * kpad + lane];
      l_j = remote_lse[slot_j];
    }

    float m = fmaxf(l_self, l_j);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    const float w_self = expf(l_self - m);
    float w_j = (lane < cnt) ? expf(l_j - m) : 0.0f;
    // reduce the remote weights first, then add w_self exactly once
    float z = w_j;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) z += __shfl_xor_sync(0xffffffffu, z, off);
    z += w_self;

    float acc[LPT];
    const scalar_t* srow = self_out + row * D;
#pragma unroll
    for (int i = 0; i < LPT; ++i) acc[i] = to_f32(srow[lane * LPT + i]) * w_self;

    for (int j = 0; j < cnt; ++j) {
      const float w_b = __shfl_sync(0xffffffffu, w_j, j);
      const int s_b = __shfl_sync(0xffffffffu, slot_j, j);
      const scalar_t* rrow = remote_out + (int64_t)s_b * D;
#pragma unroll
      for (int i = 0; i < LPT; ++i) acc[i] += w_b * to_f32(rrow[lane * LPT + i]);
    }

    const float inv_z = 1.0f / z;
    scalar_t* orow = out + row * D;
#pragma unroll
    for (int i = 0; i < LPT; ++i) orow[lane * LPT + i] = from_f32<scalar_t>(acc[i] * inv_z);
    if (lane == 0) lse[row] = m + logf(z);
  }
}

inline int kpad_of(int64_t kmax) {
  int kpad = 1;
  while (kpad < kmax) kpad <<= 1;
  return kpad;
}

template <typename scalar_t>
void launch_merge(
    const scalar_t* self_out, const float* self_lse,
    const scalar_t* remote_out, const float* remote_lse,
    const int* slots, const int* counts,
    scalar_t* out, float* lse,
    int64_t P, int64_t D, int kpad, cudaStream_t stream) {
  const int threads = kWarpsPerBlock * 32;
  const int64_t blocks = (P + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (D == 64) {
    merge_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
        self_out, self_lse, remote_out, remote_lse, slots, counts, out, lse, P, kpad);
  } else if (D == 128) {
    merge_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
        self_out, self_lse, remote_out, remote_lse, slots, counts, out, lse, P, kpad);
  } else {
    TORCH_CHECK(false, "head_dim must be 64 or 128, got ", D);
  }
}

std::vector<torch::Tensor> merge_forward_cuda(
    torch::Tensor self_out,
    torch::Tensor self_lse,
    torch::Tensor remote_out,
    torch::Tensor remote_lse,
    torch::Tensor mapping,
    int64_t kmax,
    bool check_capacity) {
  TORCH_CHECK(self_out.is_cuda() && self_out.dim() == 3, "self_out must be CUDA [T,H,D]");
  const at::cuda::CUDAGuard device_guard(self_out.device());
  const auto st = self_out.scalar_type();
  TORCH_CHECK(st == at::kFloat || st == at::kBFloat16 || st == at::kHalf,
              "self_out dtype must be fp32/bf16/fp16");
  TORCH_CHECK(self_lse.scalar_type() == at::kFloat && remote_lse.scalar_type() == at::kFloat,
              "lse must be fp32");
  TORCH_CHECK(mapping.scalar_type() == at::kLong, "mapping must be int64");
  TORCH_CHECK(kmax >= 1 && kmax <= 8, "kmax must be in [1,8]");

  const int64_t T = self_out.size(0);
  const int64_t H = self_out.size(1);
  const int64_t D = self_out.size(2);
  const int64_t P = T * H;
  const int64_t M = mapping.size(0);
  TORCH_CHECK(remote_out.numel() == M * D, "remote_out shape mismatch");
  TORCH_CHECK(remote_lse.numel() == M, "remote_lse shape mismatch");
  TORCH_CHECK(self_lse.numel() == P, "self_lse shape mismatch");
  const int kpad = kpad_of(kmax);

  auto self_out_c = self_out.contiguous();
  auto self_lse_c = self_lse.contiguous();
  auto remote_out_c = remote_out.reshape({M, D}).contiguous();
  auto remote_lse_c = remote_lse.contiguous();
  auto mapping_c = mapping.contiguous();

  auto iopts = torch::TensorOptions().dtype(torch::kInt32).device(self_out.device());
  auto slots = torch::empty({P, kpad}, iopts);
  auto counts = torch::zeros({P}, iopts);
  auto out = torch::empty({T, H, D}, self_out.options());
  auto lse = torch::empty({T, H}, self_out.options().dtype(torch::kFloat));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (M > 0) {
    auto overflow = check_capacity ? torch::zeros({1}, iopts) : torch::Tensor();
    const int64_t blocks = (M + kBuildThreads - 1) / kBuildThreads;
    build_slots_atomic_kernel<<<blocks, kBuildThreads, 0, stream>>>(
        mapping_c.data_ptr<int64_t>(), slots.data_ptr<int>(), counts.data_ptr<int>(),
        check_capacity ? overflow.data_ptr<int>() : nullptr, M, kpad);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (check_capacity) {
      // synchronizes the current stream of the input device (guard above)
      TORCH_CHECK(overflow.cpu().item<int>() == 0,
                  "capacity exceeded: a query row has more remote contributions than ",
                  "kmax allows (slots kpad=", kpad, "); refusing to merge with a ",
                  "partially filled slot table");
    }
  }

  const float* sl = self_lse_c.data_ptr<float>();
  const float* rl = remote_lse_c.data_ptr<float>();
  const int* sp = slots.data_ptr<int>();
  const int* cp = counts.data_ptr<int>();
  if (st == at::kFloat) {
    launch_merge<float>(self_out_c.data_ptr<float>(), sl, remote_out_c.data_ptr<float>(), rl,
                        sp, cp, out.data_ptr<float>(), lse.data_ptr<float>(), P, D, kpad, stream);
  } else if (st == at::kBFloat16) {
    launch_merge<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(self_out_c.data_ptr<at::BFloat16>()), sl,
        reinterpret_cast<__nv_bfloat16*>(remote_out_c.data_ptr<at::BFloat16>()), rl,
        sp, cp,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), lse.data_ptr<float>(),
        P, D, kpad, stream);
  } else {
    launch_merge<__half>(
        reinterpret_cast<__half*>(self_out_c.data_ptr<at::Half>()), sl,
        reinterpret_cast<__half*>(remote_out_c.data_ptr<at::Half>()), rl,
        sp, cp,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), lse.data_ptr<float>(),
        P, D, kpad, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

}  // namespace moba_opt
