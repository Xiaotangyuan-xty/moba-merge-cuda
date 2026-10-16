// Pybind bindings for the CUDA merge kernels (plan 4.7).

#include <torch/extension.h>

namespace moba_opt {
std::vector<torch::Tensor> merge_forward_cuda(
    torch::Tensor self_out,
    torch::Tensor self_lse,
    torch::Tensor remote_out,
    torch::Tensor remote_lse,
    torch::Tensor mapping,
    int64_t kmax,
    bool check_capacity);
}  // namespace moba_opt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_forward", &moba_opt::merge_forward_cuda,
        "atomic slot build + global-softmax merge (CUDA)",
        py::arg("self_out"), py::arg("self_lse"), py::arg("remote_out"),
        py::arg("remote_lse"), py::arg("mapping"), py::arg("kmax"),
        py::arg("check_capacity") = true);
}
