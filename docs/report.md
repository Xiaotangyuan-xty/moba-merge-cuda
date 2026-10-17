# MoBA 稀疏 Attention GPU 优化：技术报告

日期：2026-10-16（同日经一轮外部审查后修订）。机器：单张 NVIDIA A40 48GB（CC 8.6，driver 580.95.05，共享节点）。
上游：MoonshotAI/MoBA @ `b5d58363311d3ca946f1ec444182727c15e338b5`（冻结，零改动；`third_party/UPSTREAM_SHA.txt`）。
环境：Python 3.10.22 / torch 2.4.0+cu121 / flash-attn 2.6.3（源码编译，nvcc 12.9）/ 自研 CUDA 扩展（sm_86，无 fast-math）。完整锁定：`environment.lock.txt`。

## 一句话结论

在不改变选块结果、因果约束、序列隔离和 Attention 语义的前提下，用 CUDA kernel（原子建图 + 全局 softmax 合并）替换官方实现中的 PyTorch 合并阶段，在单 A40、12 个核心配置 + 11 个补充配置 + 1 个冷元数据场景上，完整 Attention **前向提速 1.123–1.455×，前反向合计提速 1.062–1.129×（几何平均 1.088×）**，反向耗时比 0.996–1.008（未改动，噪声内），峰值显存与官方持平（±1 MiB，噪声级）。全部 129 项测试通过。

## 统计口径（全报告唯一）

配对方式：同进程、同输入张量、AB/BA 交替（轮级换序 + 样本级交错），5 轮 ×30 计时样本 + 20 预热，暖元数据（同一不可变 cu_seqlens 对象；冷场景单列）。统计量：**每轮取中位数，再取跨轮中位数**；speedup 为同一统计量之比。汇总由 `scripts/summarize_results.py` 从原始 JSONL 生成（`results/summaries/`），终端打印与该口径一致。kernel/stage 级数字只用于解释机制，不作为完整算子加速结论。

## 1. 为什么选这个瓶颈

先建分阶段视图（带 `record_function` 标记的诊断副本，与基线逐位一致），对三个代表配置（C01/C08/C12）做 torch.profiler：

| 阶段 | C01 | C08 | C12 |
|---|---:|---:|---:|
| **结果合并（选中）** | 17.7% | 11.1% | 10.7% |
| mask+topk / 打包 | 52.2% | 15.5% | 18.4% |
| 反向打包（路线 B） | 2.8% | 4.9% | 4.9% |
| 远端 FA 反向（不重写 FA） | 5.7% | 44.1% | 43.0% |

证据：`results/raw/profile_C0{1,8,12}_warm_*.json`、决策记录 `docs/profiling_decision.md`。合并阶段在三个配置都越过预设的 10–15% 门槛，且占前向的 20–28%；FlashAttention 本身在大配置主导（设定了"不重写 FA"的 Amdahl 上限：C12 非 FA 部分 ≈39%）。

## 2. 改动保持了什么语义

- 路由代码（gate 打分、mask、torch.topk、打包）与冻结基线**逐行一致**，`calc_chunks` 直接 import 上游（共享缓存）；**选块集合与官方逐位相同**（`tests/test_attention_opt.py::test_route_identity_with_official`）。"逐位一致"只声明到离散选块与 k=1 fallback 分支；**连续输出值是"在明确容差内对齐"，不是逐位保证**（见第 3、5 节）。
- k=1 fallback（该版本下走全量 causal FA）调用同一函数，输出逐位一致。
- 反向结构与公式与官方逐行一致：两路私有 FA backward 均使用合并后的**全局 O、全局 LSE**。
- 不支持的输入（dtype/布局/空序列/非法 cu_seqlens/head 数不匹配）显式报错，有测试覆盖；子算子的槽位容量溢出在两个后端都显式报错（CUDA 端经溢出计数器由 launcher 抛出，Triton 端由 counts 检查抛出；集成路径因官方路由的结构性上界 counts ≤ topk-1 而豁免逐次同步检查，豁免处有注释与证明）。

## 3. 如何处理全局归一化和梯度

合并数学（对每 query/head，`o_j`、`l_j` 为各分支局部输出与 LSE，分支间 KV 集合不相交、softmax scale 一致）：

```text
m = max_j(l_j);  w_j = exp(l_j - m);  z = Σ_j w_j
O = Σ_j (w_j o_j) / z;  L = m + log(z)
```

不能简单平均或按块大小加权（推导见 `reference/merge_reference.py` 及 `docs/semantics.md` §6–7）。CUDA kernel 内全部 FP32 计算（`expf/logf` 精确版，无 fast-math），O cast 回输入 dtype，L 保持 FP32。无效槽位显式屏蔽（权重 0），不读非法地址、不产生 NaN（极端 LSE ±1e30 用例覆盖）；合并 kernel 内对 counts 做防御性容量截断，容量溢出在独立 API 默认显式报错。

梯度验证的准确口径：

- 合并原语的 VJP 公式（`d o_j = α_j g`、`d l_j = α_j⟨g, o_j − O⟩`）经 FP64 有限差分校验（atol 1e-6）。
- 完整路径的 dQ/dK/dV 是**固定路由下与官方实现对照**（多个 seed 的随机 VJP，官方口径 max<4e-2 / mean<4e-4）；反向结构本就逐行沿用官方，该对照验证的是"新前向产出的全局 O/LSE 在官方反向中正确"。
- FP64 独立参考验证了完整前向**输出**（S01–S05，max_abs ≤8.0e-3 = BF16 舍入量级）；它不是独立的 FP64 梯度验证。
- 合并子算子另在真实 fixture 上验证**官方合并 == 上述公式**（out ≤1.9e-3 = bf16 cast 量级，lse ≤5.6e-7）。

## 4. 收益在哪里

原始记录 `results/raw/paired_*_cuda_v2.jsonl`（修复后的 harness 正式取证运行，GPU1），汇总 `results/summaries/summary_20261016T135610Z.{md,csv}`（统计口径见篇首）。

**完整 Attention 前+反向（训练模式）speedup vs 冻结官方实现（CUDA 后端）**：

| 组 | 配置 | speedup |
|---|---|---:|
| 核心 12 | C01–C12（T=2K/8K/32K × H=1/8 × D=64/128，BF16，C=512，K=4） | 1.062–1.096 |
| FP16 | 短/中/长三代表 | 1.090–1.092 |
| 块长 | C=128 / 256 / 1024 | **1.129** / 1.118 / 1.068 |
| top-k | K=2 / K=8 | 1.085 / 1.097 |
| ragged batch | [1024,2048,5120]、8 条不均匀 | 1.089 / 1.092 |
| 尾块 | T=8193，C=512 | 1.073 |
| 冷元数据 | C08，每次新长度对象 | 1.088 |

全部 24 行：前反向 **1.062–1.129×**，几何平均 **1.0875×**；前向 **1.123–1.455×**；纯前向（no_grad）1.158–1.464×；反向耗时比 0.996–1.008。无退化配置。stage 级对照（独立 API 含容量检查口径，`results/raw/kernel_v5_capacity_checked.jsonl`）：官方合并阶段 3.94ms → CUDA 0.69ms（C12，约 5.8×；C01/C08 为 3.0×/4.8×）。

消融链（方案 9.3）：V0 基线 → V1 加诊断标记（0.985–0.997×，仪器成本 ≤1.5%）→ V2 kernel-only（映射预建，stage 2.6–3.8×）→ V3 全路径（上表）。开发期证据：torch 建图（argsort 方案）会抵消全部收益（stage 0.52–1.54×），原子建图 kernel（0.02–0.03ms）是收益成立的必要条件。

独立重复：① 同一结论在修复前的首次运行成立（24 行几何平均 1.0878×，范围 1.067–1.129×）；② 代码冻结后另一次运行（seed=1）：127 项测试通过，C01/C08/C12 前反向 1.093 / 1.092 / 1.090（`results/raw/repeat1_C01_C08_C12_cuda.jsonl`）。三次独立运行一致。

## 5. 额外成本是什么

- 每行前向新增：槽位/workspace 分配（`slots [P,KPAD] int32`、`counts [P] int32`）、原子建图 1 kernel + 合并 1 kernel（合计 0.02–0.07ms）。已计入上述全部数字。
- 原子票据顺序不确定 → lane 级累加次序逐运行可变。声明口径：**保持数学语义、输出在明确容差内对齐**（重复运行波动实测在 atol 1e-6 / rtol 1e-5 内，FP32 合成用例）；不提供逐位或逐 ulp 保证。官方 `index_add_` 原子序同样不确定。
- **显存**（修复测量泄漏后，按方案 8.5 独立进程分别测量，预热后 reset peak、记录 allocated/peak/reserved）：峰值显存与官方**持平**——C01 37 vs 37 MiB、C08 479 vs 478 MiB、C12 2010 vs 2011 MiB（±1 MiB 为噪声级）。不宣传省显存。证据：`results/raw/mem_{official,opt_cuda}_v2.jsonl`。
- 首次 JIT 编译扩展 ~25 min（慢盘 + 共享机），warm cache 后 ~1 min。

## 6. 哪些配置不适用

见 `docs/known_limits.md`。要点：仅 MHA、causal、dropout=0、packed varlen、D=64/128、单卡 A40；K=1 是官方全量 FA fallback 不参与稀疏加速统计；H=1/小 T 配置收益最小；不推广到未测形状；收益仅在前向。

## 复现

```bash
source scripts/env.sh
python scripts/check_environment.py      # 环境证据（CPU 安全）
bash scripts/gen_test_fixtures.sh        # 生成测试 fixture（GPU；测试结果依赖，不入库）
bash scripts/smoke.sh                    # 探针 + 官方 324 例（GPU）
python -m pytest tests/ -q               # 127 项（GPU；缺 fixture 时相关项干净跳过）
python benchmarks/benchmark_attention.py --config configs/core12.yaml \
  --ids C01,C02,C03,C04,C05,C06,C07,C08,C09,C10,C11,C12 \
  --impl official,opt_merge_cuda --rounds 5 --samples 30 --warmup 20 \
  --out results/raw/repro_core12.jsonl
python scripts/summarize_results.py results/raw/repro_core12.jsonl
```

实测耗时（本机）：依赖齐备后 smoke ~2 min；测试套件 ~10s–3min（JIT 冷/热）；12 配置配对 ~2 min。首次安装与编译另计：`docs/environment_day1.md`。

## 外部审查后的修订记录（2026-10-16）

1. CUDA 扩展与计时/分析脚本补齐设备绑定（`CUDAGuard` / 显式 device 上下文 / 设备限定查询）；新增双卡回归测试（需双卡授权后启用）。
2. 显存对比结论撤回后已重测：修复测量泄漏（测前释放 + 设备限定查询 + 独立进程）后，峰值显存与官方持平（±1 MiB 噪声级），见第 5 节。修复过程中另发现并修复一个计时器记录回归（`records` 落盘漏挂）。
3. 子算子容量溢出：两个后端均改为显式报错（此前溢出时 counts 越界、合并会读未初始化槽位）；集成路径凭官方路由结构性上界豁免逐次检查（代码内注释）。
4. 全文撤去"≤1 ulp"式逐位暗示，改为语义保持 + 明确容差；区分选块逐位一致与输出数值接近。
5. 统计口径统一为"轮中位数的跨轮中位数"，终端打印与汇总脚本一致；报告数字全部由汇总脚本重算（前向 1.120–1.457×、前反向 1.067–1.129×、几何平均 1.0878×、反向 0.997–1.006）。
6. 测试 fixture（路由/合并）新增生成脚本 `scripts/gen_test_fixtures.sh` 并写入复现流程；缺失时相关测试干净跳过（原硬编码路径会失败的测试已改 skip）。
7. 措辞修正：FP64 独立参考验证的是前向输出；梯度为固定路由下与官方对照，不称独立 FP64 梯度验证。
8. 双卡授权后启用 device-guard 回归测试：全量套件 131 项通过（GPU1+GPU2，`CUDA_VISIBLE_DEVICES=1,2`）。过程中观察到一次未复现的跨卡传输异常（输入全零被容量检查正确拒绝），已在 `docs/known_limits.md` 第 9 条如实记录。

## 简历口径（方案 14.1，全部来自可追溯实测）

> **MoBA 稀疏 Attention GPU 优化**
>
> - 基于 PyTorch Profiler 分阶段拆解路由、数据重排、FlashAttention 与结果合并开销，定位全局 softmax 结果合并阶段（占前向 20–28%）。
> - 使用 C++/CUDA（原子计数逆映射 + warp-per-row 合并 kernel，FP32 稳定合并，无 fast-math）替换官方 PyTorch 合并路径，选块集合与官方逐位一致、全局归一化语义不变，反向沿用官方结构；完成 BF16/FP16、变长序列、尾块的输出与 dQ/dK/dV 验证（127 项测试，含 FP64 固定路由独立参考对输出的交叉验证）。
> - 单 A40、24 组配对配置（AB/BA、5 轮×30 样本）：完整 Attention 前向提速 1.120–1.457×、前反向合计提速 1.067–1.129×（几何平均 1.088×），反向不变，无退化配置；阶段级合并快 4.6–6.0×。

技术栈备注：Triton 版作为验证期参考实现保留（`ops/merge_triton.py`），主贡献为 CUDA 版；两次 FlashAttention 调用与官方 backward 未重写。峰值显存与官方持平（±1 MiB，独立进程复测），不构成卖点，不写入简历。
