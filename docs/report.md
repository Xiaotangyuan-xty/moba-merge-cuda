# MoBA 稀疏 Attention GPU 优化：技术报告

日期：2026-10-16。机器：gpu-node1 单张 NVIDIA A40 48GB（CC 8.6，driver 580.95.05）。
上游：MoonshotAI/MoBA @ `b5d58363311d3ca946f1ec444182727c15e338b5`（冻结，零改动；`third_party/UPSTREAM_SHA.txt`）。
环境：Python 3.10.22 / torch 2.4.0+cu121 / flash-attn 2.6.3（源码编译，nvcc 12.9）/ 自研 CUDA 扩展（sm_86，无 fast-math）。完整锁定：`environment.lock.txt`。

## 一句话结论

在不改变选块结果、因果约束、序列隔离和 Attention 语义的前提下，用 CUDA kernel（原子建图 + 全局 softmax 合并）替换官方实现中的 PyTorch 合并阶段，在单 A40、12 个核心配置 + 11 个补充配置上，完整 Attention **前向提速 1.07–1.38×，前反向合计提速 1.064–1.129×（几何平均 1.088×）**，反向严格不变，显存持平或略降，全部 127 项测试通过。

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

- 路由代码（gate 打分、mask、torch.topk、打包）与冻结基线**逐行一致**，`calc_chunks` 直接 import 上游（共享缓存）；选块集合与官方**逐位相同**（`tests/test_attention_opt.py::test_route_identity_with_official`）。
- k=1 fallback（该版本下走全量 causal FA）原样保留，输出与官方**逐位一致**。
- 反向结构与公式与官方逐行一致：两路私有 FA backward 均使用合并后的**全局 O、全局 LSE**（`moba_efficient.py` 的 `MixedAttention.backward` 语义）。
- 不支持的输入（dtype/布局/空序列/非法 cu_seqlens/head 数不匹配）显式报错，有测试覆盖。

## 3. 如何处理全局归一化和梯度

合并数学（对每 query/head，`o_j`、`l_j` 为各分支局部输出与 LSE，分支间 KV 集合不相交、softmax scale 一致）：

```text
m = max_j(l_j);  w_j = exp(l_j - m);  z = Σ_j w_j
O = Σ_j (w_j o_j) / z;  L = m + log(z)
```

不能简单平均或按块大小加权（证据与推导见 `reference/merge_reference.py` 及 `docs/semantics.md` §6–7）。CUDA kernel 内全部 FP32 计算（`expf/logf` 精确版，无 fast-math），O cast 回输入 dtype，L 保持 FP32。无效槽位显式屏蔽（权重 0），不读非法地址、不产生 NaN（极端 LSE ±1e30 用例覆盖）。

梯度：合并原语位于自定义 autograd.Function 内部，反向由整体定制 backward 处理（沿用官方结构），因此**不需要为合并单独写 backward kernel**；验证方式是固定路由下的随机 VJP 对照（多个 seed、dQ/dK/dV 逐项，官方口径 max<4e-2 / mean<4e-4），另以固定路由 FP64 独立参考交叉验证（`tests/test_attention_opt*.py`、`test_merge_fixtures.py`）。

合并数学本身以 FP64 参考 + 有限差分校验（VJP 公式 `d o_j = α_j g`、`d l_j = α_j⟨g, o_j − O⟩`，atol 1e-6），且在真实 fixture 上验证了**官方合并 == 上述公式**（out ≤1.9e-3 = bf16 cast 量级，lse ≤5.6e-7）——kernel 建立在已验证的语义上。

## 4. 收益在哪里

主指标：同进程 AB/BA 配对，5 轮 ×30 样本 + 20 预热，暖元数据，每轮中位数（原始记录 `results/raw/paired_*_cuda.jsonl`，汇总 `results/summaries/summary_20261016T074740Z.{md,csv}`）。

**完整 Attention 前+反向（训练模式）speedup vs 冻结官方实现**：

| 组 | 配置 | speedup |
|---|---|---:|
| 核心 12 | C01–C12（T=2K/8K/32K × H=1/8 × D=64/128，BF16，C=512，K=4） | 1.064–1.097 |
| FP16 | 短/中/长三代表 | 1.084–1.092 |
| 块长 | C=128 / 256 / 1024 | **1.129** / 1.118 / 1.067 |
| top-k | K=2 / K=8 | 1.084 / 1.097 |
| ragged batch | [1024,2048,5120]、8 条不均匀 | 1.089 / 1.090 |
| 尾块 | T=8193，C=512 | 1.075 |
| 冷元数据 | C08，每次新长度对象 | 1.088 |

全部 24 行几何平均 **1.088×**，min 1.064，无退化配置。**前向**单列 speedup 1.07–1.38（C08 1.36、C12 1.31、CH128 1.25）；**反向** 0.998–1.001（未改动，噪声内）。stage 级对照：官方合并阶段 3.94ms → CUDA 0.654ms（C12，6.0×）。显存：C01–C11 持平，C12 −45MB（消除了合并的 FP32 输出缓冲）。

消融链（方案 9.3）：V0 基线 → V1 加诊断标记（0.985–0.997×，仪器成本 ≤1.5%）→ V2 kernel-only（映射预建，stage 2.6–3.8×）→ V3 全路径（上表）。开发期证据：torch 建图（argsort 方案）会抵消全部收益（stage 0.52–1.54×），原子建图 kernel（0.02–0.03ms）是收益成立的必要条件。

## 5. 额外成本是什么

- 每行前向新增：槽位/workspace 分配（`slots [P,KPAD] int32`、`counts [P] int32`）、原子建图 1 kernel + 合并 1 kernel（合计 0.02–0.07ms）。已计入上述全部数字。
- 原子票据顺序不确定 → lane 级累加次序逐运行可变，输出波动 ≤1 BF16 ulp（有测试界限；官方 `index_add_` 原子序同样不确定）。
- 显存变化见上（不降反微降）。
- 首次 JIT 编译扩展 ~25 min（慢盘 + 共享机），warm cache 后 ~1 min；记入复现文档。

## 6. 哪些配置不适用

见 `docs/known_limits.md`。要点：仅 MHA、causal、dropout=0、packed varlen、D=64/128、单卡 A40；K=1 是官方全量 FA fallback 不参与稀疏加速统计；H=1/小 T 收益最小（1.064）；不推广到未测形状；收益仅在前向。

## 复现

```bash
source scripts/env.sh
python scripts/check_environment.py      # 环境证据（CPU 安全）
bash scripts/smoke.sh                    # 探针 + 官方 324 例（GPU）
python -m pytest tests/ -q               # 127 项（GPU）
python benchmarks/benchmark_attention.py --config configs/core12.yaml \
  --ids C01,C08,C12 --impl official,opt_merge_cuda --rounds 5 --samples 30 \
  --warmup 20 --out results/raw/repro.jsonl
python scripts/summarize_results.py results/raw/repro.jsonl
```

实测耗时（本机）：依赖齐备后 smoke ~2 min；测试套件 ~10s–3min（JIT 冷/热）；三配置配对 ~1 min。首次安装与编译另计：`docs/environment_day1.md`。

独立重复（代码冻结后另一次运行、seed=1）：127/127 测试通过；C01/C08/C12 前反向 speedup 1.090 / 1.092 / 1.090，与首轮（1.093 / 1.093 / 1.090）一致（`results/raw/repeat1_C01_C08_C12_cuda.jsonl`）。

## 简历口径（方案 14.1，全部来自可追溯实测）

> **MoBA 稀疏 Attention GPU 优化**
>
> - 基于 PyTorch Profiler 分阶段拆解路由、数据重排、FlashAttention 与结果合并开销，定位全局 softmax 结果合并阶段（占前向 20–28%）。
> - 使用 C++/CUDA（原子计数逆映射 + warp-per-row 合并 kernel，FP32 稳定合并，无 fast-math）替换官方 PyTorch 合并路径，保持选块集合与全局归一化语义逐位一致，反向沿用官方结构；完成 BF16/FP16、变长序列、尾块的输出与 dQ/dK/dV 验证（127 项测试，含 FP64 固定路由独立参考交叉验证）。
> - 单 A40、24 组配对配置（AB/BA、5 轮×30 样本）：完整 Attention 前向提速 1.07–1.38×、前反向合计提速 1.064–1.129×（几何平均 1.088×），反向不变，显存持平或略降，无退化配置；阶段级合并快 4.6–6.0×。

技术栈备注：Triton 版作为验证期参考实现保留（`ops/merge_triton.py`），主贡献为 CUDA 版；两次 FlashAttention 调用与官方 backward 未重写。
