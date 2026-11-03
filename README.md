# MoBA 稀疏 Attention GPU 优化（moba-opt）

**当前功能：prefill 与增量 cached decode**。

- `opt_gqa_cuda`：原生 MHA/GQA/MQA prefill，head_dim=8..256（8 的倍数），支持前向与一阶梯度。
- `PagedKVCache` / `ContiguousKVCache`：inference-only 增量 K/V、绝对 query 位置、逐 token/多 token decode、重置与页回收；Triton Attention 直接按页表读取物理 K/V，不重建全量连续 cache。
- decode head_dim=1..256 的整数；BF16/FP16；页大小与 MoBA chunk 无需整除。
- `prepare_decode`：固定活动请求、每步一个 token 的预分配执行接口，增量页表与 GPU 长度推进；支持显式 GPU 计算路径 CUDA Graph。默认保留 finite 检查，输出工作区会复用，详见 [准备/执行与 Graph 报告](docs/prepared_decode_20261102.md)。
- prefill 阶段说明见 `docs/coverage_20261101.md`；decode API、边界和验收见 `docs/decode_paged_20261101.md`。

2026-11-03 单 A40 完整回归：**531 项通过、2 项双卡测试按授权范围跳过**。新增实验性 Qwen2.5-0.5B base 完整 24 层集成、完整 GPU 前向 Graph、固定语料质量与配对延迟测试，详见 [完整模型报告](docs/full_model_20261103.md)。

**完整模型的新结论**：dense Graph 的稳态 decode 约 4.13–4.38 ms，相对原生 HF eager FA2 为 4.28–4.61×；这是固定容量 batch=1 的 Graph/集成收益，不是 MoBA 算法收益或完整请求加速。FA2 自动 split-KV dense Graph 已快于本次真正稀疏的配置。直接替换为 MoBA topk=2/4/8 时，固定 8188 个目标的 PPL 约 63.00/24.56/14.72，dense 为 12.40；**没有无损加速结论**。packed QKV 保留为 opt-in 候选。

最新准备接口+Graph 对上一阶段 cab3 eager、相同严格检查契约，28 配置×两轮的 attend/step 几何平均分别改善 **2.017×/2.612×**。trusted 模式单列，不能与严格模式混用；仍有配置慢于 dense FA core。全部数据见准备/执行报告。

上一阶段按请求路由与 metadata 优化对冻结 cache v1 的结果为 **1.321×/1.661×**，保留在 [历史阶段报告](docs/decode_perf_20261102.md)，不与新基线混成同一数字。

下方加速数字属于历史 MHA 基线，不代表新 GQA/decode 的性能；完整模型 Graph 仅验证了实验性 Qwen2 batch=1 固定容量路径，模型服务、动态 batching、采样与 cache backward 不在当前声明范围。

## 历史 MHA 基线结果（56eedcd，不代表新 GQA/decode 的性能）

状态：完成（经一轮外部审查修订）。**主结果**：CUDA 全局 softmax 合并替换官方 PyTorch 合并路径，单 A40 上完整 Attention 前向提速 1.123–1.455×、前反向合计 1.062–1.129×（24 组配对几何平均 1.088×；统计口径：轮中位数的跨轮中位数），反向不变，峰值显存持平（独立进程复测），129 项测试全过，选块与官方逐位一致。技术报告：`docs/report.md`。

- 上游固定版本：MoonshotAI/MoBA @ `b5d58363311d3ca946f1ec444182727c15e338b5`（`third_party/UPSTREAM_SHA.txt`，third_party 不入库、零改动）
- 硬件：单张 NVIDIA A40 48GB（CC 8.6，driver 580.95.05，共享节点）
- 环境：Python 3.10 + torch 2.4.0+cu121 + flash-attn 2.6.3（源码编译）+ triton 3.0.0；锁定清单 `environment.lock.txt`；机器相关路径由 `scripts/env.local.sh` 覆盖（不入库）
- 原始数据（`results/`）不入库；汇总由 `scripts/summarize_results.py` 从原始 JSONL 重新生成

## GPU 使用纪律（共享节点）

1. 任何 GPU 运行必须先得到项目所有者当次许可。
2. 运行前检查 `nvidia-smi` 的显存占用与进程列表（不能只看利用率）。
3. 不停止他人进程；测量期间发现外部任务进入，立即停止本项目任务并保留中断记录。
4. 长任务一律在 tmux 中运行并写日志到 `logs/`。

## 目录结构

```text
third_party/MoBA/      # 冻结上游（不入库）
src/moba_opt/
  api.py               # 实现注册/统一入口（official / official_nvtx / opt_merge_atomic / opt_merge_cuda）
  baseline_adapter.py  # 冻结基线调用 + 支持范围校验
  mixed_attention.py   # 优化 autograd 路径（官方路由与反向，替换前向合并）
  grouped_attention.py # 原生 GQA/MQA prefill 与一阶反向
  cache_layout.py      # CPU 逻辑页分配、容量预检、重置和回收
  kv_cache.py          # 连续/分页 KV cache 与增量 decode API
  diagnostic.py        # 带 record_function 阶段标记的诊断副本（含路由/合并 hook）
  ops/merge.cu         # CUDA 主贡献：原子建图 + warp-per-row 全局 softmax 合并
  ops/bindings.cpp     # pybind 绑定与输入检查
  ops/merge_cuda.py    # 扩展 JIT 加载器
  ops/merge_triton.py  # Triton 参考实现（验证期；含确定性 torch 建图）
  ops/paged_decode.py  # Triton 增量写入、完整块均值、路由打分和直接分页 Attention
reference/             # FP64 固定路由/分组/decode 参考、合并数学参考、路由导出
tests/                 # prefill/梯度、合并子算子、KV 分页、decode 与错误边界
benchmarks/            # 配对计时（AB/BA）、kernel/stage 基准、分阶段 profiling
configs/               # core12 / supplementary / smoke / correctness
scripts/               # env.sh、check_environment、smoke、summarize_results
docs/                  # semantics / profiling_decision / numerical_validation / known_limits / report
results/               # 原始 JSONL、trace、fixture、汇总（不入库）
```

## 快速开始（复现三级）

```bash
source scripts/env.sh
python scripts/check_environment.py    # 1) 环境证据采集（CPU 安全）
bash scripts/gen_test_fixtures.sh      # 2) 生成测试 fixture（GPU；tests/ 依赖，不入库；缺失时相关测试干净跳过）
bash scripts/smoke.sh                  # 3) 接口探针 + 官方 324 例（GPU）
python -m pytest tests/ -q             #    本项目测试（GPU）
# 4) 核心性能复现（GPU）
python benchmarks/benchmark_attention.py --config configs/core12.yaml \
  --ids C01,C02,C03,C04,C05,C06,C07,C08,C09,C10,C11,C12 \
  --impl official,opt_merge_cuda --rounds 5 --samples 30 --warmup 20 \
  --out results/raw/repro_core12.jsonl
python scripts/summarize_results.py results/raw/repro_core12.jsonl
```

首次环境安装与扩展编译耗时见 `docs/environment_day1.md` / `docs/report.md`「复现」节。
