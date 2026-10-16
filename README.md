# MoBA 稀疏 Attention GPU 优化（moba-opt）

状态：完成（经一轮外部审查修订）。**主结果**：CUDA 全局 softmax 合并替换官方 PyTorch 合并路径，单 A40 上完整 Attention 前向提速 1.123–1.455×、前反向合计 1.062–1.129×（24 组配对几何平均 1.088×；统计口径：轮中位数的跨轮中位数），反向不变，峰值显存持平（独立进程复测），129 项测试全过，选块与官方逐位一致。技术报告：`docs/report.md`。

- 上游固定版本：MoonshotAI/MoBA @ `b5d58363311d3ca946f1ec444182727c15e338b5`（`third_party/UPSTREAM_SHA.txt`，third_party 不入库、零改动）
- 机器：共享节点 gpu-node1，3× NVIDIA A40 48GB（CC 8.6，driver 580.95.05）
- 环境：conda `/mnt/sda1/taig/envs/moba-opt`（Python 3.10 + torch 2.4.0+cu121 + flash-attn 2.6.3 源码编译 + triton 3.0.0）；锁定清单 `environment.lock.txt`
- 所有代码、原始数据、报告均保存在服务器 `/mnt/sda1/taig/moba-opt/`，git 仅本地

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
  diagnostic.py        # 带 record_function 阶段标记的诊断副本（含路由/合并 hook）
  ops/merge.cu         # CUDA 主贡献：原子建图 + warp-per-row 全局 softmax 合并
  ops/bindings.cpp     # pybind 绑定与输入检查
  ops/merge_cuda.py    # 扩展 JIT 加载器
  ops/merge_triton.py  # Triton 参考实现（验证期；含确定性 torch 建图）
reference/             # FP64 固定路由参考、合并数学参考、路由导出
tests/                 # 127 项：参考自验证 / 子算子 / fixture 交叉 / 完整路径（两个后端）
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
