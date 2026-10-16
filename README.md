# MoBA 稀疏 Attention GPU 优化（moba-opt）

状态：第 1 周，环境搭建阶段。尚无任何性能结果。

- 上游固定版本：MoonshotAI/MoBA @ `b5d58363311d3ca946f1ec444182727c15e338b5`
  （见 `third_party/UPSTREAM_SHA.txt`，third_party 不入库）
- 机器：共享节点 gpu-node1，3× NVIDIA A40 48GB（CC 8.6，driver 580.95.05）
- 环境：conda 独立环境 `/mnt/sda1/taig/envs/moba-opt`（Python 3.10 + torch 2.4.0 + flash-attn 2.6.3）
- 所有代码、原始数据、报告均保存在服务器 `/mnt/sda1/taig/moba-opt/`，git 仅本地

## GPU 使用纪律（共享节点）

1. 任何 GPU 运行必须先得到项目所有者当次许可。
2. 运行前检查 `nvidia-smi` 的显存占用与进程列表（不能只看利用率）。
3. 不停止他人进程；测量期间发现外部任务进入，立即停止本项目任务并保留中断记录。
4. 长任务一律在 tmux 中运行并写日志到 `logs/`。

## 快速开始

```bash
source scripts/env.sh
python scripts/check_environment.py   # 仅采集信息，不初始化 CUDA 上下文
bash scripts/smoke.sh                 # GPU-gated：接口探针 + 官方正确性测试
```

## 目录

见方案文档第 11 节；`docs/semantics.md` 为源码语义记录，`docs/environment_day1.md` 为第 1 天环境核对结果。
