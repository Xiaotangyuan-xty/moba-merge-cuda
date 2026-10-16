# 第 1 天环境核对（2026-10-16，gpu-node1）

实测命令输出为准；与所有者预先勘察一致。

| 项目 | 实测值 |
|---|---|
| GPU | 3× NVIDIA A40，各 46068 MiB，compute capability 8.6 |
| 核对时 GPU 状态 | 三卡显存占用 0 MiB、利用率 0%、无 compute apps（仅本次查询时刻） |
| Driver | 580.95.05 |
| 系统 nvcc | CUDA 12.9（/usr/local/cuda/bin/nvcc；默认 SSH PATH 中无 nvcc、CUDA_HOME 未设置） |
| GCC | 默认 13.3.0；另有 /usr/bin/gcc-11、g++-11 |
| Conda | 23.3.1（/home/taig/anaconda3） |
| 新环境 | /mnt/sda1/taig/envs/moba-opt，Python 3.10.22，pip 26.2.1 |
| tmux | /usr/bin/tmux 可用 |
| CPU / 内存 | 112 核；503 GB（available ~382 GB） |
| 磁盘 | /home 仅剩 7.8 GB（100% 用量，禁止新增写入）；/mnt/sda1 可用 2.0 TB |
| 网络 | GitHub/PyPI/USTC 可直连但不稳定：本次 git clone GitHub 两次 TLS/连接超时后第三次成功 |

既有环境（不修改、不复用）：venv-a40（torch 2.4.0+cu121，vllm-flash-attn 2.6.1）、PD env（torch 2.13.0+cu130，flash-attn-4 beta）。

## 已执行的安装动作

1. 创建目录：`/mnt/sda1/taig/{moba-opt,envs,cache/{conda-pkgs,moba-opt/{pip,hf,torch_ext,triton}},tmp/moba-opt}`。
2. `conda create -p /mnt/sda1/taig/envs/moba-opt python=3.10 -y`（CONDA_PKGS_DIRS 指向 sda1）。
3. clone MoonshotAI/MoBA 到 `third_party/MoBA` 并 checkout 固定 SHA `b5d5836…`（已核对 `git log -1`）。
4. torch 2.4.0、flash-attn 2.6.3 及 MoBA 其余依赖的安装记录见 `logs/` 与本文件后续补充。
