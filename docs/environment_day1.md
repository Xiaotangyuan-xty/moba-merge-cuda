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

## 安装结果（2026-10-16 完成）

| 步骤 | 结果 | 耗时/日志 |
|---|---|---|
| torch 2.4.0（PyPI，cu121，ABI=False） | 成功 | ~8.5 min，USTC 镜像，logs/pip_torch.log |
| numpy einops transformers accelerate pyyaml pytest | 成功 | logs/pip_deps.log |
| transformers 版本钉死 4.49.0 | 必要修复 | 最新版要求 torch>=2.6，导致 `import moba` 失败 |
| flash-attn 2.6.3 | **源码编译成功** | ~23 min，logs/pip_fa_build2.log |

flash-attn 安装经过：v2.6.3 release 只发布 cu118 wheel；setup.py 按本机 CUDA 猜的 cu123+torch2.4 wheel URL 不存在（下载失败）。改用 `FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=32 TORCH_CUDA_ARCH_LIST=8.6` 以系统 nvcc 12.9 + gcc 13.3 源码编译，torch 侧为 cu121（同为 CUDA 12 major，通过 FA 版本检查）。

最终环境：python 3.10.22 / torch 2.4.0+cu121 / flash_attn 2.6.3 / triton 3.0.0 / einops 0.8.2 / transformers 4.49.0 / accelerate 1.15.0。完整依赖：`environment.lock.txt`；环境证据：`results/environment/env_20261016T040301Z.json`。

已验证（CPU 侧，未初始化 CUDA）：`import moba` 正常；`_flash_attn_varlen_forward/backward` 签名与 MoBA 调用参数匹配。

注意：首次 import torch/flash_attn 约需 2–4 分钟（sda1 机械盘加载大型 .so），后续走页缓存应显著加快；计时实验不受影响但脚本启动要预留时间。

## 待 GPU 许可后执行（尚未运行）

1. `python scripts/probe_interfaces.py`（方案 6.2 探针）
2. `python -m pytest -q tests/test_moba_attn.py`（官方正确性，third_party/MoBA 下）
3. `python benchmarks/benchmark_attention.py --config configs/core12.yaml --ids C01,C08,C12 ...`（基线计时）
