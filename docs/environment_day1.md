# 第 1 天环境核对（2026-10-16，共享 A40 节点）

实测命令输出为准；与所有者预先勘察一致。

| 项目 | 实测值 |
|---|---|
| GPU | 3× NVIDIA A40，各 46068 MiB，compute capability 8.6 |
| 核对时 GPU 状态 | 三卡显存占用 0 MiB、利用率 0%、无 compute apps（仅本次查询时刻） |
| Driver | 580.95.05 |
| 系统 nvcc | CUDA 12.9（/usr/local/cuda/bin/nvcc；默认 SSH PATH 中无 nvcc、CUDA_HOME 未设置） |
| GCC | 默认 13.3.0；另有 gcc-11/g++-11 |
| Conda | 23.3.1 |
| 新环境 | 独立 conda 前缀环境 `moba-opt`，Python 3.10.22，pip 26.2.1 |
| tmux | 可用 |
| CPU / 内存 | 112 核；503 GB（available ~382 GB） |
| 磁盘 | 系统盘 home 仅剩 7.8 GB（禁止新增写入）；数据盘可用 2.0 TB |
| 网络 | GitHub/PyPI/USTC 可直连但不稳定：本次 git clone GitHub 两次 TLS/连接超时后第三次成功 |

既有环境（不修改、不复用）：一个 torch 2.4.0+cu121 + vllm-flash-attn 2.6.1 环境、一个 torch 2.13.0+cu130 + flash-attn-4 beta 环境。

## 已执行的安装动作

1. 在数据盘创建目录：项目仓库 / conda 环境 / 下载与编译缓存 / 临时编译目录。
2. `conda create -p <envs>/moba-opt python=3.10 -y`（CONDA_PKGS_DIRS 指向数据盘）。
3. clone MoonshotAI/MoBA 到 `third_party/MoBA` 并 checkout 固定 SHA `b5d5836…`（已核对 `git log -1`）。
4. torch 2.4.0、flash-attn 2.6.3 及 MoBA 其余依赖的安装记录见下节。

## 安装结果（2026-10-16 完成）

| 步骤 | 结果 | 耗时 |
|---|---|---|
| torch 2.4.0（PyPI，cu121，ABI=False） | 成功 | ~8.5 min，USTC 镜像 |
| numpy einops transformers accelerate pyyaml pytest | 成功 | — |
| transformers 版本钉死 4.49.0 | 必要修复 | 最新版要求 torch>=2.6，导致 `import moba` 失败 |
| flash-attn 2.6.3 | **源码编译成功** | ~23 min |

flash-attn 安装经过：v2.6.3 release 只发布 cu118 wheel；setup.py 按本机 CUDA 猜的 cu123+torch2.4 wheel URL 不存在（下载失败）。改用 `FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=32 TORCH_CUDA_ARCH_LIST=8.6` 以系统 nvcc 12.9 + gcc 13.3 源码编译，torch 侧为 cu121（同为 CUDA 12 major，通过 FA 版本检查）。

最终环境：python 3.10.22 / torch 2.4.0+cu121 / flash_attn 2.6.3 / triton 3.0.0 / einops 0.8.2 / transformers 4.49.0 / accelerate 1.15.0。完整依赖：`environment.lock.txt`。

已验证（CPU 侧，未初始化 CUDA）：`import moba` 正常；`_flash_attn_varlen_forward/backward` 签名与 MoBA 调用参数匹配。

注意：该节点数据盘为机械盘，首次 import torch/flash_attn 约需 2–4 分钟（加载大型 .so），后续走页缓存显著加快；计时实验不受影响但脚本启动要预留时间。
