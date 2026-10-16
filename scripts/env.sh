# Source this before any moba-opt work on gpu-node1:
#   source scripts/env.sh
export MOBA_OPT_ROOT=/mnt/sda1/taig/moba-opt
export MOBA_OPT_ENV=/mnt/sda1/taig/envs/moba-opt
export PATH="$MOBA_OPT_ENV/bin:$PATH"
export PYTHONPATH="$MOBA_OPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# CUDA toolchain (system nvcc 12.9). Default SSH env has neither CUDA_HOME nor nvcc in PATH.
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# /home has only ~7.8G free: keep every cache on the data disk.
export CONDA_PKGS_DIRS=/mnt/sda1/taig/cache/conda-pkgs
export PIP_CACHE_DIR=/mnt/sda1/taig/cache/moba-opt/pip
export HF_HOME=/mnt/sda1/taig/cache/moba-opt/hf
export TORCH_EXTENSIONS_DIR=/mnt/sda1/taig/cache/moba-opt/torch_ext
export TRITON_CACHE_DIR=/mnt/sda1/taig/cache/moba-opt/triton
export TMPDIR=/mnt/sda1/taig/tmp/moba-opt
