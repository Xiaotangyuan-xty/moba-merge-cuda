# Source this before any moba-opt work:
#   source scripts/env.sh
# Machine-specific overrides live in scripts/env.local.sh (gitignored).

export MOBA_OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Machine-specific overrides first (env paths, cache dirs, CUDA_HOME, ...)
if [[ -f "$MOBA_OPT_ROOT/scripts/env.local.sh" ]]; then
  source "$MOBA_OPT_ROOT/scripts/env.local.sh"
fi

# Conda environment: absolute prefix path (e.g. /path/to/envs/moba-opt).
export MOBA_OPT_ENV="${MOBA_OPT_ENV:-$MOBA_OPT_ROOT/.venv}"
export PATH="$MOBA_OPT_ENV/bin:$PATH"
export PYTHONPATH="$MOBA_OPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# CUDA toolchain (needed only for source builds of flash-attn / the extension).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

# Caches default inside the project; override via env.local.sh on machines
# with a small home directory.
export MOBA_OPT_CACHE="${MOBA_OPT_CACHE:-$MOBA_OPT_ROOT/.cache}"
export PIP_CACHE_DIR="$MOBA_OPT_CACHE/pip"
export HF_HOME="$MOBA_OPT_CACHE/hf"
export TORCH_EXTENSIONS_DIR="$MOBA_OPT_CACHE/torch_ext"
export TRITON_CACHE_DIR="$MOBA_OPT_CACHE/triton"
export TMPDIR="${TMPDIR:-$MOBA_OPT_CACHE/tmp}"
mkdir -p "$TMPDIR"
