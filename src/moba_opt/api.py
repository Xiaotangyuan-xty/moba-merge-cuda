"""Unified implementation dispatch for tests and benchmarks (plan section 7.3 run_impl)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MOBA_DIR = _REPO_ROOT / "third_party" / "MoBA"
if _MOBA_DIR.is_dir() and str(_MOBA_DIR) not in sys.path:
    sys.path.insert(0, str(_MOBA_DIR))

_IMPLS = {}


def register(name, fn):
    if name in _IMPLS:
        raise KeyError(f"impl already registered: {name}")
    _IMPLS[name] = fn


def available_impls():
    return sorted(_IMPLS)


def run_impl(name, q, k, v, cu_seqlens, max_seqlen, config):
    if name not in _IMPLS:
        raise KeyError(f"unknown impl {name!r}; available: {available_impls()}")
    return _IMPLS[name](q, k, v, cu_seqlens, max_seqlen, config)
