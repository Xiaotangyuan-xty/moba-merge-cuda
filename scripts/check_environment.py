#!/usr/bin/env python
"""Collect environment evidence (plan section 6.4) into results/environment/.

CPU-safe: imports and version queries only; never initializes a CUDA context
(nvidia-smi and NVML-based queries are monitoring, not GPU compute).
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc}>"


def main():
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": {"version": sys.version, "executable": sys.executable},
        "env_vars": {k: os.environ.get(k) for k in (
            "CUDA_HOME", "CONDA_PKGS_DIRS", "PIP_CACHE_DIR", "HF_HOME",
            "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TMPDIR",
        )},
        "nvcc": sh("/usr/local/cuda/bin/nvcc --version | tail -n2"),
        "gcc": sh("gcc --version | head -n1"),
        "g++": sh("g++ --version | head -n1"),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1"),
        "gpus": sh("nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader"),
        "gpu_processes": sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"),
        "upstream_sha_file": (ROOT / "third_party" / "UPSTREAM_SHA.txt").read_text().strip()
        if (ROOT / "third_party" / "UPSTREAM_SHA.txt").exists() else None,
        "upstream_head": sh(f"git -C {ROOT}/third_party/MoBA rev-parse HEAD"),
        "repo_head": sh(f"git -C {ROOT} rev-parse HEAD"),
        "repo_uncommitted": sh(f"git -C {ROOT} status --porcelain"),
    }

    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cxx11_abi": torch.compiled_with_cxx11_abi(),
            "cuda_available_nvml": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"<import failed: {exc}>"

    for mod in ("flash_attn", "triton", "einops", "transformers", "accelerate"):
        try:
            m = __import__(mod)
            info.setdefault("modules", {})[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            info.setdefault("modules", {})[mod] = f"<import failed: {exc}>"

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    info["pip_freeze"] = freeze.stdout.strip().splitlines()

    out_dir = ROOT / "results" / "environment"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"env_{ts}.json"
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    (ROOT / "environment.lock.txt").write_text(freeze.stdout)

    print(f"wrote {out}")
    print(f"wrote {ROOT / 'environment.lock.txt'}")
    for key in ("driver", "nvcc", "gcc"):
        print(f"{key}: {info[key].splitlines()[0] if info[key] else '?'}")
    if isinstance(info.get("torch"), dict):
        t = info["torch"]
        print(f"torch {t['version']} cuda={t['cuda_build']} abi={t['cxx11_abi']} devices={t['device_count']}")
    print(f"modules: {info.get('modules')}")


if __name__ == "__main__":
    main()
