import os
import random
from typing import Optional

import numpy as np
import torch


def canonical_device(dev: Optional[str]) -> str:
    if dev is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = dev.strip().lower()
    if dev == "cuda":
        return "cuda:0"
    if dev.startswith("cuda:"):
        try:
            int(dev.split(":", 1)[1])
        except Exception as e:
            raise ValueError(f"Bad --device '{dev}': {e}")
        return dev
    if dev == "cpu":
        return "cpu"
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'.")


def device_is_cuda(dev: str) -> bool:
    return dev.startswith("cuda")


def cuda_index(dev: str) -> int:
    if not device_is_cuda(dev):
        return 0
    return int(dev.split(":", 1)[1]) if ":" in dev else 0


def configure_determinism(seed: int, dev: str, strict: bool, force_math_sdpa: bool) -> None:
    # Best-effort: should be set before CUDA context / kernels are chosen.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_is_cuda(dev) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Disable TF32 (can change numerics across runs/paths)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    # Reduce silent kernel drift in attention
    if force_math_sdpa and device_is_cuda(dev):
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    if strict:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)
