import hashlib
import sys

import constriction
import numpy as np
import torch
import transformers

from .determinism import cuda_index, device_is_cuda

# Truncated SHA-256 size for environment compatibility checks.
ENV_HASH_NBYTES = 8


def env_fingerprint_string(
    *,
    dev: str,
    dtype: str,
    use_cache: bool,
    perfect: bool,
    strict_det: bool,
    force_math_sdpa: bool,
    model_id: str = "",
    revision: str = "",
    cache_policy: str = "dynamic",
    sink_tokens: int = 4,
) -> str:
    def _safe_call(fn, default="?"):
        try:
            return fn()
        except Exception:
            return default

    parts = [
        f"py={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"torch={torch.__version__}",
        f"transformers={transformers.__version__}",
        f"constr={getattr(constriction, '__version__', '?')}",
        f"numpy={np.__version__}",
        f"dev={dev}",
        f"dtype={dtype}",
        f"use_cache={int(use_cache)}",
        f"perfect={int(perfect)}",
        f"strict_det={int(strict_det)}",
        f"math_sdpa={int(force_math_sdpa)}",
        f"model_id={model_id or ''}",
        f"revision={revision or ''}",
        f"cache_policy={cache_policy}",
        f"sink_tokens={int(sink_tokens)}",
        f"tf32_matmul={int(torch.backends.cuda.matmul.allow_tf32)}",
        f"tf32_cudnn={int(torch.backends.cudnn.allow_tf32)}",
    ]
    if device_is_cuda(dev) and torch.cuda.is_available():
        idx = cuda_index(dev)
        name = _safe_call(lambda: torch.cuda.get_device_name(idx), default="unknown")
        cap = _safe_call(lambda: torch.cuda.get_device_capability(idx), default=("?", "?"))
        props = _safe_call(lambda: torch.cuda.get_device_properties(idx), default=None)
        driver = _safe_call(lambda: torch.cuda.driver_version(), default="?")
        current_idx = _safe_call(lambda: torch.cuda.current_device(), default="?")

        total_mem_mb = "?"
        sm_count = "?"
        clock_khz = "?"
        if props is not None:
            total_mem_mb = int(getattr(props, "total_memory", 0) / (1024 * 1024))
            sm_count = getattr(props, "multi_processor_count", "?")
            clock_khz = getattr(props, "memory_clock_rate", "?")

        flash_sdp = _safe_call(
            lambda: int(getattr(torch.backends.cuda, "flash_sdp_enabled")()),
            default="?",
        )
        mem_eff_sdp = _safe_call(
            lambda: int(getattr(torch.backends.cuda, "mem_efficient_sdp_enabled")()),
            default="?",
        )
        math_sdp = _safe_call(
            lambda: int(getattr(torch.backends.cuda, "math_sdp_enabled")()),
            default="?",
        )

        parts += [
            f"cuda={torch.version.cuda}",
            f"cuda_driver={driver}",
            f"cuda_current={current_idx}",
            f"cudnn={torch.backends.cudnn.version()}",
            f"gpu{idx}={name}",
            f"gpu{idx}_cc={cap[0]}.{cap[1]}",
            f"gpu{idx}_mem_mb={total_mem_mb}",
            f"gpu{idx}_sm={sm_count}",
            f"gpu{idx}_mem_clock_khz={clock_khz}",
            f"sdp_flash={flash_sdp}",
            f"sdp_mem_eff={mem_eff_sdp}",
            f"sdp_math={math_sdp}",
        ]
    return "|".join(parts)


def env_fingerprint_hash_bytes(**kwargs) -> bytes:
    # Truncated digest bytes for compact compatibility checks.
    s = env_fingerprint_string(**kwargs).encode("utf-8")
    return hashlib.sha256(s).digest()[:ENV_HASH_NBYTES]
