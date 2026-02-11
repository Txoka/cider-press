from typing import List, Tuple

import constriction
import numpy as np
import torch

from .model import LLM, Stepper


def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:  # pragma: no cover
        return lambda x, **kwargs: x


@torch.inference_mode()
def compress_tokens_sequential(
    llm: LLM,
    tokens: List[int],
    *,
    perfect: bool,
    show_progress: bool,
    stepper_cls=Stepper,
) -> Tuple[np.ndarray, float]:
    if len(tokens) == 0:
        return np.zeros((0,), dtype=np.uint32), 0.0

    tqdm = _get_tqdm()
    model_family = constriction.stream.model.Categorical(perfect=perfect)
    encoder = constriction.stream.queue.RangeEncoder()
    stepper = stepper_cls(llm)

    total_nll_bits = 0.0
    it = tqdm(tokens, desc="encode", unit="tok", disable=(not show_progress))
    for t in it:
        probs_np = stepper.probs_np()
        p = float(probs_np[0, int(t)])
        if p < 1e-30:
            p = 1e-30
        total_nll_bits += -np.log2(p)

        encoder.encode(np.asarray([int(t)], dtype=np.int32), model_family, probs_np)
        stepper.step(int(t))

    compressed_u32 = encoder.get_compressed()
    avg_nll = float(total_nll_bits / max(1, len(tokens)))
    return compressed_u32, avg_nll


@torch.inference_mode()
def decompress_tokens_sequential(
    llm: LLM,
    compressed_u32: np.ndarray,
    *,
    n_tokens: int,
    perfect: bool,
    show_progress: bool,
    stepper_cls=Stepper,
) -> List[int]:
    if n_tokens == 0:
        return []

    tqdm = _get_tqdm()
    model_family = constriction.stream.model.Categorical(perfect=perfect)
    decoder = constriction.stream.queue.RangeDecoder(compressed_u32)
    stepper = stepper_cls(llm)

    out_tokens: List[int] = []
    it = tqdm(range(n_tokens), desc="decode", unit="tok", disable=(not show_progress))
    for _ in it:
        probs_np = stepper.probs_np()
        sym = decoder.decode(model_family, probs_np)
        t = int(sym[0])
        out_tokens.append(t)
        stepper.step(t)

    return out_tokens
