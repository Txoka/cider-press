from dataclasses import dataclass
import os
from typing import List, Optional

import numpy as np
import torch


def _disable_optional_torchvision() -> None:
    if os.environ.get("CIDERPRESS_DISABLE_TORCHVISION", "1").lower() in {"0", "false", "no"}:
        return
    try:
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    import_utils._torchvision_available = False
    import_utils._torchvision_version = "N/A"


_disable_optional_torchvision()

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMConfig:
    model_id: str
    revision: Optional[str]
    device: str
    dtype: str
    max_ctx: int
    use_cache: bool
    trust_remote_code: bool


def load_model(model_id: str, revision: Optional[str], dtype: torch.dtype, trust_remote_code: bool):
    # HF API drift: try dtype=..., fall back torch_dtype=...
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            use_fast=True,
            trust_remote_code=cfg.trust_remote_code,
        )

        torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.dtype]
        self.model = load_model(cfg.model_id, cfg.revision, torch_dtype, cfg.trust_remote_code).eval().to(cfg.device)

        self.vocab_size = int(self.model.get_input_embeddings().num_embeddings)

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        if bos is None and eos is None:
            raise ValueError("Tokenizer has neither bos_token_id nor eos_token_id; need a start token.")
        self.start_id = int(bos) if bos is not None else int(eos)

        if cfg.max_ctx > 0:
            self.max_ctx = int(cfg.max_ctx)
        else:
            m = getattr(self.model.config, "max_position_embeddings", None)
            self.max_ctx = int(m) if m is not None else 2048

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=False)


class Stepper:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.device = llm.cfg.device
        self.use_cache = llm.cfg.use_cache
        self.ctx: List[int] = [llm.start_id]
        self.past = None

        x0 = torch.tensor([[llm.start_id]], dtype=torch.long, device=self.device)
        out0 = self._forward(input_ids=x0, use_cache=self.use_cache)
        self.logits_next = out0.logits[0, -1, :]
        self.past = out0.past_key_values if self.use_cache else None

    def _forward(self, **kwargs):
        try:
            return self.llm.model(logits_to_keep=1, **kwargs)
        except TypeError:
            try:
                return self.llm.model(num_logits_to_keep=1, **kwargs)
            except TypeError:
                return self.llm.model(**kwargs)

    @torch.inference_mode()
    def step(self, token: int) -> None:
        self.ctx.append(int(token))

        if (not self.use_cache) or (self.past is None) or (len(self.ctx) > self.llm.max_ctx):
            self.ctx = self.ctx[-self.llm.max_ctx:]
            x = torch.tensor([self.ctx], dtype=torch.long, device=self.device)
            out = self._forward(input_ids=x, use_cache=self.use_cache)
            self.logits_next = out.logits[0, -1, :]
            self.past = out.past_key_values if self.use_cache else None
            return

        x = torch.tensor([[int(token)]], dtype=torch.long, device=self.device)
        out = self._forward(input_ids=x, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self.logits_next = out.logits[0, -1, :]

    @torch.inference_mode()
    def probs_np(self) -> np.ndarray:
        probs = torch.softmax(self.logits_next.float(), dim=-1)  # fp32 softmax
        return probs.detach().cpu().numpy().astype(np.float32, copy=False)[None, :]
