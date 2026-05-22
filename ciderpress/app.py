import hashlib
import inspect
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

from .codec import compress_tokens_sequential, decompress_tokens_sequential
from .container import MAGIC, build_header_bytes, read_container, uvarint_encode, write_container
from .determinism import canonical_device, configure_determinism, device_is_cuda
from .fingerprint import env_fingerprint_hash_bytes
from .model import LLM, LLMConfig, Stepper
from .text_archive import build_text_archive, extract_text_archive

DEFAULT_SEED = 0


def _call_with_optional_coding(fn, *args, coding_scheme: str, **kwargs):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is not None and "coding_scheme" in sig.parameters:
        kwargs["coding_scheme"] = coding_scheme
    return fn(*args, **kwargs)


def resolve_dtype(dev: str, dtype: Optional[str]) -> str:
    if dtype is None:
        dtype = "bf16" if device_is_cuda(dev) else "fp32"
    if dev == "cpu" and dtype != "fp32":
        print("Warning: forcing fp32 on CPU.", file=sys.stderr)
        dtype = "fp32"
    return dtype


def _read_input_text(path: str, encoding: str) -> Tuple[str, bytes, str, str]:
    if os.path.isdir(path):
        text, n_files, n_dirs, skipped = build_text_archive(path)
        raw = text.encode("utf-8")
        summary = f"text archive: {n_files} files, {n_dirs} dirs, {skipped} skipped"
        return text, raw, "text_archive", summary

    raw = open(path, "rb").read()
    try:
        text = raw.decode(encoding)
    except Exception as e:
        raise SystemExit(f"Failed to decode input as {encoding}: {e}")
    return text, raw, "text", "text file"


def cmd_compress(
    args,
    *,
    llm_cls=LLM,
    compress_fn=compress_tokens_sequential,
    decompress_fn=decompress_tokens_sequential,
) -> int:
    dev = canonical_device(args.device)
    dtype = resolve_dtype(dev, args.dtype)
    show_progress = (not args.no_progress)
    coding_scheme = getattr(args, "coding", None) or "range"

    if args.determinism == "on":
        configure_determinism(DEFAULT_SEED, dev, strict=args.strict_determinism, force_math_sdpa=args.force_math_sdpa)

    text, raw, content_kind, input_summary = _read_input_text(args.inp, args.encoding)

    llm = llm_cls(LLMConfig(
        model_id=args.model,
        revision=args.revision,
        device=dev,
        dtype=dtype,
        max_ctx=args.max_ctx,
        use_cache=(not args.no_cache),
        trust_remote_code=args.trust_remote_code,
    ))

    tokens = llm.encode_text(text)
    compressed_u32, avg_nll_bits = _call_with_optional_coding(
        compress_fn,
        llm,
        tokens,
        perfect=bool(args.perfect),
        show_progress=show_progress,
        coding_scheme=coding_scheme,
    )
    payload = compressed_u32.tobytes(order="C")

    env_hash: Optional[bytes] = None
    if not args.no_store_envhash:
        env_hash = env_fingerprint_hash_bytes(
            dev=dev,
            dtype=dtype,
            use_cache=(not args.no_cache),
            perfect=bool(args.perfect),
            strict_det=bool(args.strict_determinism),
            force_math_sdpa=bool(args.force_math_sdpa),
            model_id=args.model,
            revision=args.revision or "",
        )

    header_bytes = build_header_bytes(
        model_id=None if args.no_store_model else args.model,
        revision=None if args.no_store_revision else args.revision,
        max_ctx=int(llm.max_ctx),
        n_tokens=len(tokens),
        dev=dev,
        dtype=dtype,
        use_cache=(not args.no_cache),
        perfect=bool(args.perfect),
        strict_det=bool(args.strict_determinism),
        trust_remote_code=bool(args.trust_remote_code),
        force_math_sdpa=bool(args.force_math_sdpa),
        env_hash=env_hash,
        coding_scheme=coding_scheme,
        content_kind=content_kind,
    )

    write_container(args.out, header_bytes, payload)

    in_bytes = len(raw)
    out_bytes = os.path.getsize(args.out)
    header_total = len(MAGIC) + len(uvarint_encode(len(header_bytes))) + len(header_bytes)
    payload_bytes = len(payload)

    bits_per_token_total = (out_bytes * 8.0) / max(1, len(tokens))
    bits_per_token_payload = (payload_bytes * 8.0) / max(1, len(tokens))
    bits_per_byte_total = (out_bytes * 8.0) / max(1, in_bytes)
    bits_per_byte_payload = (payload_bytes * 8.0) / max(1, in_bytes)

    print(f"Compressed: {in_bytes} -> {out_bytes} bytes (ratio {out_bytes / max(1, in_bytes):.6f})")
    print(f"Header bytes: {header_total} | Payload bytes: {payload_bytes}")
    print(f"Input: {input_summary}")
    print(f"Tokens: {len(tokens)}, vocab={llm.vocab_size}, device={dev}, dtype={dtype}, perfect={bool(args.perfect)}, coding={coding_scheme}")
    print(f"bits/token TOTAL:   {bits_per_token_total:.6f}")
    print(f"bits/token PAYLOAD: {bits_per_token_payload:.6f}")
    print(f"bits/byte  TOTAL:   {bits_per_byte_total:.6f}")
    print(f"bits/byte  PAYLOAD: {bits_per_byte_payload:.6f}")
    print(f"Model NLL bits/token (encoded probs): {avg_nll_bits:.6f}")
    print(f"Overhead vs NLL (PAYLOAD): {bits_per_token_payload - avg_nll_bits:.6f} bits/token")

    if args.verify:
        header, payload2, _ = read_container(args.out)
        n_tokens = int(header["n_tokens"])
        if (len(payload2) % 4) != 0:
            raise SystemExit(f"Corrupt payload length {len(payload2)}; expected multiple of 4 bytes for uint32 stream.")
        n_words = len(payload2) // 4
        comp_u32 = np.frombuffer(payload2, dtype=np.uint32, count=n_words)
        dec_tokens = _call_with_optional_coding(
            decompress_fn,
            llm,
            comp_u32,
            n_tokens=n_tokens,
            perfect=bool(header["perfect"]),
            show_progress=show_progress,
            coding_scheme=str(header.get("coding_scheme", "range")),
        )
        dec_text = llm.decode_tokens(dec_tokens)

        if content_kind == "text_archive":
            dec_raw = dec_text.encode("utf-8")
        else:
            dec_raw = dec_text.encode(args.encoding)

        if hashlib.md5(dec_raw).hexdigest() != hashlib.md5(raw).hexdigest():
            raise SystemExit("[VERIFY FAIL] Roundtrip payload does not match original.")
        print("[VERIFY OK] Roundtrip bytes match original.")

    return 0


def cmd_decompress(args, argv=None, *, llm_cls=LLM, decompress_fn=decompress_tokens_sequential) -> int:
    if argv is None:
        argv = sys.argv

    dev = canonical_device(args.device)
    dtype = resolve_dtype(dev, args.dtype)
    show_progress = (not args.no_progress)

    header, payload, header_total = read_container(args.inp)
    model_id = header.get("model") or args.model
    revision = header.get("revision") or args.revision
    coding_scheme = getattr(args, "coding", None) or str(header.get("coding_scheme", "range"))
    content_kind = str(header.get("content_kind", "text"))

    dev_used = dev if "--device" in argv else header["device"]
    dtype_used = dtype if "--dtype" in argv else header["dtype"]

    if device_is_cuda(dev_used) and not torch.cuda.is_available():
        raise SystemExit(f"Header requires {dev_used} but CUDA is not available on this machine.")

    stored_env: Optional[bytes] = header.get("env_hash") or None
    if stored_env is not None and not args.force:
        use_model_identity = int(header.get("version", 0)) >= 4
        computed_env = env_fingerprint_hash_bytes(
            dev=dev_used,
            dtype=dtype_used,
            use_cache=bool(header["use_cache"]) if "--no-cache" not in argv else (not args.no_cache),
            perfect=bool(header["perfect"]) if "--perfect" not in argv else bool(args.perfect),
            strict_det=bool(header["strict_det"]) if "--strict-determinism" not in argv else bool(args.strict_determinism),
            force_math_sdpa=bool(header["force_math_sdpa"]) if "--force-math-sdpa" not in argv else bool(args.force_math_sdpa),
            model_id=(model_id or "") if use_model_identity else "",
            revision=(revision or "") if use_model_identity else "",
        )
        if computed_env != stored_env:
            print("Env-hash mismatch (truncated sha256 digest).")
            print(f"  stored:   {stored_env.hex()}")
            print(f"  computed: {computed_env.hex()}")
            print("Refusing to decode without --force (likely to fail anyway).")
            raise SystemExit(2)

    if args.determinism == "on":
        configure_determinism(
            DEFAULT_SEED,
            dev_used,
            strict=bool(header["strict_det"]) if "--strict-determinism" not in argv else bool(args.strict_determinism),
            force_math_sdpa=bool(header["force_math_sdpa"]) if "--force-math-sdpa" not in argv else bool(args.force_math_sdpa),
        )

    llm = llm_cls(LLMConfig(
        model_id=model_id,
        revision=revision,
        device=dev_used,
        dtype=dtype_used,
        max_ctx=int(header.get("max_ctx", 0)) if args.max_ctx == 0 else args.max_ctx,
        use_cache=bool(header.get("use_cache", True)) if "--no-cache" not in argv else (not args.no_cache),
        trust_remote_code=args.trust_remote_code or bool(header.get("trust_remote_code", False)),
    ))

    n_tokens = int(header["n_tokens"])
    if (len(payload) % 4) != 0:
        raise SystemExit(f"Corrupt payload length {len(payload)}; expected multiple of 4 bytes for uint32 stream.")
    n_words = len(payload) // 4
    comp_u32 = np.frombuffer(payload, dtype=np.uint32, count=n_words)

    out_tokens = _call_with_optional_coding(
        decompress_fn,
        llm,
        comp_u32,
        n_tokens=n_tokens,
        perfect=bool(header["perfect"]),
        show_progress=show_progress,
        coding_scheme=coding_scheme,
    )
    out_text = llm.decode_tokens(out_tokens)

    if content_kind == "text_archive":
        n_files, n_dirs = extract_text_archive(out_text, args.out)
        print(f"Decompressed {n_tokens} tokens to {args.out} ({n_files} files, {n_dirs} dirs)")
    elif content_kind == "text":
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"Decompressed {n_tokens} tokens to {args.out}")
    else:
        raise SystemExit(f"Unsupported content kind: {content_kind}")

    print(f"Header bytes: {header_total} | Payload bytes: {len(payload)} | Coding: {coding_scheme} | Content: {content_kind}")
    return 0
