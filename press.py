#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CiderPress compatibility entrypoint."""

import sys

from ciderpress import app as _app
from ciderpress.cli import build_parser
from ciderpress import codec as _codec
from ciderpress.container import (
    CODING_ENUM,
    CODING_ENUM_REV,
    CONTENT_ENUM,
    CONTENT_ENUM_REV,
    DTYPE_ENUM,
    DTYPE_ENUM_REV,
    MAGIC,
    build_header_bytes,
    pack_bytes,
    pack_str,
    parse_header_bytes,
    read_container,
    unpack_bytes,
    unpack_str,
    uvarint_decode,
    uvarint_encode,
    write_container,
)
from ciderpress.determinism import canonical_device, configure_determinism, cuda_index, device_is_cuda
from ciderpress.fingerprint import ENV_HASH_NBYTES, env_fingerprint_hash_bytes, env_fingerprint_string
from ciderpress.model import LLM, LLMConfig, Stepper, load_model


def compress_tokens_sequential(llm, tokens, *, perfect, show_progress, coding_scheme=_codec.CODING_SCHEME_RANGE):
    return _codec.compress_tokens_sequential(
        llm,
        tokens,
        perfect=perfect,
        show_progress=show_progress,
        coding_scheme=coding_scheme,
        stepper_cls=Stepper,
    )


def decompress_tokens_sequential(
    llm,
    compressed_u32,
    *,
    n_tokens,
    perfect,
    show_progress,
    coding_scheme=_codec.CODING_SCHEME_RANGE,
):
    return _codec.decompress_tokens_sequential(
        llm,
        compressed_u32,
        n_tokens=n_tokens,
        perfect=perfect,
        show_progress=show_progress,
        coding_scheme=coding_scheme,
        stepper_cls=Stepper,
    )


def cmd_compress(args) -> int:
    return _app.cmd_compress(
        args,
        llm_cls=LLM,
        compress_fn=compress_tokens_sequential,
        decompress_fn=decompress_tokens_sequential,
    )


def cmd_decompress(args, argv=None) -> int:
    return _app.cmd_decompress(
        args,
        argv=argv,
        llm_cls=LLM,
        decompress_fn=decompress_tokens_sequential,
    )


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    parser = build_parser()
    args = parser.parse_args(argv[1:])

    if args.cmd == "compress":
        return cmd_compress(args)
    if args.cmd == "decompress":
        return cmd_decompress(args, argv=argv)
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
