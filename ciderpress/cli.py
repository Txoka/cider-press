import argparse
import sys
from typing import Dict, Iterable, List, Tuple

from .app import cmd_compress, cmd_decompress
from .fingerprint import ENV_HASH_NBYTES

# Exposed CLI argument definitions for programmatic reuse.
# Each item is (flag, kwargs for argparse.add_argument).
COMMON_ARGUMENTS: List[Tuple[str, Dict]] = [
    ("--model", {"default": "Qwen/Qwen2.5-0.5B", "help": "HF model id"}),
    ("--revision", {"default": None, "help": "HF revision/commit (optional)"}),
    ("--trust-remote-code", {"action": "store_true"}),
    ("--device", {"default": "cuda", "help": "cpu | cuda | cuda:N"}),
    ("--dtype", {"default": "bf16", "choices": [None, "fp32", "bf16", "fp16"]}),
    ("--max-ctx", {"type": int, "default": 0}),
    ("--no-cache", {"action": "store_true"}),
    ("--perfect", {"action": "store_true"}),
    ("--strict-determinism", {"action": "store_true"}),
    (
        "--determinism",
        {
            "choices": ["off", "on"],
            "default": "off",
            "help": "Enable/disable deterministic runtime configuration (default: off).",
        },
    ),
    (
        "--force-math-sdpa",
        {
            "action": "store_true",
            "help": "Disable flash/mem-efficient SDPA; use math SDPA for stability.",
        },
    ),
    (
        "--no-progress",
        {"action": "store_true", "help": "Disable tqdm progress bars (useful for logs / piping)."},
    ),
    (
        "--coding",
        {
            "choices": ["range", "ans"],
            "default": None,
            "help": "Entropy coding scheme. Compress defaults to range; decompress reads the file header unless overridden.",
        },
    ),
    (
        "--cache-policy",
        {
            "choices": ["dynamic", "streaming_llm"],
            "default": None,
            "help": "KV-cache policy. Decompress reads the file header unless overridden.",
        },
    ),
    (
        "--sink-tokens",
        {"type": int, "default": 4, "help": "Number of initial attention-sink tokens for --cache-policy streaming_llm."},
    ),
    ("--no-store-model", {"action": "store_true", "help": "Do not store model id in header."}),
    ("--no-store-revision", {"action": "store_true", "help": "Do not store revision in header."}),
    (
        "--no-store-envhash",
        {
            "action": "store_true",
            "help": f"Do not store env hash digest bytes ({ENV_HASH_NBYTES}B by default).",
        },
    ),
]

COMPRESS_ARGUMENTS: List[Tuple[str, Dict]] = [
    ("--in", {"dest": "inp", "required": True, "help": "Input text file or directory of UTF-8 text files."}),
    ("--out", {"dest": "out", "required": True}),
    ("--encoding", {"default": "utf-8", "help": "Input text encoding (default utf-8)."}),
    ("--verify", {"action": "store_true", "help": "Decompress immediately and verify byte-equality."}),
]

DECOMPRESS_ARGUMENTS: List[Tuple[str, Dict]] = [
    ("--in", {"dest": "inp", "required": True}),
    ("--out", {"dest": "out", "required": True}),
    ("--force", {"action": "store_true", "help": "Ignore env-hash mismatch and attempt decode anyway."}),
]


def add_arguments(parser: argparse.ArgumentParser, specs: Iterable[Tuple[str, Dict]]) -> None:
    for flag, kwargs in specs:
        parser.add_argument(flag, **kwargs)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    add_arguments(parser, COMMON_ARGUMENTS)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CiderPress: LLM-guided compression with constriction (LLMCF5 format; env hash stored as bytes)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compress")
    add_common_arguments(pc)
    add_arguments(pc, COMPRESS_ARGUMENTS)

    pd = sub.add_parser("decompress")
    add_common_arguments(pd)
    add_arguments(pd, DECOMPRESS_ARGUMENTS)

    return ap


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
