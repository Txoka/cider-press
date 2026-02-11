"""CiderPress package."""

from .app import cmd_compress, cmd_decompress
from .cli import (
    COMMON_ARGUMENTS,
    COMPRESS_ARGUMENTS,
    DECOMPRESS_ARGUMENTS,
    add_common_arguments,
    build_parser,
    main,
)

__all__ = [
    "COMMON_ARGUMENTS",
    "COMPRESS_ARGUMENTS",
    "DECOMPRESS_ARGUMENTS",
    "add_common_arguments",
    "build_parser",
    "cmd_compress",
    "cmd_decompress",
    "main",
]
