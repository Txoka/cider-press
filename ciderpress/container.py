from typing import Optional, Tuple

from .determinism import cuda_index, device_is_cuda

MAGIC = b"LMCF"


def uvarint_encode(n: int) -> bytes:
    if n < 0:
        raise ValueError("uvarint requires non-negative int")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def uvarint_decode(buf: bytes, off: int) -> Tuple[int, int]:
    shift = 0
    val = 0
    while True:
        if off >= len(buf):
            raise ValueError("Truncated uvarint")
        b = buf[off]
        off += 1
        val |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return val, off
        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large")


def pack_str(s: Optional[str]) -> bytes:
    if not s:
        return uvarint_encode(0)
    b = s.encode("utf-8")
    return uvarint_encode(len(b)) + b


def unpack_str(buf: bytes, off: int) -> Tuple[Optional[str], int]:
    n, off = uvarint_decode(buf, off)
    if n == 0:
        return None, off
    if off + n > len(buf):
        raise ValueError("Truncated string field")
    s = buf[off:off + n].decode("utf-8")
    off += n
    return s, off


def pack_bytes(b: Optional[bytes]) -> bytes:
    if not b:
        return uvarint_encode(0)
    return uvarint_encode(len(b)) + b


def unpack_bytes(buf: bytes, off: int) -> Tuple[Optional[bytes], int]:
    n, off = uvarint_decode(buf, off)
    if n == 0:
        return None, off
    if off + n > len(buf):
        raise ValueError("Truncated bytes field")
    out = buf[off:off + n]
    off += n
    return out, off


DTYPE_ENUM = {"fp32": 0, "bf16": 1, "fp16": 2}
DTYPE_ENUM_REV = {v: k for k, v in DTYPE_ENUM.items()}
CODING_ENUM = {"range": 0, "ans": 1}
CODING_ENUM_REV = {v: k for k, v in CODING_ENUM.items()}
CONTENT_ENUM = {"text": 0, "text_archive": 1}
CONTENT_ENUM_REV = {v: k for k, v in CONTENT_ENUM.items()}


def build_header_bytes(
    *,
    model_id: Optional[str],
    revision: Optional[str],
    max_ctx: int,
    n_tokens: int,
    dev: str,
    dtype: str,
    use_cache: bool,
    perfect: bool,
    strict_det: bool,
    trust_remote_code: bool,
    force_math_sdpa: bool,
    env_hash: Optional[bytes],
    coding_scheme: str = "range",
    content_kind: str = "text",
) -> bytes:
    if coding_scheme not in CODING_ENUM:
        raise ValueError(f"Unsupported coding scheme: {coding_scheme}")
    if content_kind not in CONTENT_ENUM:
        raise ValueError(f"Unsupported content kind: {content_kind}")

    version = 6
    flags = 0
    flags |= (1 if perfect else 0) << 0
    flags |= (1 if strict_det else 0) << 1
    flags |= (1 if use_cache else 0) << 2
    flags |= (1 if trust_remote_code else 0) << 3
    flags |= (DTYPE_ENUM[dtype] & 0x3) << 4
    flags |= (1 if device_is_cuda(dev) else 0) << 6
    flags |= (1 if force_math_sdpa else 0) << 7

    hb = bytearray()
    hb += uvarint_encode(version)
    hb += uvarint_encode(flags)
    hb += uvarint_encode(int(max_ctx))
    hb += uvarint_encode(int(n_tokens))
    hb += uvarint_encode(int(cuda_index(dev)))
    hb += pack_str(model_id)
    hb += pack_str(revision)
    hb += pack_bytes(env_hash)
    hb += uvarint_encode(CODING_ENUM[coding_scheme])
    hb += uvarint_encode(CONTENT_ENUM[content_kind])
    return bytes(hb)


def parse_header_bytes(hb: bytes) -> dict:
    off = 0
    coding_id = CODING_ENUM["range"]
    content_id = CONTENT_ENUM["text"]
    version, off = uvarint_decode(hb, off)
    if version == 1:
        flags, off = uvarint_decode(hb, off)
        seed, off = uvarint_decode(hb, off)
        max_ctx, off = uvarint_decode(hb, off)
        vocab_size, off = uvarint_decode(hb, off)
        start_id, off = uvarint_decode(hb, off)
        n_tokens, off = uvarint_decode(hb, off)
        n_words_u32, off = uvarint_decode(hb, off)
        cuda_idx, off = uvarint_decode(hb, off)
        model_id, off = unpack_str(hb, off)
        revision, off = unpack_str(hb, off)
        env_hash, off = unpack_bytes(hb, off)
    elif version == 2:
        flags, off = uvarint_decode(hb, off)
        max_ctx, off = uvarint_decode(hb, off)
        n_tokens, off = uvarint_decode(hb, off)
        n_words_u32, off = uvarint_decode(hb, off)
        cuda_idx, off = uvarint_decode(hb, off)
        model_id, off = unpack_str(hb, off)
        revision, off = unpack_str(hb, off)
        env_hash, off = unpack_bytes(hb, off)
    elif version in (3, 4, 5, 6):
        flags, off = uvarint_decode(hb, off)
        max_ctx, off = uvarint_decode(hb, off)
        n_tokens, off = uvarint_decode(hb, off)
        cuda_idx, off = uvarint_decode(hb, off)
        model_id, off = unpack_str(hb, off)
        revision, off = unpack_str(hb, off)
        env_hash, off = unpack_bytes(hb, off)
        if version >= 5:
            coding_id, off = uvarint_decode(hb, off)
        else:
            coding_id = CODING_ENUM["range"]
        if version >= 6:
            content_id, off = uvarint_decode(hb, off)
        else:
            content_id = CONTENT_ENUM["text"]
    else:
        raise ValueError(f"Unsupported header version: {version}")

    dtype_enum = (flags >> 4) & 0x3
    dtype = DTYPE_ENUM_REV.get(dtype_enum, "fp32")
    if int(coding_id) not in CODING_ENUM_REV:
        raise ValueError(f"Unsupported coding scheme id: {coding_id}")
    if int(content_id) not in CONTENT_ENUM_REV:
        raise ValueError(f"Unsupported content kind id: {content_id}")

    is_cuda = bool((flags >> 6) & 1)
    dev = f"cuda:{cuda_idx}" if is_cuda else "cpu"

    out = {
        "version": version,
        "flags": flags,
        "perfect": bool((flags >> 0) & 1),
        "strict_det": bool((flags >> 1) & 1),
        "use_cache": bool((flags >> 2) & 1),
        "trust_remote_code": bool((flags >> 3) & 1),
        "dtype": dtype,
        "device": dev,
        "force_math_sdpa": bool((flags >> 7) & 1),
        "max_ctx": int(max_ctx),
        "n_tokens": int(n_tokens),
        "model": model_id,
        "revision": revision,
        "env_hash": env_hash,
        "coding_scheme": CODING_ENUM_REV[int(coding_id)],
        "content_kind": CONTENT_ENUM_REV[int(content_id)],
    }
    if version in (1, 2):
        out["n_words_u32"] = int(n_words_u32)
    if version == 1:
        out["seed"] = int(seed)
        out["vocab_size"] = int(vocab_size)
        out["start_id"] = int(start_id)
    return out


def write_container(path: str, header_bytes: bytes, payload: bytes) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(uvarint_encode(len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def read_container(path: str) -> Tuple[dict, bytes, int]:
    data = open(path, "rb").read()
    if len(data) < len(MAGIC) + 1:
        raise ValueError("File too small / invalid container.")
    if data[:len(MAGIC)] != MAGIC:
        raise ValueError("Bad magic; not a valid LLMCF5 container.")
    off = len(MAGIC)
    hlen, off = uvarint_decode(data, off)
    if off + hlen > len(data):
        raise ValueError("Truncated header")
    hb = data[off:off + hlen]
    off += hlen
    header = parse_header_bytes(hb)
    payload = data[off:]
    header_total = len(MAGIC) + len(uvarint_encode(hlen)) + hlen
    return header, payload, header_total
