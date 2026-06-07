import argparse
import hashlib
import importlib
import sys
from pathlib import Path
import types

import numpy as np
import pytest

from conftest import load_cider


def test_canonical_device_defaults_to_cpu(monkeypatch):
    cider = load_cider(monkeypatch, cuda_available=False, module_name="cider_t1")
    assert cider.canonical_device(None) == "cpu"


def test_canonical_device_defaults_to_cuda_when_available(monkeypatch):
    cider = load_cider(monkeypatch, cuda_available=True, module_name="cider_t2")
    assert cider.canonical_device(None) == "cuda:0"


def test_canonical_device_valid_and_invalid(cider):
    assert cider.canonical_device("cuda") == "cuda:0"
    assert cider.canonical_device("CUDA:2") == "cuda:2"
    assert cider.canonical_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        cider.canonical_device("cuda:x")
    with pytest.raises(ValueError):
        cider.canonical_device("mps")


def test_uvarint_roundtrip(cider):
    vals = [0, 1, 127, 128, 255, 300, 16384, 2**32 - 1, 2**63 - 1]
    for v in vals:
        enc = cider.uvarint_encode(v)
        dec, off = cider.uvarint_decode(enc, 0)
        assert dec == v
        assert off == len(enc)


def test_uvarint_errors(cider):
    with pytest.raises(ValueError):
        cider.uvarint_encode(-1)
    with pytest.raises(ValueError, match="Truncated uvarint"):
        cider.uvarint_decode(b"", 0)
    with pytest.raises(ValueError, match="uvarint too large"):
        cider.uvarint_decode(b"\x80" * 11 + b"\x00", 0)


def test_pack_unpack_str_and_bytes(cider):
    packed_none = cider.pack_str(None)
    val, off = cider.unpack_str(packed_none, 0)
    assert val is None
    assert off == len(packed_none)

    packed = cider.pack_str("hello")
    val, off = cider.unpack_str(packed, 0)
    assert val == "hello"
    assert off == len(packed)

    bpacked_none = cider.pack_bytes(None)
    bval, boff = cider.unpack_bytes(bpacked_none, 0)
    assert bval is None
    assert boff == len(bpacked_none)

    bpacked = cider.pack_bytes(b"abc")
    bval, boff = cider.unpack_bytes(bpacked, 0)
    assert bval == b"abc"
    assert boff == len(bpacked)



def test_unpack_truncated_fields(cider):
    with pytest.raises(ValueError, match="Truncated string field"):
        cider.unpack_str(cider.uvarint_encode(3) + b"ab", 0)
    with pytest.raises(ValueError, match="Truncated bytes field"):
        cider.unpack_bytes(cider.uvarint_encode(3) + b"ab", 0)


def test_header_roundtrip(cider):
    env_hash = hashlib.sha256(b"x").digest()
    hb = cider.build_header_bytes(
        model_id="m",
        revision="r",
        max_ctx=1024,
        n_tokens=3,
        dev="cuda:2",
        dtype="bf16",
        use_cache=True,
        perfect=False,
        strict_det=True,
        trust_remote_code=True,
        force_math_sdpa=True,
        env_hash=env_hash,
    )
    out = cider.parse_header_bytes(hb)
    assert out["version"] == 7
    assert out["model"] == "m"
    assert out["revision"] == "r"
    assert out["max_ctx"] == 1024
    assert out["n_tokens"] == 3
    assert "n_words_u32" not in out
    assert out["dtype"] == "bf16"
    assert out["device"] == "cuda:2"
    assert out["use_cache"] is True
    assert out["strict_det"] is True
    assert out["trust_remote_code"] is True
    assert out["force_math_sdpa"] is True
    assert out["env_hash"] == env_hash
    assert out["coding_scheme"] == "range"
    assert out["content_kind"] == "text"
    assert out["cache_policy"] == "dynamic"
    assert out["sink_tokens"] == 4


def test_header_roundtrip_with_ans_coding(cider):
    hb = cider.build_header_bytes(
        model_id="m",
        revision=None,
        max_ctx=16,
        n_tokens=3,
        dev="cpu",
        dtype="fp32",
        use_cache=False,
        perfect=False,
        strict_det=False,
        trust_remote_code=False,
        force_math_sdpa=False,
        env_hash=None,
        coding_scheme="ans",
        content_kind="text_archive",
        cache_policy="streaming_llm",
        sink_tokens=4,
    )
    out = cider.parse_header_bytes(hb)
    assert out["version"] == 7
    assert out["coding_scheme"] == "ans"
    assert out["content_kind"] == "text_archive"
    assert out["cache_policy"] == "streaming_llm"


def test_parse_header_rejects_unknown_version(cider):
    with pytest.raises(ValueError, match="Unsupported header version"):
        cider.parse_header_bytes(cider.uvarint_encode(8) + cider.uvarint_encode(0))


def test_write_read_container_roundtrip(cider, tmp_path):
    header = cider.build_header_bytes(
        model_id="m",
        revision=None,
        max_ctx=16,
        n_tokens=0,
        dev="cpu",
        dtype="fp32",
        use_cache=False,
        perfect=False,
        strict_det=False,
        trust_remote_code=False,
        force_math_sdpa=False,
        env_hash=None,
    )
    payload = b"\x01\x02\x03"
    path = tmp_path / "x.llmz"
    cider.write_container(str(path), header, payload)

    parsed, out_payload, header_total = cider.read_container(str(path))
    assert parsed["model"] == "m"
    assert out_payload == payload
    assert header_total == len(cider.MAGIC) + len(cider.uvarint_encode(len(header))) + len(header)


def test_text_archive_roundtrip_skips_non_utf8(cider, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "sub").mkdir()
    (src / "sub" / "a.txt").write_text("hello\n", encoding="utf-8")
    (src / "empty").mkdir()
    (src / "binary.bin").write_bytes(b"\xff\xfe\x00")
    (src / "nul.txt").write_bytes(b"text\x00")

    archive_mod = importlib.import_module("ciderpress.text_archive")
    text, n_files, n_dirs, skipped = archive_mod.build_text_archive(str(src))
    assert n_files == 1
    assert n_dirs == 2
    assert skipped == 2

    out = tmp_path / "out"
    restored_files, restored_dirs = archive_mod.extract_text_archive(text, str(out))
    assert restored_files == 1
    assert restored_dirs == 2
    assert (out / "sub" / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert (out / "empty").is_dir()
    assert not (out / "binary.bin").exists()
    assert not (out / "nul.txt").exists()


def test_read_container_errors(cider, tmp_path):
    small = tmp_path / "small.bin"
    small.write_bytes(b"x")
    with pytest.raises(ValueError, match="File too small"):
        cider.read_container(str(small))

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"NOTLLM0")
    with pytest.raises(ValueError, match="Bad magic"):
        cider.read_container(str(bad))

    trunc = tmp_path / "trunc.bin"
    trunc.write_bytes(cider.MAGIC + cider.uvarint_encode(10) + b"abc")
    with pytest.raises(ValueError, match="Truncated header"):
        cider.read_container(str(trunc))


def test_env_fingerprint_and_hash(monkeypatch):
    cider = load_cider(monkeypatch, cuda_available=True, module_name="cider_t3")
    fp = cider.env_fingerprint_string(
        dev="cuda:0",
        dtype="bf16",
        use_cache=True,
        perfect=True,
        strict_det=False,
        force_math_sdpa=True,
    )
    assert "torch=" in fp
    assert "transformers=" in fp
    assert "dev=cuda:0" in fp
    assert "gpu0=fake-gpu-0" in fp

    h1 = cider.env_fingerprint_hash_bytes(
        dev="cuda:0",
        dtype="bf16",
        use_cache=True,
        perfect=True,
        strict_det=False,
        force_math_sdpa=True,
    )
    h2 = cider.env_fingerprint_hash_bytes(
        dev="cuda:0",
        dtype="bf16",
        use_cache=True,
        perfect=True,
        strict_det=False,
        force_math_sdpa=True,
    )
    assert isinstance(h1, bytes)
    assert len(h1) == cider.ENV_HASH_NBYTES
    assert h1 == h2

    h3 = cider.env_fingerprint_hash_bytes(
        dev="cuda:0",
        dtype="bf16",
        use_cache=True,
        perfect=True,
        strict_det=False,
        force_math_sdpa=True,
        model_id="other/model",
        revision="main",
    )
    assert h1 != h3


class _FakeStepper:
    def __init__(self, llm):
        del llm
        self.tokens = []

    def probs_np(self):
        probs = np.array([[0.8, 0.1, 0.1]], dtype=np.float32)
        return probs

    def step(self, token):
        self.tokens.append(int(token))


def test_compress_decompress_roundtrip_with_stub_stepper(cider, monkeypatch):
    monkeypatch.setattr(cider, "Stepper", _FakeStepper)

    class _L:
        pass

    llm = _L()
    tokens = [0, 1, 2, 0]
    comp, nll = cider.compress_tokens_sequential(llm, tokens, perfect=True, show_progress=False)
    out = cider.decompress_tokens_sequential(llm, comp, n_tokens=len(tokens), perfect=True, show_progress=False)
    assert out == tokens
    assert isinstance(nll, float)


def test_ans_compress_decompress_roundtrip_with_stub_stepper(cider, monkeypatch):
    monkeypatch.setattr(cider, "Stepper", _FakeStepper)

    class _L:
        pass

    llm = _L()
    tokens = [0, 1, 2, 0]
    comp, nll = cider.compress_tokens_sequential(
        llm,
        tokens,
        perfect=True,
        show_progress=False,
        coding_scheme="ans",
    )
    out = cider.decompress_tokens_sequential(
        llm,
        comp,
        n_tokens=len(tokens),
        perfect=True,
        show_progress=False,
        coding_scheme="ans",
    )
    assert out == tokens
    assert isinstance(nll, float)


def test_compress_and_decompress_empty(cider):
    class _L:
        pass

    comp, nll = cider.compress_tokens_sequential(_L(), [], perfect=False, show_progress=False)
    assert comp.dtype == np.uint32
    assert comp.size == 0
    assert nll == 0.0

    out = cider.decompress_tokens_sequential(_L(), np.array([], dtype=np.uint32), n_tokens=0, perfect=False, show_progress=False)
    assert out == []


def test_load_model_fallback_to_torch_dtype(cider, monkeypatch):
    calls = []

    class _AF:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(kwargs)
            if "dtype" in kwargs:
                raise TypeError("old API")
            return "ok"

    model_mod = importlib.import_module("ciderpress.model")
    monkeypatch.setattr(model_mod, "AutoModelForCausalLM", _AF)
    out = model_mod.load_model("m", None, "fp32", False)
    assert out == "ok"
    assert len(calls) == 2
    assert "dtype" in calls[0]
    assert "torch_dtype" in calls[1]


def test_llm_init_and_codec_methods(cider, monkeypatch):
    class _Tok:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = None
        eos_token = "<eos>"
        pad_token = None

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(c) % 3 for c in text]

        def decode(self, toks, clean_up_tokenization_spaces=False, skip_special_tokens=False):
            assert clean_up_tokenization_spaces is False
            assert skip_special_tokens is False
            return "|".join(str(x) for x in toks)

    class _Emb:
        num_embeddings = 99

    class _Model:
        config = types.SimpleNamespace(max_position_embeddings=123)

        def eval(self):
            return self

        def to(self, _dev):
            return self

        def get_input_embeddings(self):
            return _Emb()

    model_mod = importlib.import_module("ciderpress.model")
    monkeypatch.setattr(model_mod, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok()))
    monkeypatch.setattr(model_mod, "load_model", lambda *a, **k: _Model())

    cfg = model_mod.LLMConfig(
        model_id="m",
        revision=None,
        device="cpu",
        dtype="fp32",
        max_ctx=0,
        use_cache=False,
        trust_remote_code=False,
    )
    llm = model_mod.LLM(cfg)
    assert llm.vocab_size == 99
    assert llm.start_id == 1
    assert llm.max_ctx == 123
    assert llm.tokenizer.pad_token == llm.tokenizer.eos_token
    assert llm.encode_text("ab") == [1, 2]
    assert llm.decode_tokens([1, 2]) == "1|2"


def test_streaming_llm_sink_cache_keeps_sinks_and_recent(cider, monkeypatch):
    model_mod = importlib.import_module("ciderpress.model")
    monkeypatch.setattr(model_mod, "torch", types.SimpleNamespace(cat=lambda xs, dim: np.concatenate(xs, axis=dim)))
    cache = model_mod.StreamingLLMSinkCache(window_length=5, num_sink_tokens=2)
    for i in range(7):
        k = np.full((1, 1, 1, 1), i, dtype=np.float32)
        v = np.full((1, 1, 1, 1), i, dtype=np.float32)
        cache.update(k, v, 0, {})
    assert cache.get_seq_length() == 5
    assert cache.key_cache[0].reshape(-1).tolist() == [0, 1, 4, 5, 6]
    assert cache.get_mask_sizes(np.zeros((1,), dtype=np.int64), 0) == (6, 0)


def test_determinism_logic_in_pytest(cider, monkeypatch):
    class _FakeLLM:
        def __init__(self, _cfg=None):
            pass

        def encode_text(self, _text):
            return [0, 1, 2, 0]

    class _S:
        def __init__(self, llm):
            del llm

        def probs_np(self):
            return np.array([[0.7, 0.2, 0.1]], dtype=np.float32)

        def step(self, token):
            del token

    monkeypatch.setattr(cider, "Stepper", _S)

    llm = _FakeLLM()
    tokens = llm.encode_text("abc")

    def prob_hash():
        st = cider.Stepper(llm)
        blobs = []
        for t in tokens:
            blobs.append(st.probs_np().tobytes(order="C"))
            st.step(t)
        return hashlib.sha256(b"".join(blobs)).hexdigest()

    h1 = prob_hash()
    h2 = prob_hash()
    assert h1 == h2

    comp1, _ = cider.compress_tokens_sequential(llm, tokens, perfect=True, show_progress=False)
    comp2, _ = cider.compress_tokens_sequential(llm, tokens, perfect=True, show_progress=False)
    assert np.array_equal(comp1, comp2)

    dec = cider.decompress_tokens_sequential(llm, comp1, n_tokens=len(tokens), perfect=True, show_progress=False)
    assert dec == tokens


def test_main_compress_then_decompress(cider, monkeypatch, tmp_path):
    class _FakeLLM:
        def __init__(self, cfg):
            self.cfg = cfg
            self.max_ctx = 64
            self.vocab_size = 256
            self.start_id = 0

        def encode_text(self, text):
            return [ord(c) for c in text]

        def decode_tokens(self, toks):
            return "".join(chr(int(x)) for x in toks)

    monkeypatch.setattr(cider, "LLM", _FakeLLM)
    monkeypatch.setattr(
        cider,
        "compress_tokens_sequential",
        lambda llm, tokens, perfect, show_progress: (np.array(tokens, dtype=np.uint32), 0.5),
    )
    monkeypatch.setattr(
        cider,
        "decompress_tokens_sequential",
        lambda llm, compressed_u32, n_tokens, perfect, show_progress: [int(x) for x in compressed_u32.tolist()][:n_tokens],
    )

    inp = tmp_path / "in.txt"
    out = tmp_path / "out.llmz"
    rec = tmp_path / "rec.txt"
    inp.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["cider.py", "compress", "--in", str(inp), "--out", str(out), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--no-progress"])
    cider.main()
    assert out.exists()

    monkeypatch.setattr(sys, "argv", ["cider.py", "decompress", "--in", str(out), "--out", str(rec), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--no-progress"])
    cider.main()
    assert rec.read_text(encoding="utf-8") == "hello"


def test_main_compress_then_decompress_text_directory(cider, monkeypatch, tmp_path):
    class _FakeLLM:
        def __init__(self, cfg):
            self.cfg = cfg
            self.max_ctx = 64
            self.vocab_size = 256
            self.start_id = 0

        def encode_text(self, text):
            return [ord(c) for c in text]

        def decode_tokens(self, toks):
            return "".join(chr(int(x)) for x in toks)

    monkeypatch.setattr(cider, "LLM", _FakeLLM)
    monkeypatch.setattr(
        cider,
        "compress_tokens_sequential",
        lambda llm, tokens, perfect, show_progress: (np.array(tokens, dtype=np.uint32), 0.5),
    )
    monkeypatch.setattr(
        cider,
        "decompress_tokens_sequential",
        lambda llm, compressed_u32, n_tokens, perfect, show_progress: [int(x) for x in compressed_u32.tolist()][:n_tokens],
    )

    src = tmp_path / "src"
    src.mkdir()
    (src / "sub").mkdir()
    (src / "sub" / "a.txt").write_text("hello\n", encoding="utf-8")
    (src / "binary.bin").write_bytes(b"\xff\xfe\x00")
    out = tmp_path / "dir.llmz"
    rec = tmp_path / "rec"

    monkeypatch.setattr(sys, "argv", ["cider.py", "compress", "--in", str(src), "--out", str(out), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--no-progress", "--verify"])
    cider.main()
    assert out.exists()

    header, _, _ = cider.read_container(str(out))
    assert header["content_kind"] == "text_archive"

    monkeypatch.setattr(sys, "argv", ["cider.py", "decompress", "--in", str(out), "--out", str(rec), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--no-progress"])
    cider.main()
    assert (rec / "sub" / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert not (rec / "binary.bin").exists()


def test_main_auto_streaming_llm_for_input_beyond_context(cider, monkeypatch, tmp_path):
    class _FakeLLM:
        def __init__(self, cfg):
            self.cfg = cfg
            self.max_ctx = int(cfg.max_ctx)
            self.vocab_size = 256
            self.start_id = 0

        def encode_text(self, text):
            return [ord(c) for c in text]

        def decode_tokens(self, toks):
            return "".join(chr(int(x)) for x in toks)

    monkeypatch.setattr(cider, "LLM", _FakeLLM)
    monkeypatch.setattr(
        cider,
        "compress_tokens_sequential",
        lambda llm, tokens, perfect, show_progress, coding_scheme=None: (np.array(tokens, dtype=np.uint32), 0.5),
    )

    inp = tmp_path / "in.txt"
    out = tmp_path / "out.llmz"
    inp.write_text("abcdef", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["cider.py", "compress", "--in", str(inp), "--out", str(out), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--max-ctx", "5", "--no-progress"])
    cider.main()
    header, _, _ = cider.read_container(str(out))
    assert header["cache_policy"] == "streaming_llm"
    assert header["sink_tokens"] == 4


def test_main_decompress_envhash_mismatch_exits(cider, monkeypatch, tmp_path):
    class _FakeLLM:
        def __init__(self, cfg):
            self.cfg = cfg
            self.max_ctx = 64
            self.vocab_size = 256
            self.start_id = 0

        def encode_text(self, text):
            return [ord(c) for c in text]

        def decode_tokens(self, toks):
            return "".join(chr(int(x)) for x in toks)

    monkeypatch.setattr(cider, "LLM", _FakeLLM)
    monkeypatch.setattr(cider, "compress_tokens_sequential", lambda llm, tokens, perfect, show_progress: (np.array(tokens, dtype=np.uint32), 0.5))

    inp = tmp_path / "in.txt"
    out = tmp_path / "out.llmz"
    inp.write_text("x", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["cider.py", "compress", "--in", str(inp), "--out", str(out), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--no-progress"])
    cider.main()

    app_mod = importlib.import_module("ciderpress.app")
    monkeypatch.setattr(app_mod, "env_fingerprint_hash_bytes", lambda **kwargs: b"\x00" * cider.ENV_HASH_NBYTES)
    monkeypatch.setattr(sys, "argv", ["cider.py", "decompress", "--in", str(out), "--out", str(tmp_path / "rec.txt"), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--no-progress"])

    with pytest.raises(SystemExit) as exc:
        cider.main()
    assert exc.value.code == 2


def test_cli_common_arguments_exposed():
    cli_mod = importlib.import_module("ciderpress.cli")
    names = [flag for flag, _ in cli_mod.COMMON_ARGUMENTS]
    assert "--model" in names
    assert "--revision" in names
    assert "--device" in names
    assert "--dtype" in names
    assert "--coding" in names


def test_cli_has_no_test_subcommand():
    cli_mod = importlib.import_module("ciderpress.cli")
    parser = cli_mod.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["test"])
    assert exc.value.code == 2


def test_roundtrip_with_asset_file(cider, monkeypatch, tmp_path):
    class _FakeLLM:
        def __init__(self, cfg):
            self.cfg = cfg
            self.max_ctx = 64
            self.vocab_size = 256
            self.start_id = 0

        def encode_text(self, text):
            return [ord(c) for c in text]

        def decode_tokens(self, toks):
            return "".join(chr(int(x)) for x in toks)

    monkeypatch.setattr(cider, "LLM", _FakeLLM)
    monkeypatch.setattr(
        cider,
        "compress_tokens_sequential",
        lambda llm, tokens, perfect, show_progress: (np.array(tokens, dtype=np.uint32), 0.5),
    )
    monkeypatch.setattr(
        cider,
        "decompress_tokens_sequential",
        lambda llm, compressed_u32, n_tokens, perfect, show_progress: [int(x) for x in compressed_u32.tolist()][:n_tokens],
    )

    asset = Path(__file__).resolve().parents[1] / "tests" / "assets" / "roundtrip_input.txt"
    assert asset.exists()

    out = tmp_path / "roundtrip.llmz"
    rec = tmp_path / "roundtrip_recovered.txt"

    monkeypatch.setattr(sys, "argv", ["press.py", "compress", "--in", str(asset), "--out", str(out), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--no-progress"])
    cider.main()

    monkeypatch.setattr(sys, "argv", ["press.py", "decompress", "--in", str(out), "--out", str(rec), "--model", "m", "--device", "cpu", "--dtype", "fp32", "--no-progress"])
    cider.main()

    assert rec.read_text(encoding="utf-8") == asset.read_text(encoding="utf-8")
