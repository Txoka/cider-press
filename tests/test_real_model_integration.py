from pathlib import Path
import importlib
import sys

REAL_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


def _purge_test_stubs():
    # Always reload project modules so they bind to whatever runtime is active now.
    sys.modules.pop("press", None)
    sys.modules.pop("ciderpress", None)
    for name in list(sys.modules.keys()):
        if name.startswith("ciderpress."):
            sys.modules.pop(name, None)

    # Only purge fake dependency stubs injected by tests/conftest.py.
    for base in ("torch", "transformers", "constriction"):
        mod = sys.modules.get(base)
        if mod is not None and getattr(mod, "__version__", None) == "0.test":
            sys.modules.pop(base, None)
            for name in list(sys.modules.keys()):
                if name.startswith(base + "."):
                    sys.modules.pop(name, None)

    importlib.invalidate_caches()


def _import_runtime_required():
    _purge_test_stubs()
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import constriction  # noqa: F401


def _load_press_required():
    import press

    return press


def test_real_model_token_compression_roundtrip():
    _import_runtime_required()
    press = _load_press_required()

    llm = press.LLM(
        press.LLMConfig(
            model_id=REAL_MODEL_ID,
            revision=None,
            device="cpu",
            dtype="fp32",
            max_ctx=64,
            use_cache=True,
            trust_remote_code=False,
        )
    )

    text = "Real model integration token roundtrip.\\n"
    tokens = llm.encode_text(text)
    assert len(tokens) > 0

    comp, _ = press.compress_tokens_sequential(llm, tokens, perfect=False, show_progress=False)
    dec = press.decompress_tokens_sequential(llm, comp, n_tokens=len(tokens), perfect=False, show_progress=False)
    assert dec == tokens


def test_real_model_cli_file_roundtrip(tmp_path, monkeypatch):
    _import_runtime_required()
    press = _load_press_required()

    asset = Path(__file__).resolve().parents[1] / "tests" / "assets" / "roundtrip_input.txt"
    assert asset.exists()

    out = tmp_path / "real_model.llmz"
    rec = tmp_path / "real_model_recovered.txt"

    monkeypatch.setattr(
        press.sys,
        "argv",
        [
            "press.py",
            "compress",
            "--in",
            str(asset),
            "--out",
            str(out),
        ],
    )
    press.main()

    monkeypatch.setattr(
        press.sys,
        "argv",
        [
            "press.py",
            "decompress",
            "--in",
            str(out),
            "--out",
            str(rec),
        ],
    )
    press.main()

    assert rec.read_text(encoding="utf-8") == asset.read_text(encoding="utf-8")
