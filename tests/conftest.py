import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CIDER_PATH = ROOT / "press.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeTensor:
    def __init__(self, arr):
        self.arr = np.array(arr, dtype=np.float32)

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _FakeCuda:
    def __init__(self, available=False):
        self._available = available

    def is_available(self):
        return self._available

    def get_device_name(self, idx):
        return f"fake-gpu-{idx}"

    def manual_seed_all(self, seed):
        return None


class _FakeEncoder:
    def __init__(self):
        self.data = []

    def encode(self, symbols, _model, _probs):
        self.data.append(int(symbols[0]))

    def get_compressed(self):
        return np.array(self.data, dtype=np.uint32)


class _FakeDecoder:
    def __init__(self, words):
        self.words = list(np.array(words, dtype=np.uint32).tolist())
        self.idx = 0

    def decode(self, _model, _probs):
        if self.idx >= len(self.words):
            raise IndexError("decoder underflow")
        out = self.words[self.idx]
        self.idx += 1
        return np.array([out], dtype=np.int32)


class _FakeCategorical:
    def __init__(self, perfect=False):
        self.perfect = perfect


def _build_fake_torch(cuda_available=False):
    mod = types.ModuleType("torch")
    mod.__version__ = "0.test"
    mod.float32 = "float32"
    mod.bfloat16 = "bfloat16"
    mod.float16 = "float16"
    mod.long = "long"
    mod.cuda = _FakeCuda(available=cuda_available)
    mod.version = types.SimpleNamespace(cuda="0.0")
    mod.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=False),
            enable_flash_sdp=lambda _x: None,
            enable_mem_efficient_sdp=lambda _x: None,
            enable_math_sdp=lambda _x: None,
        ),
        cudnn=types.SimpleNamespace(
            allow_tf32=False,
            benchmark=False,
            deterministic=False,
            version=lambda: 0,
        ),
    )

    def _inference_mode():
        def _decorator(fn):
            return fn

        return _decorator

    mod.inference_mode = _inference_mode
    mod.manual_seed = lambda _seed: None
    mod.use_deterministic_algorithms = lambda *_args, **_kwargs: None
    mod.set_float32_matmul_precision = lambda _x: None

    def _tensor(data, dtype=None, device=None):
        del dtype, device
        return np.array(data)

    def _softmax(x, dim=-1):
        arr = np.array(x, dtype=np.float32)
        exp = np.exp(arr - np.max(arr, axis=dim, keepdims=True))
        out = exp / exp.sum(axis=dim, keepdims=True)
        return _FakeTensor(out)

    mod.tensor = _tensor
    mod.softmax = _softmax
    mod.dtype = object
    return mod


def _build_fake_transformers():
    mod = types.ModuleType("transformers")
    mod.__version__ = "0.test"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("from_pretrained should be mocked in tests")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("from_pretrained should be mocked in tests")

    mod.AutoTokenizer = _AutoTokenizer
    mod.AutoModelForCausalLM = _AutoModelForCausalLM
    return mod


def _build_fake_constriction():
    mod = types.ModuleType("constriction")
    mod.__version__ = "0.test"
    mod.stream = types.SimpleNamespace(
        model=types.SimpleNamespace(Categorical=_FakeCategorical),
        queue=types.SimpleNamespace(RangeEncoder=_FakeEncoder, RangeDecoder=_FakeDecoder),
    )
    return mod


def load_cider(monkeypatch, *, cuda_available=False, module_name="cider_under_test"):
    monkeypatch.setitem(sys.modules, "torch", _build_fake_torch(cuda_available=cuda_available))
    monkeypatch.setitem(sys.modules, "transformers", _build_fake_transformers())
    monkeypatch.setitem(sys.modules, "constriction", _build_fake_constriction())
    for name in list(sys.modules.keys()):
        if name in {"ciderpress", "press"} or name.startswith("ciderpress."):
            sys.modules.pop(name, None)
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, CIDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cider(monkeypatch):
    return load_cider(monkeypatch)
