# CiderPress

`CiderPress` is an LLM-guided text compressor built on constriction range coding.
The implementation is modular under `ciderpress/` with `press.py` as the main CLI entrypoint.
A backward-compatible alias `cider.py` is also available.

## Create a Python venv

1. Create the virtual environment:

```bash
python3 -m venv .venv
```

2. Activate it:

```bash
source .venv/bin/activate
```

3. Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python press.py --help
```

Determinism configuration is optional and disabled by default. Enable it explicitly when needed:

```bash
python press.py compress --determinism on ...
```

Environment compatibility hash is stored as a truncated 16-byte SHA-256 digest by default.

## Run tests

```bash
pytest -q
```

## Architecture

See `ARCHITECTURE.md` for the modular architecture and refactor plan.
