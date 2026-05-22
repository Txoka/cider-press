# CiderPress

`CiderPress` is an LLM-guided text compressor built on constriction entropy coding.
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

The default entropy coder is range coding. ANS is also available:

```bash
python press.py compress --coding ans --in input.txt --out output.llmz
python press.py decompress --in output.llmz --out recovered.txt
```

Directories are supported as text archives. CiderPress records UTF-8 regular files and
directories in deterministic path order; files that are not valid UTF-8, symlinks, and
other special files are skipped.

```bash
python press.py compress --in ./src --out src.llmz
python press.py decompress --in src.llmz --out ./src.recovered
```

When tokenized input is longer than the active context window, CiderPress automatically
uses a StreamingLLM-style attention-sink cache. The default is 4 sink tokens and the
active `--max-ctx` value as the cache window. You can force it explicitly:

```bash
python press.py compress --cache-policy streaming_llm --sink-tokens 4 --max-ctx 4096 ...
```

Environment compatibility hash is stored as a truncated 16-byte SHA-256 digest by default.

## Run tests

```bash
pytest -q
```

## Architecture

See `ARCHITECTURE.md` for the modular architecture and refactor plan.
