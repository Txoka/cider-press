# CiderPress Architecture

## Goal
Build a deterministic, testable LLM-guided text compressor with clean boundaries between CLI, model inference, entropy coding, and file format handling.

## Recommended structure

```text
src/ciderpress/
  __init__.py
  cli.py                # argparse / command dispatch
  config.py             # dataclasses + runtime options
  determinism.py        # seed and backend controls
  fingerprint.py        # env fingerprint string/hash logic
  container.py          # uvarint, header packing/parsing, file read/write
  model/
    __init__.py
    hf_llm.py           # tokenizer/model loading and token<->text conversion
    stepper.py          # sequential logits/probability stepping
  codec/
    __init__.py
    range_codec.py      # compress/decompress token streams with constriction
  app/
    __init__.py
    compress.py         # orchestration for compress command
    decompress.py       # orchestration for decompress command
    selftest.py         # deterministic confidence checks
tests/
  unit/
  integration/
```

## Boundary rules
- `container.py` must not import `torch` or `transformers`.
- `model/*` must not know about file/container format.
- `codec/*` must accept token probabilities through a narrow interface and be model-agnostic.
- `app/*` composes components and enforces policy checks (env hash, vocab/start token matching).
- `cli.py` handles argument parsing only, then calls `app/*`.

## Interfaces to stabilize first
- `LLMBackend` protocol
  - `encode_text(text) -> list[int]`
  - `decode_tokens(tokens) -> str`
  - `vocab_size`, `start_id`, `max_ctx`
  - `next_probs(ctx_state) -> np.ndarray`
- `ContainerHeader` dataclass
  - explicit schema versioning and compatibility checks.
- `Codec` interface
  - `compress(tokens, prob_provider) -> np.ndarray`
  - `decompress(words, n_tokens, prob_provider) -> list[int]`

## Refactor sequence
1. Extract `container.py` exactly as-is and keep behavior bit-identical.
2. Extract `fingerprint.py` + `determinism.py`.
3. Extract range codec functions.
4. Wrap existing `LLM`/`Stepper` under `model/*`.
5. Move command bodies into `app/*`.
6. Keep `cider.py` as a thin compatibility entrypoint until migration is complete.

## Why this architecture fits this project
- Current code is functionally solid but monolithic; this split isolates high-churn ML code from stable binary/container code.
- Determinism logic becomes testable independently.
- Integration tests can exercise orchestration while unit tests mock model and coder layers.
- Backward compatibility is easier to enforce through a dedicated container module.
