# Testing

This repo has a small, deliberate test suite: **21 contract tests** over the parts of the code whose behavior is stable by design. It is meant to stay cheap to maintain — including for forks.

## Philosophy: contracts, not structure

A test here pins a **behavioral promise** (what a function guarantees to its callers), never an implementation detail. The intended property: **a test only fails when behavior actually changed.** If you fork this repo, rewrite internals freely — a red test means you changed something user-visible, which is exactly when you want to know.

A few places accept **medium churn** deliberately, each documented inline where it occurs:

- `test_pdf_chat_stuffs_retrieved_context_into_prompt` patches the internal `load_vectordb` seam (the alternative is a real Chroma + Ollama instance).
- `test_command_dispatch` patches the internal `pull_model_in_background` seam (the alternative is asyncio + a live Ollama pull).
- The handler tests patch two values that src binds **at import time** (`chat_api_handler.config`, `chat_api_handler.openai_api_key`). A behavior-preserving refactor to call-time lookups would require updating the two fixtures at the top of `tests/test_chat_api_handler.py` — a two-line fix, accepted as the cost of avoiding real env vars and network.

## What is covered

| File | Contracts |
|------|-----------|
| `tests/test_database_operations.py` (10) | message roundtrips (text + binary), LLM-context feed ordering/filtering, per-session delete isolation, session id listing, settings read-through defaults + persistence, restart durability |
| `tests/test_utils.py` (3) | base64/data-URL wire format, `/command` dispatch, config save/load roundtrip |
| `tests/test_chat_api_handler.py` (8) | endpoint routing (ollama/openai/unknown), request bodies + auth headers, image wire formats (Ollama bare-b64 vs OpenAI data URL), RAG context stuffing, error-shape handling (string vs nested object) |

Counts are executed test cases, not test functions: `test_database_operations.py` has 9 test functions but one is parametrized over `["image", "audio"]`, so it runs 10 cases — 21 in total.

## What deliberately stays untested (and why)

- **UI structure and CSS** (`app.py`, `html_templates.py`) — Streamlit class names churn every release; tests would break without any behavior change.
- **Audio transcription** (`audio_handler.py`) — needs Whisper model downloads; the browser microphone path can't be unit-tested anyway.
- **Real model calls / PDF ingestion** — network- and model-dependent; the request *construction* is tested, the live call is not.
- **Trivial helpers** (`convert_ns_to_seconds`, `get_avatar`, timestamp formatting) — near-zero regression value for the count they add.
- **The manual smoke test remains**: launch the app, send a text message, record audio once. That is the irreducible browser part.

## Running

From the **repo root**:

```bash
pip install -r requirements-dev.txt   # includes requirements.txt
pytest
```

CI runs the same suite on Ubuntu and Windows (Python 3.10) on every push/PR to `main` — which also doubles as a "does it still install on Windows" check.

## Gotchas when writing new tests

The src modules bind several things **at import time** — patch the right namespace:

| To fake | Patch | Why |
|---------|-------|-----|
| OpenAI key | `chat_api_handler.openai_api_key` | `os.getenv` ran at import; `monkeypatch.setenv` does nothing |
| Ollama base URL | `chat_api_handler.config` | module-level `load_config()` froze it |
| Vector DB | `chat_api_handler.load_vectordb` | from-import binding; patching `vectordb_handler` does nothing |
| HTTP | `requests.post` (module attribute) | call-time lookup, reaches all src modules |
| Session state | `streamlit.session_state` → `FakeSessionState` (conftest) | never rely on bare-mode session_state |

Other invariants:

- An autouse `no_network` fixture replaces `requests.get/post` and blocks raw socket connects (which also covers `aiohttp` and `requests.Session`) — a test that reaches for the network fails loudly instead of silently hitting a real endpoint.
- DB tests build their own `DatabaseManager` on `tmp_path` and must `close()` it (Windows cannot delete open sqlite files). Never assert through the module-level singleton in `database_operations`.
- `tests/conftest.py` chdirs into a throwaway **sandbox** (with its own minimal `config.yaml` and `chat_sessions/`) before anything imports the src modules — their import-time side effects (config read, sqlite creation) land in the sandbox, the repo's real `config.yaml` and `chat_sessions/` are never read or written, and pytest works from any invocation directory.
