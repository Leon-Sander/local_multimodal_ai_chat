"""Shared fixtures for the contract-level test suite.

IMPORTANT - import-time side effects in src/ are load-bearing: utils.py,
chat_api_handler.py and database_operations.py read the CWD-relative
config.yaml when first imported, and database_operations.py opens a sqlite
database at the configured path at import time. The module-level code below
therefore chdirs into a throwaway sandbox seeded with a minimal config.yaml
BEFORE any test module (and hence any src module) is imported: every
import-time side effect lands in the sandbox, the repo's real config.yaml and
chat_sessions/ are never read or written, and the suite works from any
invocation directory.
"""
import atexit
import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest
import requests

_SANDBOX = Path(tempfile.mkdtemp(prefix="local-multimodal-chat-tests-"))
(_SANDBOX / "chat_sessions").mkdir()
(_SANDBOX / "config.yaml").write_text(
    "\n".join(
        [
            "ollama:",
            '  embedding_model: "sandbox-embed"',
            "  base_url: http://sandbox-ollama:11434",
            "",
            'whisper_model: "sandbox-whisper"',
            "",
            "chromadb:",
            '  chromadb_path: "chroma_db"',
            '  collection_name: "pdfs"',
            "",
            'chat_sessions_database_path: "./chat_sessions/chat_sessions.db"',
            "",
        ]
    )
)
# Best-effort cleanup; ignore_errors because the module-level singleton in
# database_operations holds its sandbox db open for the whole session.
atexit.register(shutil.rmtree, _SANDBOX, ignore_errors=True)


def pytest_configure(config):
    # chdir in the configure hook, not at conftest import: pytest resolves
    # testpaths against the CWD first, and collection (which imports the test
    # modules and therefore the src modules) only starts afterwards - so the
    # import-time side effects still land in the sandbox.
    os.chdir(_SANDBOX)


class FakeSessionState(dict):
    """Stand-in for st.session_state covering the four access patterns used in
    src/: st.session_state["key"], .get("key", default), attribute access, and
    membership tests. Never rely on Streamlit's bare-mode session_state - it is
    a process-global singleton with undocumented semantics."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


class DummyResponse:
    """Minimal requests.Response stand-in; .json() is repeatably callable."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class RecordingPost:
    """Callable that records every requests.post(...) call and returns a
    canned response. The signature mirrors the real
    requests.post(url, data=None, json=None, ...) so that a refactor in src to
    positional arguments cannot silently record the body under the wrong key."""

    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.calls = []

    def __call__(self, url=None, data=None, json=None, headers=None, **kwargs):
        self.calls.append({"url": url, "data": data, "json": json, "headers": headers})
        return DummyResponse(self.response_payload)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test hits the network without explicitly patching
    it: requests.get/post are replaced for a clear error message, and raw
    socket connects are blocked as a catch-all (this also covers aiohttp and
    requests.Session). captured_post overrides requests.post for tests that
    expect an HTTP call."""

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "Network access is disabled in tests; use the captured_post "
            "fixture or patch the network call explicitly."
        )

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def captured_post(monkeypatch):
    """Factory: captured_post(payload) patches requests.post to record calls
    and answer with `payload`. Works because src modules do `import requests`
    and look up requests.post at call time."""

    def _install(response_payload):
        recorder = RecordingPost(response_payload)
        monkeypatch.setattr(requests, "post", recorder)
        return recorder

    return _install


@pytest.fixture
def fake_session_state(monkeypatch):
    """Replace st.session_state for every src module at once (they all do
    `import streamlit as st`, so patching the module attribute reaches all)."""
    import streamlit as st

    state = FakeSessionState()
    monkeypatch.setattr(st, "session_state", state)
    return state


@pytest.fixture
def db_manager(tmp_path):
    """A DatabaseManager on a throwaway sqlite file. close() in teardown is
    mandatory: an open handle makes tmp_path cleanup fail on Windows."""
    from database_operations import DatabaseManager

    manager = DatabaseManager(str(tmp_path / "test.db"))
    yield manager
    manager.close()
