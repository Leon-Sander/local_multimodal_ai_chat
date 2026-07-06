"""Contract tests for the pure helpers in src/utils.py."""
import base64

import utils


def test_base64_conversions():
    payload = b"\x00\xff fake image bytes"

    encoded = utils.convert_bytes_to_base64(payload)
    assert isinstance(encoded, str)
    assert base64.b64decode(encoded) == payload

    # The prefix is the OpenAI vision data-URL wire format - pinning it
    # verbatim pins behavior, not styling.
    prefixed = utils.convert_bytes_to_base64_with_prefix(payload)
    prefix = "data:image/jpeg;base64,"
    assert prefixed.startswith(prefix)
    assert base64.b64decode(prefixed[len(prefix):]) == payload


def test_command_dispatch(monkeypatch):
    calls = []
    # Accepted churn (see TESTING.md): pull_model_in_background is an internal
    # seam - patching it is the price of not running asyncio + a live Ollama
    # pull here. Renaming that function means updating this one line.
    monkeypatch.setattr(
        utils,
        "pull_model_in_background",
        lambda model_name: calls.append(model_name) or "pull result",
    )

    # /pull dispatches exactly once with the model name and forwards the result.
    assert utils.command("/pull llama3") == "pull result"
    assert calls == ["llama3"]

    # /help and unknown commands answer with a string and never dispatch a
    # pull. Deliberately no assertions on wording - message text is not a
    # contract.
    assert isinstance(utils.command("/help"), str)
    assert isinstance(utils.command("/nonsense"), str)
    assert calls == ["llama3"]


def test_config_save_load_roundtrip(monkeypatch, tmp_path):
    # save_config hardcodes the relative path "config.yaml", so run in an
    # empty tmp dir - both functions resolve the path at call time, the real
    # config.yaml is never touched.
    monkeypatch.chdir(tmp_path)
    config = {"ollama": {"base_url": "http://example:11434"}, "values": [1, 2, 3]}

    utils.save_config(config)

    assert utils.load_config() == config
    assert utils.load_config(str(tmp_path / "config.yaml")) == config
