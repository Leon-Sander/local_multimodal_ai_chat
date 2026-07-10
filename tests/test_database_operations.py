"""Contract tests for the SQLite repositories.

Every test runs against a real sqlite database on tmp_path via the db_manager
fixture - no mocking. The module-level db_manager singleton that
database_operations.py creates at import time lands in the conftest sandbox
(never in the repo's real chat_sessions/); tests never assert through it.
"""
import pytest

from database_operations import DatabaseManager

SESSION = "2024-01-15 09:30:00"
OTHER_SESSION = "2024-02-20 18:45:00"


def test_save_and_load_text_message_roundtrip(db_manager):
    db_manager.message_repo.save_message(SESSION, "user", "text", "hello world")

    messages = db_manager.message_repo.load_messages(SESSION)

    assert len(messages) == 1
    message = messages[0]
    assert message["content"] == "hello world"
    assert message["sender_type"] == "user"
    assert message["message_type"] == "text"


@pytest.mark.parametrize("message_type", ["image", "audio"])
def test_blob_message_roundtrip_preserves_exact_bytes(db_manager, message_type):
    payload = b"\x89PNG\r\n\x1a\n\x00\xff"  # non-UTF-8: catches accidental text decoding

    db_manager.message_repo.save_message(SESSION, "user", message_type, payload)
    messages = db_manager.message_repo.load_messages(SESSION)

    assert len(messages) == 1
    assert isinstance(messages[0]["content"], bytes)
    assert messages[0]["content"] == payload


def test_load_last_k_text_messages_excludes_non_text(db_manager):
    repo = db_manager.message_repo
    repo.save_message(SESSION, "user", "text", "first text")
    repo.save_message(SESSION, "user", "image", b"\x00\x01")
    repo.save_message(SESSION, "user", "audio", b"\x02\x03")
    repo.save_message(SESSION, "assistant", "text", "second text")

    messages = repo.load_last_k_text_messages(SESSION, 10)

    # Blob rows must never leak into the LLM context feed.
    assert [m["message_type"] for m in messages] == ["text", "text"]
    assert [m["content"] for m in messages] == ["first text", "second text"]


def test_load_last_k_text_messages_returns_k_most_recent_oldest_first(db_manager):
    repo = db_manager.message_repo
    for i in range(1, 6):
        repo.save_message(SESSION, "user", "text", f"m{i}")

    messages = repo.load_last_k_text_messages(SESSION, 2)

    # The k newest messages, re-sorted chronologically: this is the LLM
    # context feed - returning them newest-first would feed the conversation
    # history to the model backwards.
    assert [m["content"] for m in messages] == ["m4", "m5"]


def test_delete_chat_history_removes_only_target_session(db_manager):
    repo = db_manager.message_repo
    repo.save_message(SESSION, "user", "text", "to be deleted")
    repo.save_message(SESSION, "user", "image", b"\x00")
    repo.save_message(OTHER_SESSION, "user", "text", "survivor")

    repo.delete_chat_history(SESSION)

    assert repo.load_messages(SESSION) == []
    assert [m["content"] for m in repo.load_messages(OTHER_SESSION)] == ["survivor"]
    assert repo.get_all_chat_history_ids() == [OTHER_SESSION]


def test_get_all_chat_history_ids_distinct_and_ascending(db_manager):
    repo = db_manager.message_repo
    # Inserted out of chronological order, one session appearing twice.
    for session in [
        "2024-03-01 10:00:00",
        "2024-01-15 09:30:00",
        "2024-03-01 10:00:00",
        "2024-02-20 18:45:00",
    ]:
        repo.save_message(session, "user", "text", "msg")

    ids = repo.get_all_chat_history_ids()

    # Distinct and ascending - for the timestamp-formatted ids the app
    # generates, ascending string order doubles as chronological sidebar order.
    assert ids == [
        "2024-01-15 09:30:00",
        "2024-02-20 18:45:00",
        "2024-03-01 10:00:00",
    ]


def test_get_setting_persists_default_on_first_read(db_manager):
    repo = db_manager.settings_repo

    first = repo.get_setting("chat_memory_length", 4)
    assert int(first) == 4

    # Read-through write: the default was persisted, so a different fallback
    # is ignored. Asserted via int() coercion throughout - stored values may
    # come back as strings and every consumer in app.py coerces, so the
    # persisted VALUE is the contract, not the return type.
    assert int(repo.get_setting("chat_memory_length", 99)) == 4


def test_update_setting_overwrites(db_manager):
    repo = db_manager.settings_repo

    repo.update_setting("chunk_size", 2048)
    assert int(repo.get_setting("chunk_size", 1)) == 2048

    repo.update_setting("chunk_size", 512)
    assert int(repo.get_setting("chunk_size", 1)) == 512


def test_data_persists_across_manager_instances(tmp_path):
    db_path = str(tmp_path / "persist.db")

    first = DatabaseManager(db_path)
    first.message_repo.save_message(SESSION, "user", "text", "survives restart")
    first.settings_repo.update_setting("chunk_size", 2048)
    first.close()

    # A fresh manager re-runs _initialize_database on the existing file:
    # chat history surviving an app restart is a user-facing promise.
    second = DatabaseManager(db_path)
    try:
        messages = second.message_repo.load_messages(SESSION)
        assert [m["content"] for m in messages] == ["survives restart"]
        assert int(second.settings_repo.get_setting("chunk_size", 1)) == 2048
    finally:
        second.close()
