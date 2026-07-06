"""Contract tests for endpoint routing and request construction.

Everything is exercised through the public ChatAPIHandler.chat() entry point;
HTTP and vector-db access are patched at the seams. Namespace discipline
matters (see TESTING.md): openai_api_key, config and load_vectordb must be
patched on the chat_api_handler module because they are bound there at import
time.
"""
import base64
from types import SimpleNamespace

import pytest

import chat_api_handler
from chat_api_handler import ChatAPIHandler

OLLAMA_RESPONSE = {"message": {"content": "the answer"}}
OPENAI_RESPONSE = {"choices": [{"message": {"content": "the answer"}}]}


@pytest.fixture
def ollama_config(monkeypatch):
    # Accepted churn (see TESTING.md): config is bound at import time in
    # chat_api_handler, so this fixture patches that binding. A refactor to
    # call-time load_config() would require updating this fixture, not the
    # tests.
    monkeypatch.setattr(
        chat_api_handler,
        "config",
        {"ollama": {"base_url": "http://test-ollama:11434"}},
    )


@pytest.fixture
def openai_key(monkeypatch):
    # Accepted churn (see TESTING.md): os.getenv ran once at module import,
    # so monkeypatch.setenv would be useless. A refactor to call-time getenv
    # would require updating this fixture, not the tests.
    monkeypatch.setattr(chat_api_handler, "openai_api_key", "sk-test")


def test_chat_ollama_endpoint_posts_to_ollama(fake_session_state, captured_post, ollama_config):
    fake_session_state.update(endpoint_to_use="ollama", model_to_use="llama3")
    post = captured_post(OLLAMA_RESPONSE)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    answer = ChatAPIHandler.chat(user_input="new question", chat_history=list(history))

    assert answer == "the answer"
    assert len(post.calls) == 1
    call = post.calls[0]
    assert call["url"] == "http://test-ollama:11434/api/chat"
    assert call["data"] is None  # the body must travel as json=, not form data
    assert call["json"]["model"] == "llama3"
    assert call["json"]["stream"] is False
    assert call["json"]["messages"] == history + [{"role": "user", "content": "new question"}]


def test_chat_openai_endpoint_posts_to_openai(fake_session_state, captured_post, openai_key):
    fake_session_state.update(endpoint_to_use="openai", model_to_use="gpt-4o")
    post = captured_post(OPENAI_RESPONSE)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    answer = ChatAPIHandler.chat(user_input="a question", chat_history=list(history))

    assert answer == "the answer"
    assert len(post.calls) == 1
    call = post.calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["data"] is None  # the body must travel as json=, not form data
    assert call["json"]["model"] == "gpt-4o"
    # Non-empty prior history: dropping conversation memory must fail here.
    assert call["json"]["messages"] == history + [{"role": "user", "content": "a question"}]


def test_chat_unknown_endpoint_raises_value_error(fake_session_state):
    fake_session_state.update(endpoint_to_use="azure", model_to_use="anything")

    # The autouse no_network fixture guarantees this raises before any HTTP
    # attempt - a network call would surface as RuntimeError, not ValueError.
    with pytest.raises(ValueError):
        ChatAPIHandler.chat(user_input="q", chat_history=[])


def test_ollama_image_chat_wire_format(fake_session_state, captured_post, ollama_config):
    fake_session_state.update(endpoint_to_use="ollama", model_to_use="llava")
    post = captured_post(OLLAMA_RESPONSE)
    image = b"\x89PNG\r\n\x1a\n fake image"
    history = [{"role": "user", "content": "earlier turn"}]

    ChatAPIHandler.chat(user_input="what is shown here?", chat_history=list(history), image=image)

    messages = post.calls[0]["json"]["messages"]
    assert messages[:-1] == history  # prior turns must still reach the model
    # Ollama vision wire format: bare base64 in an "images" list next to the
    # text content.
    assert messages[-1] == {
        "role": "user",
        "content": "what is shown here?",
        "images": [base64.b64encode(image).decode("utf-8")],
    }


def test_openai_image_chat_data_url_wire_format(fake_session_state, captured_post, openai_key):
    fake_session_state.update(endpoint_to_use="openai", model_to_use="gpt-4o")
    post = captured_post(OPENAI_RESPONSE)
    image = b"\x89PNG\r\n\x1a\n fake image"
    history = [{"role": "user", "content": "earlier turn"}]

    ChatAPIHandler.chat(user_input="describe this", chat_history=list(history), image=image)

    messages = post.calls[0]["json"]["messages"]
    assert messages[:-1] == history  # prior turns must still reach the model
    message = messages[-1]
    assert message["role"] == "user"
    text_parts = [p for p in message["content"] if p.get("type") == "text"]
    image_parts = [p for p in message["content"] if p.get("type") == "image_url"]
    assert [p["text"] for p in text_parts] == ["describe this"]
    # OpenAI vision wire format: a data URL whose payload roundtrips back to
    # the input bytes (roundtrip assert, not golden string).
    url = image_parts[0]["image_url"]["url"]
    prefix = "data:image/jpeg;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix):]) == image


def test_pdf_chat_stuffs_retrieved_context_into_prompt(
    fake_session_state, captured_post, ollama_config, monkeypatch
):
    chunks = [
        SimpleNamespace(page_content="first retrieved chunk"),
        SimpleNamespace(page_content="second retrieved chunk"),
    ]
    search_calls = []

    def fake_similarity_search(query, k):
        search_calls.append((query, k))
        return chunks

    # Must patch the chat_api_handler namespace: the from-import at the top of
    # the module means patching vectordb_handler.load_vectordb does nothing.
    monkeypatch.setattr(
        chat_api_handler,
        "load_vectordb",
        lambda: SimpleNamespace(similarity_search=fake_similarity_search),
    )
    fake_session_state.update(
        endpoint_to_use="ollama",
        model_to_use="llama3",
        pdf_chat=True,
        retrieved_documents=2,
    )
    post = captured_post(OLLAMA_RESPONSE)
    history = [{"role": "user", "content": "earlier turn"}]

    ChatAPIHandler.chat(user_input="what does the paper say?", chat_history=list(history))

    assert search_calls == [("what does the paper say?", 2)]
    messages = post.calls[0]["json"]["messages"]
    assert messages[:-1] == history  # prior turns must still reach the model
    prompt = messages[-1]["content"]
    # Substring asserts only - the exact prompt template wording is not a
    # contract, but every retrieved chunk and the question must reach the model.
    assert "first retrieved chunk" in prompt
    assert "second retrieved chunk" in prompt
    assert "what does the paper say?" in prompt


def test_ollama_api_call_surfaces_error_string(fake_session_state, captured_post, ollama_config):
    fake_session_state.update(endpoint_to_use="ollama", model_to_use="missing-model")
    # Ollama error values are plain strings (unlike OpenAI's nested objects).
    captured_post({"error": "model 'missing-model' not found"})

    answer = ChatAPIHandler.chat(user_input="q", chat_history=[])

    assert isinstance(answer, str)
    assert "model 'missing-model' not found" in answer


def test_openai_api_call_surfaces_error_message(fake_session_state, captured_post, openai_key):
    fake_session_state.update(endpoint_to_use="openai", model_to_use="gpt-4o")
    # OpenAI error values are nested objects with a message field.
    captured_post(
        {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}}
    )

    answer = ChatAPIHandler.chat(user_input="q", chat_history=[])

    assert isinstance(answer, str)
    assert "Incorrect API key provided" in answer
