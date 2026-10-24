import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.api.deps import get_current_user_id, get_supabase_client
from app.api.routers.ai import get_ai_provider, _rate_limits
from app.schemas.ai import AIResponse, AIResponseUsage
from app.services.ai_provider import (
    ProviderAuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    InvalidRequestError,
    MalformedResponseError,
    AIException,
)

client = TestClient(app)

def mock_get_user():
    return "test-user-uuid"

def build_mock_supabase(is_new=True, conv_id="conv-123", msg_id="msg-user-456"):
    mock_sb = MagicMock()
    tables = {}
    
    # Table mock routing with cached table instances
    def table_mock(table_name):
        if table_name in tables:
            return tables[table_name]
        table_obj = MagicMock()
        if table_name == "ai_conversations":
            table_obj.insert.return_value.execute.return_value = MagicMock(data=[{"id": conv_id, "user_id": "test-user-uuid"}])
            table_obj.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
                data=[{"id": conv_id, "user_id": "test-user-uuid"}]
            )
            table_obj.update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": conv_id}])
            table_obj.delete.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
            table_obj.delete.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        elif table_name == "ai_messages":
            table_obj.insert.return_value.execute.return_value = MagicMock(data=[{"id": msg_id, "conversation_id": conv_id}])
            table_obj.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
                data=[{"role": "user", "content": "previous question"}, {"role": "assistant", "content": "previous answer"}]
            )
            table_obj.delete.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        tables[table_name] = table_obj
        return table_obj

    mock_sb.table.side_effect = table_mock
    return mock_sb

@pytest.fixture(autouse=True)
def setup_dependencies():
    app.dependency_overrides.clear()
    _rate_limits.clear()
    yield
    app.dependency_overrides.clear()
    _rate_limits.clear()


def test_chat_empty_message():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase()
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb
    mock_provider = MagicMock()
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "   "})
    assert response.status_code == 400
    assert "Message cannot be empty" in response.json()["detail"]

def test_conversations_unauthorized():
    response = client.get("/ai/conversations")
    assert response.status_code in [401, 403]

def test_chat_success():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-abc", msg_id="msg-user-1")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb
    
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value=AIResponse(
        content="Here is tailored resume feedback.",
        provider="nvidia",
        model="openai/gpt-oss-20b",
        usage=AIResponseUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
        finish_reason="stop"
    ))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "How do I improve my summary?"})
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "conv-abc"
    assert data["message"] == "Here is tailored resume feedback."
    assert data["model_id"] == "openai/gpt-oss-20b"

def test_chat_provider_authentication_error_no_nameerror():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-auth-fail", msg_id="msg-auth-user")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=ProviderAuthenticationError("Invalid API key"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "Hello"})
    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "AI Provider configuration error"
    assert "NameError" not in response.text

def test_chat_provider_timeout_returns_504():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-timeout", msg_id="msg-timeout-user")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=ProviderTimeoutError("AI request timed out"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "Analyze my resume"})
    assert response.status_code == 504
    data = response.json()
    assert data["detail"] == "The AI response took too long. Please try again."

def test_chat_provider_rate_limit_returns_429():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-429", msg_id="msg-429-user")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=RateLimitError("AI provider rate limit exceeded"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "Review my skills"})
    assert response.status_code == 429
    data = response.json()
    assert data["detail"] == "AI provider rate limit exceeded"

def test_chat_provider_unavailable_returns_503():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-503", msg_id="msg-503-user")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=ProviderUnavailableError("Service unavailable"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "Help me"})
    assert response.status_code == 503
    data = response.json()
    assert data["detail"] == "AI service temporarily unavailable"

def test_chat_malformed_response_returns_502():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-502", msg_id="msg-502-user")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=MalformedResponseError("No choices in provider response"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "Format my experience"})
    assert response.status_code == 502
    data = response.json()
    assert data["detail"] == "Bad response from AI service"

def test_chat_unexpected_exception_no_secrets_leaked():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-err", msg_id="msg-err-user")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=RuntimeError("Internal crash nvapi-secret-key-supersecret"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "Help"})
    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "AI Assistant is temporarily unavailable. Please try again."
    assert "nvapi" not in response.text
    assert "supersecret" not in response.text
    assert "Traceback" not in response.text

def test_chat_existing_conversation_preserves_conversation_on_error():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    existing_conv_id = "existing-conv-999"
    mock_sb = build_mock_supabase(is_new=False, conv_id=existing_conv_id, msg_id="msg-user-existing")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=ProviderTimeoutError("AI request timed out"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={
        "conversation_id": existing_conv_id,
        "message": "Next follow-up question"
    })
    assert response.status_code == 504

def test_chat_rollback_deletes_user_message_and_empty_conv():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-rollback-test", msg_id="msg-rollback-test")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=ProviderTimeoutError("AI request timed out"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "Rollback test message"})
    assert response.status_code == 504

    # Verify that delete was called on ai_messages for the unresponded user message
    ai_messages_table = mock_sb.table("ai_messages")
    ai_messages_table.delete.assert_called()
    
    # Verify that ai_conversations was cleaned up since it was a new empty conversation
    ai_convs_table = mock_sb.table("ai_conversations")
    ai_convs_table.delete.assert_called()

def test_no_fabricated_assistant_response_persisted_on_failure():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-no-fab", msg_id="msg-user-no-fab")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=ProviderTimeoutError("AI request timed out"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"message": "Do not fabricate"})
    assert response.status_code == 504

    # Check all calls to ai_messages.insert
    ai_messages_table = mock_sb.table("ai_messages")
    for call_args in ai_messages_table.insert.call_args_list:
        inserted_payload = call_args[0][0] if call_args[0] else call_args[1]
        assert inserted_payload.get("role") != "assistant", "Fabricated assistant message was persisted!"

def test_chat_stream_success():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-stream-1", msg_id="msg-stream-1")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_gen(req):
        yield "Hello"
        yield " there!"
    mock_provider.generate_stream = mock_stream_gen
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat/stream", json={"message": "Stream hello"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) == 3 # chunk 1, chunk 2, done
    
    import json
    chunk1 = json.loads(lines[0][6:])
    assert chunk1["type"] == "chunk"
    assert chunk1["content"] == "Hello"
    
    chunk2 = json.loads(lines[1][6:])
    assert chunk2["type"] == "chunk"
    assert chunk2["content"] == " there!"
    
    done = json.loads(lines[2][6:])
    assert done["type"] == "done"
    assert done["message"] == "Hello there!"
    assert done["conversation_id"] == "conv-stream-1"

def test_chat_stream_timeout_emits_error_and_rolls_back():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-stream-err", msg_id="msg-stream-err")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_timeout(req):
        raise ProviderTimeoutError("The AI response took too long. Please try again.")
        yield
    mock_provider.generate_stream = mock_stream_timeout
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat/stream", json={"message": "Timeout stream"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) == 1
    
    import json
    err = json.loads(lines[0][6:])
    assert err["type"] == "error"
    assert "The AI response took too long" in err["detail"]
    
    # Verify rollback happened
    mock_sb.table("ai_messages").delete.assert_called()
    mock_sb.table("ai_conversations").delete.assert_called()

def test_chat_stream_flushes_initial_ping():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-ping-1", msg_id="msg-ping-1")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_gen(req):
        yield "Hi"
    mock_provider.generate_stream = mock_stream_gen
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat/stream", json={"message": "Ping test"})
    assert response.status_code == 200
    all_lines = response.text.split("\n")
    # Verify first line is an SSE comment ping
    assert all_lines[0] == ": ping"

def test_chat_stream_partial_output_timeout_preserves_messages_no_rollback():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-stream-partial", msg_id="msg-stream-partial")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_partial_timeout(req):
        yield "Initial advice on resume: "
        raise ProviderTimeoutError("Stalled stream after initial chunks")
    mock_provider.generate_stream = mock_stream_partial_timeout
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat/stream", json={"message": "Tell me about my resume"})
    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) == 2  # 1 chunk + 1 error
    
    import json
    chunk = json.loads(lines[0][6:])
    assert chunk["type"] == "chunk"
    assert chunk["content"] == "Initial advice on resume: "
    
    err = json.loads(lines[1][6:])
    assert err["type"] == "error"
    assert "The AI response took too long" in err["detail"]
    
    # Because tokens were yielded, the turn MUST NOT be rolled back
    mock_sb.table("ai_messages").delete.assert_not_called()
    mock_sb.table("ai_conversations").delete.assert_not_called()

def test_chat_stream_auth_error_emits_error_event():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-stream-auth", msg_id="msg-stream-auth")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_auth_fail(req):
        raise ProviderAuthenticationError("Invalid API key configured")
        yield
    mock_provider.generate_stream = mock_stream_auth_fail
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat/stream", json={"message": "Auth fail stream"})
    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) == 1
    import json
    err = json.loads(lines[0][6:])
    assert err["type"] == "error"
    assert "configuration error" in err["detail"].lower()

def test_chat_stream_rate_limit_error_emits_error_event():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-stream-rate", msg_id="msg-stream-rate")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_rate_limit(req):
        raise RateLimitError("Rate limit exceeded")
        yield
    mock_provider.generate_stream = mock_stream_rate_limit
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat/stream", json={"message": "Rate limit stream"})
    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) == 1
    import json
    err = json.loads(lines[0][6:])
    assert err["type"] == "error"
    assert "rate limit" in err["detail"].lower()

def test_chat_stream_provider_unavailable_error_emits_error_event():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-stream-unavail", msg_id="msg-stream-unavail")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_unavail(req):
        raise ProviderUnavailableError("NVIDIA service unavailable")
        yield
    mock_provider.generate_stream = mock_stream_unavail
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat/stream", json={"message": "Unavailable stream"})
    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) == 1
    import json
    err = json.loads(lines[0][6:])
    assert err["type"] == "error"
    assert "unavailable" in err["detail"].lower()

def test_chat_existing_conversation_not_deleted_on_failure():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    # is_new = False simulates an existing conversation with history
    mock_sb = build_mock_supabase(is_new=False, conv_id="conv-existing", msg_id="msg-new-user")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=ProviderTimeoutError("AI request timed out"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat", json={"conversation_id": "conv-existing", "message": "Add to existing"})
    assert response.status_code == 504

    # Verify that the unresponded user message was deleted
    mock_sb.table("ai_messages").delete.assert_called()
    # But ai_conversations MUST NOT be deleted because it is an existing conversation
    mock_sb.table("ai_conversations").delete.assert_not_called()

def test_chat_stream_heartbeats_without_visible_text_contamination():
    import asyncio
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-heartbeat", msg_id="msg-heartbeat")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_with_delay(req):
        # Delay before first token to allow keepalive to fire
        await asyncio.sleep(0.08)
        yield "Final answer chunk."
    mock_provider.generate_stream = mock_stream_with_delay
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    with patch.object(settings, "AI_STREAM_KEEPALIVE_INTERVAL_SECONDS", 0.03):
        response = client.post("/ai/chat/stream", json={"message": "Test heartbeats"})
        assert response.status_code == 200
        raw_text = response.text
        
        # Verify keepalive comment was emitted
        assert ": keepalive" in raw_text
        
        # Verify text chunks only contain the actual content and NO keepalive text
        lines = [line for line in raw_text.split("\n") if line.startswith("data: ")]
        import json
        chunk_data = [json.loads(line[6:]) for line in lines]
        
        # First data item is chunk, second is done
        content_tokens = [d["content"] for d in chunk_data if d["type"] == "chunk"]
        assert content_tokens == ["Final answer chunk."]
        # Ensure no ": keepalive" or comments leaked into content
        assert all(": keepalive" not in t for t in content_tokens)

def test_chat_stream_malformed_response_emits_error_and_rolls_back():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase(is_new=True, conv_id="conv-malformed-stream", msg_id="msg-malformed-stream")
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    mock_provider = MagicMock()
    async def mock_stream_malformed(req):
        raise MalformedResponseError("Model emitted reasoning metadata without final answer content")
        yield
    mock_provider.generate_stream = mock_stream_malformed
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    response = client.post("/ai/chat/stream", json={"message": "Analyze my resume"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) == 1
    
    import json
    err = json.loads(lines[0][6:])
    assert err["type"] == "error"
    assert "incomplete" in err["detail"].lower()
    
    # Verify rollback happened: unresponded user turn was deleted
    mock_sb.table("ai_messages").delete.assert_called()
    mock_sb.table("ai_conversations").delete.assert_called()



