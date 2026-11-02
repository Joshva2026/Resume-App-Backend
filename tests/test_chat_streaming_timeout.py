import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.api.deps import get_current_user_id, get_supabase_client
from app.api.routers.ai import get_ai_provider, _rate_limits
from app.schemas.ai import AIRequest
from app.services.nvidia_provider import NvidiaAIProvider
from app.services.ai_provider import (
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderAuthenticationError,
    MalformedResponseError
)

client = TestClient(app)

def mock_get_user():
    return "test-user-uuid"

def build_mock_supabase(conv_id="conv-streaming-123", msg_id="msg-user-456"):
    mock_sb = MagicMock()
    tables = {}
    
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
            table_obj.delete.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        elif table_name == "ai_messages":
            table_obj.insert.return_value.execute.return_value = MagicMock(data=[{"id": msg_id, "conversation_id": conv_id}])
            table_obj.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
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


# =========================================================================
# 1. Successful Streaming Test
# =========================================================================
def test_chat_stream_success():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase()
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    class FakeProvider:
        async def generate_stream(self, req: AIRequest):
            yield "Hello! "
            yield "Here is your "
            yield "resume advice."

    app.dependency_overrides[get_ai_provider] = lambda: FakeProvider()

    response = client.post("/ai/chat/stream", json={"message": "Help me with my summary."})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    
    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    assert ": ping" in lines

    data_lines = [l[6:] for l in lines if l.startswith("data: ")]
    parsed_events = [json.loads(d) for d in data_lines]

    chunk_contents = [e["content"] for e in parsed_events if e["type"] == "chunk"]
    assert chunk_contents == ["Hello! ", "Here is your ", "resume advice."]

    done_events = [e for e in parsed_events if e["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["message"] == "Hello! Here is your resume advice."
    assert done_events[0]["conversation_id"] == "conv-streaming-123"

    # Verify assistant message was persisted in database
    mock_sb.table("ai_messages").insert.assert_called()


# =========================================================================
# 2. First-Token Timeout & Safe Fallback in Provider
# =========================================================================
@pytest.mark.asyncio
async def test_provider_first_token_timeout_triggers_fallback():
    provider = NvidiaAIProvider()
    
    primary_called = False
    fallback_called = False

    class MockStreamPrimaryHung:
        status_code = 200
        async def __aenter__(self):
            # Hang beyond first token timeout
            await asyncio.sleep(0.2)
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            yield  # pragma: no cover

    class MockStreamFallbackSuccess:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Fallback tokens received successfully"}}]}'
            yield 'data: [DONE]'

    def mock_stream(method, url, json=None, **kwargs):
        nonlocal primary_called, fallback_called
        model = json.get("model")
        if model == "openai/gpt-oss-20b":
            primary_called = True
            return MockStreamPrimaryHung()
        elif model == "nvidia/nemotron-3.5-lightning-30b-a3b":
            fallback_called = True
            return MockStreamFallbackSuccess()
        raise ValueError(f"Unexpected model: {model}")

    with patch("app.services.nvidia_provider.settings") as mock_settings, \
         patch("httpx.AsyncClient.stream", side_effect=mock_stream):
        
        mock_settings.NVIDIA_API_KEY = "test_key"
        mock_settings.NVIDIA_BASE_URL = "https://api.nvidia.com"
        mock_settings.NVIDIA_MODEL = "openai/gpt-oss-20b"
        mock_settings.NVIDIA_FALLBACK_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
        mock_settings.AI_FIRST_TOKEN_TIMEOUT_SECONDS = 0.05  # Fast 50ms timeout for test
        mock_settings.AI_CONNECT_TIMEOUT_SECONDS = 10
        mock_settings.AI_STREAM_READ_TIMEOUT_SECONDS = None
        mock_settings.AI_INTER_CHUNK_TIMEOUT_SECONDS = None

        req = AIRequest(
            system_prompt="System",
            user_prompt="Help",
            model_id="openai/gpt-oss-20b"
        )

        tokens = []
        async for chunk in provider.generate_stream(req):
            tokens.append(chunk)

        assert primary_called is True
        assert fallback_called is True
        assert tokens == ["Fallback tokens received successfully"]
        # Verify request.model_id was updated to fallback model
        assert req.model_id == "nvidia/nemotron-3.5-lightning-30b-a3b"


# =========================================================================
# 3. Provider Errors Communicate Structured SSE Error & Rollback
# =========================================================================
def test_chat_stream_first_token_timeout_sse_error_and_rollback():
    app.dependency_overrides[get_current_user_id] = mock_get_user
    mock_sb = build_mock_supabase()
    app.dependency_overrides[get_supabase_client] = lambda: mock_sb

    class FailingProvider:
        async def generate_stream(self, req: AIRequest):
            raise ProviderTimeoutError("The AI response took too long. Please try again.")
            yield  # generator

    app.dependency_overrides[get_ai_provider] = lambda: FailingProvider()

    response = client.post("/ai/chat/stream", json={"message": "Improve my skills."})
    assert response.status_code == 200
    
    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    data_lines = [l[6:] for l in lines if l.startswith("data: ")]
    events = [json.loads(d) for d in data_lines]

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["detail"] == "The AI response took too long. Please try again."

    # Verify rollback was called for unresponded turn
    mock_sb.table("ai_messages").delete().eq().execute.assert_called()


# =========================================================================
# 4. Mid-Stream Interruption: No Duplicate Fallback, Structured Error
# =========================================================================
@pytest.mark.asyncio
async def test_provider_mid_stream_error_does_not_duplicate_fallback():
    provider = NvidiaAIProvider()
    fallback_attempted = False

    class MockStreamMidFail:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"First partial chunk. "}}]}'
            # Model crashes mid-stream after emitting content
            raise httpx.ReadTimeout("Mid-stream read timeout")

    def mock_stream(method, url, json=None, **kwargs):
        nonlocal fallback_attempted
        model = json.get("model")
        if model == "openai/gpt-oss-20b":
            return MockStreamMidFail()
        else:
            fallback_attempted = True
            raise ValueError("Fallback should not be attempted when chunks already yielded!")

    with patch("app.services.nvidia_provider.settings") as mock_settings, \
         patch("httpx.AsyncClient.stream", side_effect=mock_stream):
        
        mock_settings.NVIDIA_API_KEY = "test_key"
        mock_settings.NVIDIA_BASE_URL = "https://api.nvidia.com"
        mock_settings.NVIDIA_MODEL = "openai/gpt-oss-20b"
        mock_settings.NVIDIA_FALLBACK_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
        mock_settings.AI_FIRST_TOKEN_TIMEOUT_SECONDS = 5.0
        mock_settings.AI_CONNECT_TIMEOUT_SECONDS = 10
        mock_settings.AI_STREAM_READ_TIMEOUT_SECONDS = None
        mock_settings.AI_INTER_CHUNK_TIMEOUT_SECONDS = None

        req = AIRequest(system_prompt="System", user_prompt="Test", model_id="openai/gpt-oss-20b")

        received = []
        with pytest.raises(ProviderTimeoutError) as exc_info:
            async for chunk in provider.generate_stream(req):
                received.append(chunk)

        # Chunks up to error were yielded
        assert received == ["First partial chunk. "]
        # Fallback must NOT have been called
        assert fallback_attempted is False
        assert "The AI response took too long" in str(exc_info.value)


# =========================================================================
# 5. Client Disconnect Handling
# =========================================================================
@pytest.mark.asyncio
async def test_chat_stream_client_disconnect():
    mock_sb = build_mock_supabase()
    producer_cancelled = False

    class HangingProvider:
        async def generate_stream(self, req: AIRequest):
            nonlocal producer_cancelled
            try:
                for i in range(100):
                    await asyncio.sleep(0.05)
                    yield f"chunk_{i}"
            except asyncio.CancelledError:
                producer_cancelled = True
                raise

    from app.api.routers.chat import chat_stream, ChatMessageRequest
    from starlette.requests import Request

    disconnect_called = False
    async def fake_receive():
        return {"type": "http.disconnect"}

    mock_request = MagicMock(spec=Request)
    # Simulate client disconnects immediately on first check
    mock_request.is_disconnected = AsyncMock(return_value=True)

    chat_req = ChatMessageRequest(message="Hello")
    response = await chat_stream(
        request=mock_request,
        chat_req=chat_req,
        user_id="test-user-uuid",
        supabase=mock_sb,
        provider=HangingProvider()
    )

    # Consume event generator
    events = []
    async for event in response.body_iterator:
        events.append(event)

    # Initial ping is sent, then aborts cleanly on client disconnect
    assert ": ping\n\n" in events
    # Unresponded user message rolled back on disconnect
    mock_sb.table("ai_messages").delete().eq().execute.assert_called()


# =========================================================================
# 6. First-Token Timeout Bounds Enforcement
# =========================================================================
def test_first_token_timeout_bounds():
    with patch("app.services.nvidia_provider.settings") as mock_settings:
        mock_settings.AI_FIRST_TOKEN_TIMEOUT_SECONDS = 15.0
        assert NvidiaAIProvider.get_first_token_timeout() == 15.0

        # Clamped lower bound (minimum 3.0s)
        mock_settings.AI_FIRST_TOKEN_TIMEOUT_SECONDS = 1.0
        assert NvidiaAIProvider.get_first_token_timeout() == 3.0

        # Clamped upper bound (maximum 45.0s)
        mock_settings.AI_FIRST_TOKEN_TIMEOUT_SECONDS = 100.0
        assert NvidiaAIProvider.get_first_token_timeout() == 45.0

        # Invalid string defaults safely
        mock_settings.AI_FIRST_TOKEN_TIMEOUT_SECONDS = "invalid"
        assert NvidiaAIProvider.get_first_token_timeout() == 15.0

