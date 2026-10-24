import pytest
import httpx
import json
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.grok_provider import XAIGrokAIProvider, GrokAIProvider
from app.schemas.ai import AIRequest, AIResponse
from app.services.ai_provider import (
    ProviderAuthenticationError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    InvalidRequestError,
    RateLimitError,
    MalformedResponseError
)
from app.core.config import settings

@pytest.fixture
def grok_provider():
    with patch.object(settings, "XAI_API_KEY", "xai-test-key-1234567890abcdefghijklmnopqrstuvwxyz"), \
         patch.object(settings, "XAI_BASE_URL", "https://api.x.ai/v1"), \
         patch.object(settings, "ATS_GROK_MODEL", "grok-2-latest"):
        return XAIGrokAIProvider()

@pytest.mark.asyncio
async def test_xai_grok_missing_api_key_raises_unavailable():
    with patch.object(settings, "XAI_API_KEY", None):
        provider = XAIGrokAIProvider()
        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await provider.generate(req)
        assert "missing XAI_API_KEY" in str(exc_info.value)

@pytest.mark.asyncio
async def test_xai_grok_successful_generate(grok_provider):
    mock_payload = {
        "id": "chatcmpl-xai-test",
        "object": "chat.completion",
        "model": "grok-2-latest",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>Analyzing candidate...</think>{\"matched_keywords\": [\"Kotlin\", \"Jetpack Compose\"]}"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "total_tokens": 160
        }
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload
    mock_resp.text = json.dumps(mock_payload)

    with patch.object(grok_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(
            system_prompt="expert recruiter system prompt",
            user_prompt="evaluate candidate resume against Android role",
            temperature=0.1,
            max_tokens=650
        )
        response = await grok_provider.generate(req)

        assert isinstance(response, AIResponse)
        assert response.provider == "xai-grok"
        assert response.model == "grok-2-latest"
        assert "<think>" not in response.content
        assert response.content == '{"matched_keywords": ["Kotlin", "Jetpack Compose"]}'
        assert response.usage.prompt_tokens == 120
        assert response.usage.completion_tokens == 40
        assert response.finish_reason == "stop"

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.x.ai/v1/chat/completions"
        sent_json = call_args[1]["json"]
        assert sent_json["model"] == "grok-2-latest"
        assert sent_json["temperature"] == 0.1
        assert len(sent_json["messages"]) == 2
        assert sent_json["messages"][0]["role"] == "system"
        assert sent_json["messages"][1]["role"] == "user"

@pytest.mark.asyncio
async def test_xai_grok_rate_limit_429(grok_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"error": {"message": "Rate limit exceeded for organization"}}
    mock_resp.text = '{"error": {"message": "Rate limit exceeded"}}'

    with patch.object(grok_provider, 'get_client') as mock_get_client, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(RateLimitError) as exc_info:
            await grok_provider.generate(req)
        assert "rate limit exceeded" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_xai_grok_auth_failure_401(grok_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": {"message": "Incorrect API key provided: xai-test-key-1234567890abcdefghijklmnopqrstuvwxyz"}}
    mock_resp.text = "Incorrect API key"

    with patch.object(grok_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderAuthenticationError) as exc_info:
            await grok_provider.generate(req)
        assert "authentication failed" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_xai_grok_gateway_timeout_504(grok_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 504
    mock_resp.json.return_value = {"error": {"message": "Gateway Timeout"}}
    mock_resp.text = "Gateway Timeout"

    with patch.object(grok_provider, 'get_client') as mock_get_client, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await grok_provider.generate(req)
        assert "gateway timed out" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_xai_grok_read_timeout_no_retry(grok_provider):
    with patch.object(grok_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ReadTimeout("Read timeout on stream")
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await grok_provider.generate(req)
        assert "took too long" in str(exc_info.value).lower()
        assert mock_client.post.call_count == 1

@pytest.mark.asyncio
async def test_xai_grok_truncated_response_raises_malformed(grok_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "1",
        "choices": [{
            "message": {"role": "assistant", "content": None},
            "finish_reason": "length"
        }]
    }
    mock_resp.text = '{"choices": [{"message": {"content": null}, "finish_reason": "length"}]}'

    with patch.object(grok_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(MalformedResponseError) as exc_info:
            await grok_provider.generate(req)
        assert "truncated" in str(exc_info.value).lower()
