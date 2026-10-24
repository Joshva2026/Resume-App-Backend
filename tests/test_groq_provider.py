import pytest
import httpx
import json
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.groq_provider import GroqAIProvider
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
def groq_provider():
    with patch.object(settings, "GROQ_API_KEY", "gsk_test_api_key_1234567890abcdef"), \
         patch.object(settings, "GROQ_BASE_URL", "https://api.groq.com/openai/v1"), \
         patch.object(settings, "ATS_GROQ_MODEL", "openai/gpt-oss-120b"):
        return GroqAIProvider()

@pytest.mark.asyncio
async def test_groq_missing_api_key_raises_unavailable():
    with patch.object(settings, "GROQ_API_KEY", None):
        provider = GroqAIProvider()
        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await provider.generate(req)
        assert "missing GROQ_API_KEY" in str(exc_info.value)

@pytest.mark.asyncio
async def test_groq_successful_generate(groq_provider):
    mock_payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "openai/gpt-oss-120b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>Processing resume...</think>{\"matched_keywords\": [\"Python\", \"Docker\"]}"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200
        }
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload
    mock_resp.text = json.dumps(mock_payload)

    with patch.object(groq_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(
            system_prompt="system evaluator",
            user_prompt="evaluate resume against JD",
            temperature=0.1,
            max_tokens=600
        )
        response = await groq_provider.generate(req)

        assert isinstance(response, AIResponse)
        assert response.provider == "groq"
        assert response.model == "openai/gpt-oss-120b"
        # Verify <think> block was removed
        assert "<think>" not in response.content
        assert response.content == '{"matched_keywords": ["Python", "Docker"]}'
        assert response.usage.prompt_tokens == 150
        assert response.usage.completion_tokens == 50
        assert response.finish_reason == "stop"

        # Verify outgoing payload structure sent to Groq
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.groq.com/openai/v1/chat/completions"
        sent_json = call_args[1]["json"]
        assert sent_json["model"] == "openai/gpt-oss-120b"
        assert sent_json["temperature"] == 0.1
        assert len(sent_json["messages"]) == 2
        assert sent_json["messages"][0]["role"] == "system"
        assert sent_json["messages"][1]["role"] == "user"

@pytest.mark.asyncio
async def test_groq_rate_limit_429(groq_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"error": {"message": "Rate limit reached for requests per minute (RPM)"}}
    mock_resp.text = '{"error": {"message": "Rate limit reached"}}'

    with patch.object(groq_provider, 'get_client') as mock_get_client, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(RateLimitError) as exc_info:
            await groq_provider.generate(req)
        assert "rate limit exceeded" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_groq_auth_failure_401(groq_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": {"message": "Invalid API Key: gsk_test_api_key_1234567890abcdef"}}
    mock_resp.text = "Invalid API Key"

    with patch.object(groq_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderAuthenticationError) as exc_info:
            await groq_provider.generate(req)
        assert "authentication failed" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_groq_gateway_timeout_504(groq_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 504
    mock_resp.json.return_value = {"error": {"message": "Gateway Timeout"}}
    mock_resp.text = "Gateway Timeout"

    with patch.object(groq_provider, 'get_client') as mock_get_client, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await groq_provider.generate(req)
        assert "gateway timed out" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_groq_server_error_503(groq_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 503
    mock_resp.json.return_value = {"error": {"message": "Service Temporarily Unavailable"}}
    mock_resp.text = "Service Unavailable"

    with patch.object(groq_provider, 'get_client') as mock_get_client, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await groq_provider.generate(req)
        assert "unavailable" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_groq_read_timeout_fails_fast(groq_provider):
    with patch.object(groq_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ReadTimeout("Read timeout occurred")
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await groq_provider.generate(req)
        assert "took too long" in str(exc_info.value).lower()
        # Ensure read timeout does not retry
        assert mock_client.post.call_count == 1

@pytest.mark.asyncio
async def test_groq_malformed_empty_choices(groq_provider):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "1", "choices": []}
    mock_resp.text = '{"choices": []}'

    with patch.object(groq_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(MalformedResponseError) as exc_info:
            await groq_provider.generate(req)
        assert "No choices" in str(exc_info.value)

@pytest.mark.asyncio
async def test_groq_truncated_response_raises_malformed(groq_provider):
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

    with patch.object(groq_provider, 'get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        req = AIRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(MalformedResponseError) as exc_info:
            await groq_provider.generate(req)
        assert "truncated" in str(exc_info.value).lower()
