import pytest
import httpx
from unittest.mock import patch, MagicMock
from app.services.nvidia_provider import NvidiaAIProvider
from app.schemas.ai import AIRequest
from app.services.ai_provider import (
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    ProviderAuthenticationError
)
import asyncio

# Create an async mock
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

@pytest.fixture
def mock_settings():
    with patch("app.services.nvidia_provider.settings") as mock:
        mock.NVIDIA_API_KEY = "test_key"
        mock.NVIDIA_BASE_URL = "https://api.nvidia.com"
        mock.NVIDIA_MODEL = "GPT-OSS-20B"
        yield mock

@pytest.fixture
def provider(mock_settings):
    return NvidiaAIProvider()

@pytest.mark.asyncio
async def test_successful_nvidia_response(provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "GPT-OSS-20B",
        "choices": [
            {
                "message": {"content": "Mock ATS feedback"},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        req = AIRequest(system_prompt="Test", user_prompt="Test Prompt")
        response = await provider.generate(req)
        
        assert response.content == "Mock ATS feedback"
        assert response.model == "GPT-OSS-20B"

@pytest.mark.asyncio
async def test_nvidia_timeout(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        
        # Patch sleep to avoid waiting during tests
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            req = AIRequest(system_prompt="Test", user_prompt="Test")
            
            with pytest.raises(ProviderTimeoutError) as exc_info:
                await provider.generate(req)
            
            assert "AI request timed out" in str(exc_info.value)
            assert mock_post.call_count == 3  # 3 retries

@pytest.mark.asyncio
async def test_nvidia_http_429(provider):
    mock_response = MagicMock()
    mock_response.status_code = 429
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            req = AIRequest(system_prompt="Test", user_prompt="Test")
            
            with pytest.raises(RateLimitError):
                await provider.generate(req)
            
            assert mock_post.call_count == 3  # Retries on 429

@pytest.mark.asyncio
async def test_nvidia_http_500(provider):
    mock_response = MagicMock()
    mock_response.status_code = 503
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            req = AIRequest(system_prompt="Test", user_prompt="Test")
            
            with pytest.raises(ProviderUnavailableError):
                await provider.generate(req)
            
            assert mock_post.call_count == 3  # Retries on 500

@pytest.mark.asyncio
async def test_nvidia_http_404(provider):
    mock_response = MagicMock()
    mock_response.status_code = 404
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        req = AIRequest(system_prompt="Test", user_prompt="Test")
        
        from app.services.ai_provider import InvalidRequestError
        with pytest.raises(InvalidRequestError):
            await provider.generate(req)
            
        assert mock_post.call_count == 1  # Should not retry 404

@pytest.mark.asyncio
async def test_nvidia_http_400(provider):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad request"
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        req = AIRequest(system_prompt="Test", user_prompt="Test")
        
        from app.services.ai_provider import InvalidRequestError
        with pytest.raises(InvalidRequestError):
            await provider.generate(req)
            
        assert mock_post.call_count == 1  # Should not retry 400

@pytest.mark.asyncio
async def test_invalid_api_key(provider):
    mock_response = MagicMock()
    mock_response.status_code = 401
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        req = AIRequest(system_prompt="Test", user_prompt="Test")
        
        with pytest.raises(ProviderAuthenticationError):
            await provider.generate(req)
            
        assert mock_post.call_count == 1  # Should not retry auth errors
