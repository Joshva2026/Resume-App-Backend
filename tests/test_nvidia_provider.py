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
            
            assert "The AI response took too long" in str(exc_info.value)
            assert mock_post.call_count == 2

@pytest.mark.asyncio
async def test_nvidia_read_timeout_no_retry(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ReadTimeout("Read timed out")
        req = AIRequest(system_prompt="Test", user_prompt="Test")
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await provider.generate(req)
        assert "The AI response took too long" in str(exc_info.value)
        assert mock_post.call_count == 1  # Verify no redundant retry on read timeout

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
            
            assert mock_post.call_count == 2  # Retries on 429

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
            
            assert mock_post.call_count == 2  # Retries on 500

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

@pytest.mark.asyncio
async def test_nvidia_read_timeout(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ReadTimeout("Read timed out")
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            req = AIRequest(system_prompt="Test", user_prompt="Test")
            
            with pytest.raises(ProviderTimeoutError) as exc_info:
                await provider.generate(req)
                
            assert "The AI response took too long" in str(exc_info.value)
            assert mock_post.call_count == 1  # Verified: do not perform redundant expensive retries on read timeout

@pytest.mark.asyncio
async def test_nvidia_malformed_empty_choices(provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "123", "choices": []}
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        req = AIRequest(system_prompt="Test", user_prompt="Test")
        
        from app.services.ai_provider import MalformedResponseError
        with pytest.raises(MalformedResponseError):
            await provider.generate(req)

@pytest.mark.asyncio
async def test_nvidia_malformed_missing_content(provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "123",
        "choices": [{"message": {"content": ""}, "finish_reason": "stop"}]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        req = AIRequest(system_prompt="Test", user_prompt="Test")
        
        from app.services.ai_provider import MalformedResponseError
        with pytest.raises(MalformedResponseError):
            await provider.generate(req)

@pytest.mark.asyncio
async def test_effective_timeout_bounds():
    with patch("app.services.nvidia_provider.settings") as mock_settings:
        # Default / within bounds
        mock_settings.AI_CONNECT_TIMEOUT_SECONDS = 10
        mock_settings.AI_REQUEST_TIMEOUT_SECONDS = 70
        mock_settings.AI_STREAM_READ_TIMEOUT_SECONDS = 85
        mock_settings.AI_INTER_CHUNK_TIMEOUT_SECONDS = 25
        conn, read, stream_read, inter_chunk = NvidiaAIProvider.get_effective_timeouts()
        assert conn == 10.0
        assert read == 70.0
        assert stream_read == 85.0
        assert inter_chunk == 25.0
        
        # Upper bound enforcement (< 90.0s for stream read to stay below Render's 100s deadline)
        mock_settings.AI_STREAM_READ_TIMEOUT_SECONDS = 150
        mock_settings.AI_INTER_CHUNK_TIMEOUT_SECONDS = 60
        conn, read, stream_read, inter_chunk = NvidiaAIProvider.get_effective_timeouts()
        assert stream_read == 90.0
        assert inter_chunk == 45.0
        
        # Lower bound enforcement
        mock_settings.AI_REQUEST_TIMEOUT_SECONDS = 2
        mock_settings.AI_STREAM_READ_TIMEOUT_SECONDS = 2
        mock_settings.AI_INTER_CHUNK_TIMEOUT_SECONDS = 1
        conn, read, stream_read, inter_chunk = NvidiaAIProvider.get_effective_timeouts()
        assert read == 5.0
        assert stream_read == 10.0
        assert inter_chunk == 5.0

@pytest.mark.asyncio
async def test_nvidia_generate_stream_success(provider):
    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            yield ': ping'
            yield 'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}'
            yield ''
            yield 'data: {"id":"2","choices":[{"delta":{"content":" World"}}]}'
            yield 'data: {"id":"3","choices":[]}'  # usage or empty choices
            yield 'data: [DONE]'

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.return_value = MockStreamResponse()
        req = AIRequest(system_prompt="System", user_prompt="Hi")
        tokens = []
        async for chunk in provider.generate_stream(req):
            tokens.append(chunk)
        assert tokens == ["Hello", " World"]

@pytest.mark.asyncio
async def test_nvidia_generate_stream_read_timeout(provider):
    class MockStreamTimeoutResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            raise httpx.ReadTimeout("Streaming read timeout")
            yield  # make it a generator

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.return_value = MockStreamTimeoutResponse()
        req = AIRequest(system_prompt="System", user_prompt="Hi")
        with pytest.raises(ProviderTimeoutError) as exc_info:
            async for _ in provider.generate_stream(req):
                pass
        assert "The AI response took too long" in str(exc_info.value)

@pytest.mark.asyncio
async def test_nvidia_generate_stream_reasoning_tokens_and_content(provider):
    class MockStreamReasoningResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            # Model emits reasoning tokens first
            yield 'data: {"choices":[{"delta":{"reasoning_content":"Thinking about user resume"}}]}'
            yield 'data: {"choices":[{"delta":{"reasoning_content":" Still analyzing skills"}}]}'
            # Then model emits final content
            yield 'data: {"choices":[{"delta":{"content":"Add Kotlin to your skills section."}}]}'
            yield 'data: [DONE]'

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.return_value = MockStreamReasoningResponse()
        req = AIRequest(system_prompt="System", user_prompt="How do I improve my resume?")
        tokens = []
        async for chunk in provider.generate_stream(req):
            tokens.append(chunk)
        # Verify reasoning is filtered out from user content stream
        assert tokens == ["Add Kotlin to your skills section."]

@pytest.mark.asyncio
async def test_nvidia_generate_stream_error_chunk_raises_exception(provider):
    class MockStreamErrorChunkResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            yield 'data: {"error":{"code":429,"message":"Rate limit exceeded for model"}}'
            yield 'data: [DONE]'

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.return_value = MockStreamErrorChunkResponse()
        req = AIRequest(system_prompt="System", user_prompt="Hi")
        with pytest.raises(RateLimitError):
            async for _ in provider.generate_stream(req):
                pass


