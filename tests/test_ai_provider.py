import pytest
from app.schemas.ai import AIRequest
from app.services.ai_provider import ProviderUnavailableError
from app.services.nvidia_provider import NvidiaAIProvider
from app.core.config import settings

@pytest.mark.asyncio
async def test_nvidia_provider_missing_key():
    # Store original key
    original_key = settings.NVIDIA_API_KEY
    settings.NVIDIA_API_KEY = None
    
    provider = NvidiaAIProvider()
    request = AIRequest(system_prompt="sys", user_prompt="user")
    
    with pytest.raises(ProviderUnavailableError):
        await provider.generate(request)
        
    # Restore key
    settings.NVIDIA_API_KEY = original_key

def test_ai_request_validation():
    # Test valid request
    req = AIRequest(system_prompt="sys", user_prompt="user", temperature=0.5, max_tokens=100)
    assert req.temperature == 0.5
    
    # Test invalid temperature
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AIRequest(system_prompt="sys", user_prompt="user", temperature=3.0)
