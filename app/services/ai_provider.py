from abc import ABC, abstractmethod
from app.schemas.ai import AIRequest, AIResponse

class AIException(Exception):
    """Base exception for AI provider errors"""
    pass

class ProviderAuthenticationError(AIException):
    pass

class ProviderUnavailableError(AIException):
    pass

class ProviderTimeoutError(AIException):
    pass

class InvalidRequestError(AIException):
    pass

class RateLimitError(AIException):
    pass

class MalformedResponseError(AIException):
    pass

class AIProvider(ABC):
    @abstractmethod
    async def generate(self, request: AIRequest) -> AIResponse:
        """Generate response from the AI provider"""
        pass
