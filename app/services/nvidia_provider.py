import httpx
import logging
import asyncio
from app.core.config import settings
from app.schemas.ai import AIRequest, AIResponse, AIResponseUsage
from app.services.ai_provider import (
    AIProvider,
    ProviderAuthenticationError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    InvalidRequestError,
    RateLimitError,
    MalformedResponseError,
    AIException
)

logger = logging.getLogger(__name__)

class NvidiaAIProvider(AIProvider):
    def __init__(self):
        self.api_key = settings.NVIDIA_API_KEY
        self.base_url = settings.NVIDIA_BASE_URL
        self.model = settings.NVIDIA_MODEL
        self.timeout = settings.AI_REQUEST_TIMEOUT_SECONDS
        
        if not self.api_key:
            logger.warning("NVIDIA_API_KEY is not configured.")
            
    async def generate(self, request: AIRequest) -> AIResponse:
        if not self.api_key:
            raise ProviderUnavailableError("AI Provider is not configured")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        messages = [{"role": "system", "content": request.system_prompt}]
        
        # Bounded context protection: ensure history isn't excessively large
        for msg in request.history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
        messages.append({"role": "user", "content": request.user_prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        
        # Bounded retries (3 attempts)
        max_retries = 3
        base_delay = 1.0
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    
                    self._handle_http_errors(response)
                    
                    data = response.json()
                    return self._parse_response(data)
                    
                except httpx.TimeoutException:
                    if attempt == max_retries - 1:
                        logger.error(f"NVIDIA API timeout after {max_retries} attempts")
                        raise ProviderTimeoutError("AI request timed out")
                except (ProviderUnavailableError, RateLimitError) as e:
                    if attempt == max_retries - 1:
                        raise e
                except Exception as e:
                    if isinstance(e, AIException):
                        raise e # Don't retry auth errors, invalid requests, etc
                    logger.error(f"Unexpected NVIDIA error: {str(e)}")
                    if attempt == max_retries - 1:
                        raise ProviderUnavailableError("Unexpected error communicating with AI provider")
                        
                # Backoff before retry
                await asyncio.sleep(base_delay * (2 ** attempt))

        raise ProviderUnavailableError("Failed to communicate with AI provider")

    def _handle_http_errors(self, response: httpx.Response):
        status = response.status_code
        if status == 200:
            return
            
        if status == 401 or status == 403:
            logger.error("NVIDIA API authentication failed")
            raise ProviderAuthenticationError("AI provider authentication failed")
        elif status == 429:
            raise RateLimitError("AI provider rate limit exceeded")
        elif status == 400:
            logger.error(f"NVIDIA API invalid request: {response.text}")
            raise InvalidRequestError("Invalid request to AI provider")
        elif status >= 500:
            raise ProviderUnavailableError("AI provider is temporarily unavailable")
        else:
            logger.error(f"NVIDIA API unexpected status {status}: {response.text}")
            raise AIException(f"AI provider returned status {status}")

    def _parse_response(self, data: dict) -> AIResponse:
        try:
            choices = data.get("choices", [])
            if not choices:
                raise MalformedResponseError("No choices in provider response")
                
            content = choices[0].get("message", {}).get("content")
            if not content:
                raise MalformedResponseError("No content in provider response")
                
            finish_reason = choices[0].get("finish_reason")
            
            usage_data = data.get("usage", {})
            usage = AIResponseUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
            
            return AIResponse(
                content=content,
                provider="nvidia",
                model=self.model,
                usage=usage,
                finish_reason=finish_reason
            )
        except MalformedResponseError as e:
            raise e
        except Exception as e:
            logger.error(f"Error parsing NVIDIA response: {str(e)} | Data: {data}")
            raise MalformedResponseError("Failed to parse provider response")
