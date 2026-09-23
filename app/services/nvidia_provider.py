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
        
        # Hard requirement: must be GPT-OSS-20B if specified or fallback to configured model, but NOT a local/gemini mock.
        # Ensure it doesn't silently fallback to anything else.
        actual_model = request.model_id if request.model_id else self.model
        
        payload = {
            "model": actual_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        
        max_retries = 3
        base_delay = 1.0
        
        # Explicit connection vs read timeouts
        # Connect timeout: 5 seconds. Read timeout (generation): 45 seconds.
        timeout_config = httpx.Timeout(45.0, connect=5.0)
        
        # Payload size diagnostic
        payload_chars = sum(len(m.get("content", "")) for m in messages)
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info("=== NVIDIA ATS DIAGNOSTIC ===")
                    logger.info("provider=nvidia")
                    host = self.base_url.split("://")[-1].split("/")[0] if "://" in self.base_url else self.base_url
                    logger.info(f"endpoint_host={host}")
                    logger.info(f"model={actual_model}")
                    logger.info(f"api_key_configured={bool(self.api_key)}")
                    logger.info(f"connect_timeout=5.0")
                    logger.info(f"read_timeout=45.0")
                    logger.info(f"max_tokens={request.max_tokens}")
                    logger.info(f"payload_chars={payload_chars}")
                    logger.info(f"attempt={attempt}")

                    clean_base_url = self.base_url.rstrip('/')
                    response = await client.post(
                        f"{clean_base_url}/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    
                    logger.info(f"result=SUCCESS")
                    logger.info(f"status_code={response.status_code}")
                    
                    self._handle_http_errors(response)
                    
                    data = response.json()
                    return self._parse_response(data)
                    
                except httpx.TimeoutException:
                    logger.info(f"result=TIMEOUT")
                    if attempt == max_retries:
                        logger.error(f"NVIDIA API timeout after {max_retries} attempts")
                        raise ProviderTimeoutError("AI request timed out")
                except (ProviderUnavailableError, RateLimitError) as e:
                    logger.info(f"result=HTTP_ERROR")
                    if attempt == max_retries:
                        raise e
                except Exception as e:
                    if isinstance(e, AIException):
                        raise e # Don't retry auth errors, invalid requests, etc
                    logger.info(f"result=HTTP_ERROR")
                    logger.error(f"Unexpected NVIDIA error: {str(e)}")
                    if attempt == max_retries:
                        raise ProviderUnavailableError("Unexpected error communicating with AI provider")
                        
                # Bounded exponential backoff before retry (e.g. 1s, 2s, 4s)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

        raise ProviderUnavailableError("Failed to communicate with AI provider")

    def _handle_http_errors(self, response: httpx.Response):
        status = response.status_code
        if status == 200:
            return
            
        if status == 401 or status == 403:
            logger.error("NVIDIA API authentication failed")
            raise ProviderAuthenticationError("AI provider authentication failed")
        elif status == 404:
            logger.error("NVIDIA API endpoint or model not found (404)")
            raise InvalidRequestError("AI provider endpoint or model not found")
        elif status == 429:
            raise RateLimitError("AI provider rate limit exceeded")
        elif status == 400 or status == 422:
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
                model=data.get("model", self.model),
                usage=usage,
                finish_reason=finish_reason
            )
        except MalformedResponseError as e:
            raise e
        except Exception as e:
            logger.error(f"Error parsing NVIDIA response: {str(e)} | Data: {data}")
            raise MalformedResponseError("Failed to parse provider response")
