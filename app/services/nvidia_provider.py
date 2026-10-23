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
    _shared_client = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        if cls._shared_client is None or cls._shared_client.is_closed:
            connect_timeout = float(getattr(settings, "AI_CONNECT_TIMEOUT_SECONDS", 5.0))
            read_timeout = float(getattr(settings, "AI_REQUEST_TIMEOUT_SECONDS", 45.0))
            timeout_config = httpx.Timeout(
                timeout=read_timeout,
                connect=connect_timeout,
                read=read_timeout,
                write=10.0,
                pool=5.0
            )
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
            cls._shared_client = httpx.AsyncClient(timeout=timeout_config, limits=limits)
        return cls._shared_client

    @classmethod
    async def close_client(cls):
        if cls._shared_client is not None and not cls._shared_client.is_closed:
            await cls._shared_client.aclose()
            cls._shared_client = None

    def __init__(self):
        self.api_key = settings.NVIDIA_API_KEY
        self.base_url = settings.NVIDIA_BASE_URL
        self.model = settings.NVIDIA_MODEL
        self.timeout = settings.AI_REQUEST_TIMEOUT_SECONDS
        
        if not self.api_key:
            logger.warning("NVIDIA_API_KEY is not configured.")
            
    async def generate(self, request: AIRequest) -> AIResponse:
        api_key = self.api_key.strip() if self.api_key else None
        if not api_key:
            raise ProviderUnavailableError("AI Provider is not configured")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
            
        # Bounded context protection: ensure history isn't excessively large
        for msg in request.history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
        messages.append({"role": "user", "content": request.user_prompt})
        
        # Model selection: configurable via settings / env or request override
        actual_model = (request.model_id or self.model or getattr(settings, "NVIDIA_MODEL", "openai/gpt-oss-20b")).strip()
        
        payload = {
            "model": actual_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        max_retries = 2
        base_delay = 1.0
        connect_timeout = float(getattr(settings, "AI_CONNECT_TIMEOUT_SECONDS", 5.0))
        read_timeout = float(getattr(settings, "AI_REQUEST_TIMEOUT_SECONDS", 40.0))
        
        # Payload size diagnostic
        payload_chars = sum(len(m.get("content", "")) for m in messages)
        
        client = self.get_client()
        for attempt in range(1, max_retries + 1):
                try:
                    logger.info("=== NVIDIA DIAGNOSTIC ===")
                    logger.info("provider=nvidia")
                    host = self.base_url.split("://")[-1].split("/")[0] if "://" in self.base_url else self.base_url
                    logger.info(f"endpoint_host={host}")
                    logger.info(f"model={actual_model}")
                    logger.info(f"api_key_configured={bool(api_key)}")
                    logger.info(f"api_key_length={len(api_key) if api_key else 0}")
                    logger.info(f"connect_timeout={connect_timeout}")
                    logger.info(f"read_timeout={read_timeout}")
                    logger.info(f"max_tokens={request.max_tokens}")
                    logger.info(f"payload_chars={payload_chars}")
                    logger.info(f"attempt={attempt}")

                    logger.info("AI request started")
                    logger.info("AI provider request started")
                    logger.info("AI_PROVIDER_START")
                    
                    import time
                    start_time = time.time()
                    
                    clean_base_url = self.base_url.rstrip('/')
                    response = await client.post(
                        f"{clean_base_url}/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    
                    duration_ms = int((time.time() - start_time) * 1000)
                    logger.info("AI provider response received")
                    logger.info(f"AI provider duration: {duration_ms}ms")
                    logger.info(f"AI response status: {response.status_code}")
                    logger.info(f"AI response content length: {len(response.text)}")
                    logger.info("AI request completed")
                    
                    logger.info(f"AI_PROVIDER_RESPONSE status={response.status_code} duration={duration_ms}ms")
                    logger.info(f"AI_PROVIDER_CONTENT length={len(response.text)}")
                    logger.info("AI_PROVIDER_END")
                    
                    self._handle_http_errors(response, actual_model, self.base_url)
                    
                    data = response.json()
                    
                    # Log token usage or finish reason if it's truncated
                    choices = data.get("choices", [])
                    if choices:
                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason == "length":
                            logger.warning("NVIDIA response truncated due to length/max_tokens.")
                            
                    logger.info(f"result=SUCCESS")
                    logger.info(f"status_code={response.status_code}")
                    return self._parse_response(data)
                    
                except httpx.ReadTimeout as e:
                    duration_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                    logger.warning(f"AI Provider read timeout after {duration_ms}ms on {actual_model} (attempt {attempt}). Not retrying to prevent gateway timeout.")
                    logger.info(f"AI_PROVIDER_TIMEOUT duration={duration_ms}ms")
                    logger.info("result=TIMEOUT")
                    raise ProviderTimeoutError("The AI response took too long. Please try again.")
                except httpx.ConnectTimeout as e:
                    duration_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                    logger.warning(f"AI Provider connect timeout after {duration_ms}ms (attempt {attempt}/{max_retries})")
                    logger.info(f"AI_PROVIDER_CONNECT_TIMEOUT duration={duration_ms}ms")
                    if attempt == max_retries:
                        raise ProviderTimeoutError("Failed to connect to AI provider in time")
                except httpx.ConnectError as e:
                    duration_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                    logger.warning(f"AI Provider connection error (attempt {attempt}/{max_retries}): {e}")
                    logger.info("result=CONNECT_ERROR")
                    if attempt == max_retries:
                        raise ProviderUnavailableError("Unable to establish connection to AI provider")
                except httpx.TimeoutException as e:
                    duration_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                    logger.warning(f"AI Provider generic timeout after {duration_ms}ms (attempt {attempt}/{max_retries})")
                    logger.info("result=TIMEOUT")
                    if attempt == max_retries:
                        raise ProviderTimeoutError("The AI response took too long. Please try again.")
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
                        
                # Bounded exponential backoff with jitter before retry
                import random
                jitter = random.uniform(0.1, 0.4)
                await asyncio.sleep((base_delay * (2 ** (attempt - 1))) + jitter)

        raise ProviderUnavailableError("Failed to communicate with AI provider")

    def _handle_http_errors(self, response: httpx.Response, actual_model: str, base_url: str):
        status = response.status_code
        if status == 200:
            return
            
        host = base_url.split("://")[-1].split("/")[0] if "://" in base_url else base_url
        logger.error(f"HTTP_FAILURE status={status} model={actual_model} host={host}")
        
        # Sanitize response body to avoid leaking keys/tokens in case provider echoes them
        sanitized_text = response.text.replace(self.api_key or "NO_KEY", "[REDACTED]")
        
        if status == 401 or status == 403:
            logger.error(f"NVIDIA API authentication failed: {sanitized_text}")
            raise ProviderAuthenticationError("AI provider authentication failed")
        elif status == 404:
            logger.error("NVIDIA API endpoint or model not found (404)")
            raise InvalidRequestError("AI provider endpoint or model not found")
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            logger.warning(f"NVIDIA API rate limit (429), Retry-After: {retry_after}")
            raise RateLimitError("AI provider rate limit exceeded")
        elif status == 400 or status == 422:
            logger.error(f"NVIDIA API invalid request: {sanitized_text}")
            raise InvalidRequestError("Invalid request to AI provider")
        elif status >= 500:
            logger.error(f"NVIDIA API provider error {status}: {sanitized_text}")
            raise ProviderUnavailableError("AI provider is temporarily unavailable")
        else:
            logger.error(f"NVIDIA API unexpected status {status}: {sanitized_text}")
            raise AIException(f"AI provider returned status {status}")

    def _parse_response(self, data: dict) -> AIResponse:
        try:
            choices = data.get("choices", [])
            if not choices:
                raise MalformedResponseError("No choices in provider response")
                
            content = choices[0].get("message", {}).get("content")
            if not content:
                raise MalformedResponseError("No content in provider response")
            
            # Remove <think>...</think> reasoning blocks if present
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
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
