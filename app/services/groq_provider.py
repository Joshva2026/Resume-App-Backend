import httpx
import logging
import asyncio
import json
import time
import random
import re
from typing import Optional, Tuple
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

class GroqAIProvider(AIProvider):
    """
    Dedicated Groq AI Provider for ATS Analysis pipeline.
    Uses Groq's OpenAI-compatible chat completions API with openai/gpt-oss-120b.
    Maintains full operational isolation from NVIDIA AI Assistant provider.
    """
    _shared_client: Optional[httpx.AsyncClient] = None
    _shared_loop = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        connect_timeout = float(getattr(settings, "AI_CONNECT_TIMEOUT_SECONDS", 10.0))
        read_timeout = float(getattr(settings, "ATS_AI_TIMEOUT_SECONDS", 35.0))
        
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if cls._shared_client is None or cls._shared_client.is_closed or cls._shared_loop != current_loop:
            timeout_config = httpx.Timeout(
                timeout=read_timeout,
                connect=connect_timeout,
                read=read_timeout,
                write=10.0,
                pool=5.0
            )
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
            cls._shared_client = httpx.AsyncClient(timeout=timeout_config, limits=limits)
            cls._shared_loop = current_loop
        return cls._shared_client

    @classmethod
    async def close_client(cls):
        if cls._shared_client is not None and not cls._shared_client.is_closed:
            await cls._shared_client.aclose()
            cls._shared_client = None

    def __init__(self):
        raw_key = settings.GROQ_API_KEY
        if not raw_key:
            xai_k = getattr(settings, "XAI_API_KEY", None)
            if xai_k and xai_k.startswith("gsk_"):
                raw_key = xai_k
        self.api_key = raw_key
        self.base_url = (settings.GROQ_BASE_URL or "https://api.groq.com/openai/v1").rstrip("/")
        self.model = settings.ATS_GROQ_MODEL or "openai/gpt-oss-120b"
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not configured.")

    async def generate(self, request: AIRequest) -> AIResponse:
        api_key = self.api_key.strip() if self.api_key else None
        if not api_key:
            raise ProviderUnavailableError("Groq AI Provider is not configured (missing GROQ_API_KEY)")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        messages.append({"role": "user", "content": request.user_prompt})

        actual_model = (request.model_id or self.model or getattr(settings, "ATS_GROQ_MODEL", "openai/gpt-oss-120b")).strip()

        payload = {
            "model": actual_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": min(request.max_tokens, 2048)
        }

        # Bounded timeouts
        configured_timeout = request.timeout or float(getattr(settings, "ATS_AI_TIMEOUT_SECONDS", 35.0))
        connect_timeout = float(getattr(settings, "AI_CONNECT_TIMEOUT_SECONDS", 10.0))
        read_timeout = max(5.0, min(configured_timeout, 60.0))

        timeout_config = httpx.Timeout(
            timeout=read_timeout,
            connect=connect_timeout,
            read=read_timeout,
            write=10.0,
            pool=5.0
        )

        max_retries = 2
        base_delay = 1.0

        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                client = self.get_client()
                logger.info(f"GROQ_ATS_REQUEST model={actual_model} attempt={attempt}/{max_retries}")
                
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout_config
                )

                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(f"GROQ_ATS_RESPONSE status={response.status_code} duration={duration_ms}ms")

                self._handle_http_errors(response, actual_model)
                data = response.json()
                return self._parse_response(data, actual_model)

            except httpx.ReadTimeout:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.warning(f"Groq Provider read timeout after {duration_ms}ms on {actual_model} (attempt {attempt}). Not retrying to prevent gateway latency.")
                raise ProviderTimeoutError("The AI response took too long. Please try again.")

            except httpx.ConnectTimeout:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.warning(f"Groq Provider connect timeout after {duration_ms}ms (attempt {attempt}/{max_retries})")
                if attempt == max_retries:
                    raise ProviderTimeoutError("Failed to connect to Groq AI provider in time")

            except httpx.ConnectError as e:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.warning(f"Groq Provider connection error (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    raise ProviderUnavailableError("Unable to establish connection to Groq AI provider")

            except httpx.TimeoutException:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.warning(f"Groq Provider generic timeout after {duration_ms}ms (attempt {attempt}/{max_retries})")
                if attempt == max_retries:
                    raise ProviderTimeoutError("The AI response took too long. Please try again.")

            except (ProviderUnavailableError, RateLimitError) as e:
                if attempt == max_retries:
                    raise e

            except Exception as e:
                if isinstance(e, AIException):
                    raise e
                logger.error(f"Unexpected Groq Provider error: {str(e)}")
                if attempt == max_retries:
                    raise ProviderUnavailableError("Unexpected error communicating with Groq AI provider")

            # Bounded backoff before retry on retryable transient errors
            jitter = random.uniform(0.1, 0.3)
            await asyncio.sleep((base_delay * attempt) + jitter)

        raise ProviderUnavailableError("Failed to communicate with Groq AI provider")

    async def generate_stream(self, request: AIRequest):
        """
        ATS pipeline uses non-streaming analysis; delegate to generate for fallback.
        """
        resp = await self.generate(request)
        yield resp.content

    def _handle_http_errors(self, response: httpx.Response, model: str):
        status = response.status_code
        if status == 200:
            return

        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text[:200]

        # Sanitize error message to avoid logging any secrets
        sanitized_msg = re.sub(r'gsk_[A-Za-z0-9_-]+', '[GROQ_KEY_REDACTED]', str(error_msg))

        if status in (401, 403):
            logger.error(f"Groq API authentication failure: {sanitized_msg}")
            raise ProviderAuthenticationError("Groq authentication failed. Please verify GROQ_API_KEY.")

        if status == 429:
            logger.warning(f"Groq API rate limit exceeded: {sanitized_msg}")
            raise RateLimitError("Groq API rate limit exceeded. Please try again shortly.")

        if status in (400, 422):
            logger.error(f"Groq API invalid request ({status}): {sanitized_msg}")
            raise InvalidRequestError(f"Groq API rejected request: {sanitized_msg}")

        if status == 504:
            logger.error(f"Groq API gateway timeout: {sanitized_msg}")
            raise ProviderTimeoutError("Groq API gateway timed out.")

        if status in (500, 502, 503):
            logger.error(f"Groq API server error ({status}): {sanitized_msg}")
            raise ProviderUnavailableError(f"Groq API service is currently unavailable ({status}).")

        logger.error(f"Groq API unexpected status ({status}): {sanitized_msg}")
        raise AIException(f"Groq API returned status {status}")

    def _parse_response(self, data: dict, model: str) -> AIResponse:
        try:
            choices = data.get("choices", [])
            if not choices:
                raise MalformedResponseError("No choices in Groq response")

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content")
            finish_reason = choice.get("finish_reason")

            if not content:
                if finish_reason in ["length", "max_tokens"]:
                    logger.warning(f"Groq response truncated due to length/max_tokens (finish_reason={finish_reason})")
                    raise MalformedResponseError("Groq response truncated due to length/max_tokens")

                reasoning = message.get("reasoning") or message.get("reasoning_content")
                if reasoning:
                    logger.warning("Groq returned reasoning without final content.")
                    raise MalformedResponseError("Groq returned reasoning metadata without completed final content")
                else:
                    raise MalformedResponseError("No content in Groq response")

            # Remove <think>...</think> reasoning blocks if present
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            usage_data = data.get("usage", {})
            usage = AIResponseUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )

            return AIResponse(
                content=clean_content,
                provider="groq",
                model=data.get("model", model),
                usage=usage,
                finish_reason=finish_reason
            )
        except MalformedResponseError:
            raise
        except Exception as e:
            logger.error(f"Error parsing Groq response: {str(e)}")
            raise MalformedResponseError("Failed to parse Groq provider response")
