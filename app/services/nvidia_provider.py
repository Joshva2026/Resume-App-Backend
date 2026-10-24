import httpx
import logging
import asyncio
import json
from typing import Optional
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
    _shared_loop = None

    @classmethod
    def get_effective_timeouts(cls) -> tuple[float, float, Optional[float], Optional[float]]:
        connect_timeout = float(getattr(settings, "AI_CONNECT_TIMEOUT_SECONDS", 10.0))
        configured_read = float(getattr(settings, "AI_REQUEST_TIMEOUT_SECONDS", 70.0))
        configured_stream_read = getattr(settings, "AI_STREAM_READ_TIMEOUT_SECONDS", None)
        configured_inter_chunk = getattr(settings, "AI_INTER_CHUNK_TIMEOUT_SECONDS", None)
        
        # Keep read timeout bounded for non-streaming
        read_timeout = max(5.0, min(configured_read, 80.0))
        
        # Stream read timeout: default 100.0s, bounded max 105.0s (strictly below Android's 120s limit)
        stream_read_timeout = 100.0
        if configured_stream_read is not None:
            stream_read_timeout = max(10.0, min(float(configured_stream_read), 105.0))
            
        inter_chunk_timeout = None
        if configured_inter_chunk is not None:
            inter_chunk_timeout = max(5.0, min(float(configured_inter_chunk), 45.0))
            
        return connect_timeout, read_timeout, stream_read_timeout, inter_chunk_timeout

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        connect_timeout, read_timeout, _, _ = cls.get_effective_timeouts()
        if (
            cls._shared_client is None 
            or cls._shared_client.is_closed 
            or (cls._shared_loop is not None and cls._shared_loop != current_loop)
        ):
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
            cls._shared_loop = None

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
        connect_timeout, default_read_timeout, _, _ = self.get_effective_timeouts()
        read_timeout = float(request.timeout) if request.timeout is not None else default_read_timeout
        read_timeout = max(5.0, min(read_timeout, 80.0))
        timeout_config = httpx.Timeout(
            timeout=read_timeout,
            connect=connect_timeout,
            read=read_timeout,
            write=10.0,
            pool=5.0
        )
        
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
                        headers=headers,
                        timeout=timeout_config
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

    async def generate_stream(self, request: AIRequest):
        api_key = self.api_key.strip() if self.api_key else None
        if not api_key:
            raise ProviderUnavailableError("AI Provider is not configured")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
            
        for msg in request.history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
        messages.append({"role": "user", "content": request.user_prompt})
        actual_model = (request.model_id or self.model or getattr(settings, "NVIDIA_MODEL", "openai/gpt-oss-20b")).strip()
        
        payload = {
            "model": actual_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True
        }
        
        connect_timeout, default_read_timeout, configured_stream_read, configured_inter_chunk = self.get_effective_timeouts()
        
        # Read timeout is disabled (None) for streaming by default so valid long generations are not terminated.
        # Can still be overridden by request.timeout if explicitly specified by caller.
        stream_read_timeout = float(request.timeout) if request.timeout is not None else configured_stream_read
        
        timeout_config = httpx.Timeout(
            timeout=stream_read_timeout,
            connect=connect_timeout,
            read=stream_read_timeout,
            write=10.0,
            pool=5.0
        )
        
        payload_chars = sum(len(m.get("content", "")) for m in messages)
        host = self.base_url.split("://")[-1].split("/")[0] if "://" in self.base_url else self.base_url
        logger.info(
            f"AI_STREAM_START model={actual_model} host={host} connect_to={connect_timeout}s "
            f"stream_read_to={stream_read_timeout} max_tokens={request.max_tokens} payload_chars={payload_chars}"
        )
        
        client = self.get_client()
        clean_base_url = self.base_url.rstrip('/')
        
        import time
        t_start = time.time()
        first_token_time = None
        last_chunk_time = t_start
        max_inter_chunk_gap = 0.0
        content_chunks_count = 0
        content_chars_count = 0
        reasoning_chunks_count = 0
        accumulated_reasoning = []
        finish_reason = None
        
        try:
            async with client.stream(
                "POST",
                f"{clean_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout_config
            ) as response:
                ttfb = time.time() - t_start
                logger.info(f"AI_STREAM_RESPONSE status={response.status_code} ttfb={int(ttfb*1000)}ms model={actual_model}")
                if response.status_code != 200:
                    await response.aread()
                    self._handle_http_errors(response, actual_model, self.base_url)
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        # Keep-alive comment from upstream
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if "error" in chunk and chunk["error"]:
                                self._handle_stream_error(chunk["error"], actual_model)
                                
                            choices = chunk.get("choices", [])
                            if choices:
                                choice = choices[0]
                                if choice.get("finish_reason"):
                                    finish_reason = choice.get("finish_reason")
                                delta = choice.get("delta", {})
                                content = delta.get("content") or delta.get("text") or choice.get("text") or ""
                                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or choice.get("reasoning_content") or ""
                                
                                now = time.time()
                                gap = now - last_chunk_time
                                if gap > max_inter_chunk_gap:
                                    max_inter_chunk_gap = gap
                                last_chunk_time = now
                                
                                if content:
                                    if first_token_time is None:
                                        first_token_time = now - t_start
                                        logger.info(f"AI_STREAM_FIRST_TOKEN ttft={int(first_token_time*1000)}ms model={actual_model}")
                                    content_chunks_count += 1
                                    content_chars_count += len(content)
                                    yield content
                                elif reasoning:
                                    reasoning_chunks_count += 1
                                    accumulated_reasoning.append(reasoning)
                        except (AIException, json.JSONDecodeError) as e:
                            if isinstance(e, AIException):
                                raise e
                            continue
                        except Exception:
                            continue

                # Requirement 4: Distinguish reasoning metadata from user-facing answer content.
                # If no user content chunks were emitted, NEVER yield internal reasoning as answer fallback.
                if content_chunks_count == 0:
                    if accumulated_reasoning:
                        logger.warning(
                            f"Model {actual_model} emitted only reasoning metadata ({reasoning_chunks_count} chunks) "
                            "without any user-facing answer content."
                        )
                        raise MalformedResponseError("Model emitted reasoning metadata without final answer content")
                    else:
                        logger.warning(f"Model {actual_model} completed stream with 0 content chunks.")
                        raise MalformedResponseError("AI provider completed without emitting answer content")

                total_duration_ms = int((time.time() - t_start) * 1000)
                ttft_ms = int(first_token_time * 1000) if first_token_time else None
                logger.info(
                    f"AI_STREAM_METRICS model={actual_model} duration_ms={total_duration_ms} "
                    f"ttfb_ms={int(ttfb*1000)} ttft_ms={ttft_ms} chunks={content_chunks_count} "
                    f"chars={content_chars_count} reasoning_chunks={reasoning_chunks_count} "
                    f"max_inter_chunk_ms={int(max_inter_chunk_gap*1000)} finish_reason={finish_reason}"
                )

        except httpx.ReadTimeout:
            duration_ms = int((time.time() - t_start) * 1000)
            logger.warning(
                f"AI Provider streaming read timeout on {actual_model} after {duration_ms}ms "
                f"(category=read_timeout, content_chunks={content_chunks_count}, chars={content_chars_count})"
            )
            raise ProviderTimeoutError("The AI response took too long. Please try again.")
        except httpx.ConnectTimeout:
            duration_ms = int((time.time() - t_start) * 1000)
            logger.warning(f"AI Provider streaming connect timeout on {actual_model} after {duration_ms}ms (category=connect_timeout)")
            raise ProviderTimeoutError("Failed to connect to AI provider in time")
        except httpx.ConnectError as e:
            duration_ms = int((time.time() - t_start) * 1000)
            logger.warning(f"AI Provider streaming connection error after {duration_ms}ms: {e} (category=connect_error)")
            raise ProviderUnavailableError("Unable to establish connection to AI provider")
        except (httpx.RemoteProtocolError, httpx.ReadError) as e:
            duration_ms = int((time.time() - t_start) * 1000)
            logger.warning(f"AI Provider streaming transport disconnect after {duration_ms}ms: {e} (category=disconnect)")
            raise ProviderUnavailableError("AI provider connection was interrupted")
        except httpx.TimeoutException:
            duration_ms = int((time.time() - t_start) * 1000)
            logger.warning(f"AI Provider streaming generic timeout after {duration_ms}ms (category=generic_timeout)")
            raise ProviderTimeoutError("The AI response took too long. Please try again.")
        except asyncio.CancelledError:
            duration_ms = int((time.time() - t_start) * 1000)
            logger.info(f"AI Provider stream cancelled by caller after {duration_ms}ms (chunks={content_chunks_count})")
            raise
        except Exception as e:
            if isinstance(e, AIException):
                raise e
            logger.error(f"Unexpected streaming error on {actual_model}: {str(e)}")
            raise ProviderUnavailableError("Unexpected error communicating with AI provider")

    def _handle_stream_error(self, err_obj: dict, actual_model: str):
        err_msg = err_obj.get("message") or "Unknown AI provider error"
        err_code = err_obj.get("code") or err_obj.get("type")
        logger.error(f"NVIDIA API streaming error event on {actual_model}: code={err_code} msg={err_msg}")
        if err_code == 429 or "rate limit" in str(err_msg).lower():
            raise RateLimitError(err_msg)
        elif err_code in [401, 403] or "auth" in str(err_msg).lower():
            raise ProviderAuthenticationError(err_msg)
        elif err_code in [400, 422]:
            raise InvalidRequestError(err_msg)
        else:
            raise ProviderUnavailableError(err_msg)

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
            finish_reason = choices[0].get("finish_reason")

            if not content:
                # If finish reason indicates truncation due to tokens, fail fast
                if finish_reason in ["length", "max_tokens"]:
                    logger.warning(f"NVIDIA response truncated due to length/max_tokens (finish_reason={finish_reason})")
                    raise MalformedResponseError("NVIDIA response truncated due to length/max_tokens")

                # If content is None/empty, check if reasoning/reasoning_content was provided
                msg = choices[0].get("message", {})
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")
                if reasoning:
                    logger.warning("Provider returned reasoning field without final content.")
                    raise MalformedResponseError("Provider returned reasoning metadata without completed final content")
                else:
                    raise MalformedResponseError("No content in provider response")
            
            # Remove <think>...</think> reasoning blocks if present
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
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
