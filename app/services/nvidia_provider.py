import httpx
import logging
import asyncio
import json
import math
import random
import re
import time
from email.utils import parsedate_to_datetime
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

_QUOTA_EXHAUSTED_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "quota exceeded",
    "quota has been exceeded",
    "billing hard limit",
    "credit balance",
    "out of credits",
)


class NvidiaAIProvider(AIProvider):
    _shared_client = None
    _shared_loop = None

    @staticmethod
    def _parse_numeric_setting(value, default: Optional[float]) -> Optional[float]:
        if isinstance(value, bool) or value is None:
            return default
        if isinstance(value, (int, float)):
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            try:
                parsed = float(stripped)
                return parsed if math.isfinite(parsed) else default
            except ValueError:
                return default
        return default

    @classmethod
    def _safe_float_setting(cls, name: str, default: float) -> float:
        raw = getattr(settings, name, default)
        parsed = cls._parse_numeric_setting(raw, default)
        return default if parsed is None else parsed

    @classmethod
    def _safe_int_setting(cls, name: str, default: int) -> int:
        raw = getattr(settings, name, default)
        parsed = cls._parse_numeric_setting(raw, float(default))
        return int(parsed) if parsed is not None else default

    @staticmethod
    def _parse_retry_after(headers) -> Optional[float]:
        if headers is None or not hasattr(headers, "get"):
            return None
        try:
            raw = headers.get("Retry-After")
        except Exception:
            return None
        if isinstance(raw, bool) or raw is None:
            return None
        if isinstance(raw, (int, float)):
            val = float(raw)
            return val if (math.isfinite(val) and val >= 0) else None
        if not isinstance(raw, str):
            return None
        s = raw.strip()
        if not s:
            return None
        try:
            val = float(s)
            return val if (math.isfinite(val) and val >= 0) else None
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(s)
            if dt is not None:
                return max(0.0, dt.timestamp() - time.time())
        except Exception:
            return None
        return None

    @staticmethod
    def _is_quota_exhausted_text(text: Optional[str]) -> bool:
        if not isinstance(text, str) or not text:
            return False
        lowered = text.lower()
        return any(marker in lowered for marker in _QUOTA_EXHAUSTED_MARKERS)

    @classmethod
    def get_effective_timeouts(cls) -> tuple[float, float, Optional[float], Optional[float]]:
        connect_timeout = max(1.0, min(cls._safe_float_setting("AI_CONNECT_TIMEOUT_SECONDS", 10.0), 30.0))
        configured_read = cls._safe_float_setting("AI_REQUEST_TIMEOUT_SECONDS", 70.0)
        raw_stream_read = getattr(settings, "AI_STREAM_READ_TIMEOUT_SECONDS", None)
        raw_inter_chunk = getattr(settings, "AI_INTER_CHUNK_TIMEOUT_SECONDS", None)

        # Keep read timeout bounded for non-streaming
        read_timeout = max(5.0, min(configured_read, 80.0))

        # Stream read timeout: default 100.0s, bounded max configurable (default 105.0s, strictly below Android's 120s limit)
        stream_read_max = max(30.0, min(cls._safe_float_setting("AI_STREAM_READ_TIMEOUT_MAX_SECONDS", 105.0), 115.0))
        parsed_stream_read = cls._parse_numeric_setting(raw_stream_read, None)
        stream_read_timeout = 100.0
        if parsed_stream_read is not None:
            stream_read_timeout = max(10.0, min(parsed_stream_read, stream_read_max))

        parsed_inter_chunk = cls._parse_numeric_setting(raw_inter_chunk, None)
        inter_chunk_timeout = None
        if parsed_inter_chunk is not None:
            inter_chunk_timeout = max(5.0, min(parsed_inter_chunk, 45.0))

        return connect_timeout, read_timeout, stream_read_timeout, inter_chunk_timeout

    @classmethod
    def get_first_token_timeout(cls) -> float:
        max_timeout = max(5.0, min(cls._safe_float_setting("AI_FIRST_TOKEN_TIMEOUT_MAX_SECONDS", 60.0), 120.0))
        val = cls._safe_float_setting("AI_FIRST_TOKEN_TIMEOUT_SECONDS", 30.0)
        return max(3.0, min(val, max_timeout))

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
            old_client = cls._shared_client
            old_loop = cls._shared_loop
            if old_client is not None and not old_client.is_closed:
                try:
                    if old_loop is not None and not old_loop.is_closed() and old_loop.is_running():
                        old_loop.call_soon_threadsafe(lambda c=old_client: asyncio.ensure_future(c.aclose()))
                except Exception:
                    pass
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
        max_retry_wait = max(0.0, min(self._safe_float_setting("AI_PROVIDER_RETRY_MAX_WAIT_SECONDS", 8.0), 30.0))
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
                pending_retry_delay: Optional[float] = None
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

                    logger.info("result=SUCCESS")
                    logger.info(f"status_code={response.status_code}")
                    return self._parse_response(data, model=actual_model)

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
                except RateLimitError as e:
                    logger.info("result=HTTP_ERROR")
                    if e.quota_exhausted or not e.retryable:
                        logger.warning(f"Aborting non-streaming retry on {actual_model}: non-retryable rate limit / quota exhausted")
                        raise e
                    if e.retry_after is not None and e.retry_after > max_retry_wait:
                        logger.warning(
                            f"Aborting non-streaming retry on {actual_model}: "
                            f"Retry-After={e.retry_after:.1f}s exceeds max_wait={max_retry_wait:.1f}s"
                        )
                        raise e
                    if attempt == max_retries:
                        raise e
                    if e.retry_after is not None:
                        pending_retry_delay = min(max_retry_wait, max(0.1, e.retry_after) + random.uniform(0.05, 0.25))
                except ProviderUnavailableError as e:
                    logger.info("result=HTTP_ERROR")
                    if attempt == max_retries:
                        raise e
                except Exception as e:
                    if isinstance(e, AIException):
                        raise e # Don't retry auth errors, invalid requests, etc
                    logger.info("result=HTTP_ERROR")
                    logger.error(f"Unexpected NVIDIA error: {str(e)}")
                    if attempt == max_retries:
                        raise ProviderUnavailableError("Unexpected error communicating with AI provider")

                # Bounded exponential backoff with jitter before retry
                jitter = random.uniform(0.1, 0.4)
                backoff_delay = pending_retry_delay if pending_retry_delay is not None else min(
                    max_retry_wait,
                    (base_delay * (2 ** (attempt - 1))) + jitter
                )
                await asyncio.sleep(backoff_delay)

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
        
        primary_model = (request.model_id or self.model or getattr(settings, "NVIDIA_MODEL", "openai/gpt-oss-20b")).strip()
        models_to_try = [primary_model]
        raw_fallback = getattr(settings, "NVIDIA_FALLBACK_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
        if isinstance(raw_fallback, str) and raw_fallback.strip():
            fallback_model = raw_fallback.strip()
            if fallback_model != primary_model:
                models_to_try.append(fallback_model)

        connect_timeout, default_read_timeout, configured_stream_read, configured_inter_chunk = self.get_effective_timeouts()
        first_token_timeout = self.get_first_token_timeout()
        max_429_retries = max(0, min(self._safe_int_setting("AI_STREAM_429_MAX_RETRIES", 2), 4))
        max_retry_wait = max(0.0, min(self._safe_float_setting("AI_PROVIDER_RETRY_MAX_WAIT_SECONDS", 8.0), 30.0))

        # Stream read timeout defaults to configured bounded value (100s) so long generations succeed
        # without dangling past Android's 120s limit. Can still be overridden by request.timeout.
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
        client = self.get_client()
        clean_base_url = self.base_url.rstrip('/')

        async def _stream_model(model_name: str, is_fallback: bool = False, attempt: int = 1):
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True
            }
            logger.info(
                f"AI_STREAM_START model={model_name} is_fallback={is_fallback} attempt={attempt} host={host} "
                f"connect_to={connect_timeout}s stream_read_to={stream_read_timeout}s "
                f"first_token_to={first_token_timeout}s max_tokens={request.max_tokens} payload_chars={payload_chars}"
            )

            t_start = time.time()
            t_first_token_deadline = t_start + first_token_timeout
            first_token_time = None
            last_chunk_time = t_start
            max_inter_chunk_gap = 0.0
            content_chunks_count = 0
            content_chars_count = 0
            reasoning_chunks_count = 0
            accumulated_reasoning = []
            finish_reason = None

            response_cm = client.stream(
                "POST",
                f"{clean_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout_config
            )

            response = None
            exc_info = (None, None, None)
            try:
                # Wait for HTTP response headers bounded by this attempt's first_token_timeout
                time_for_headers = max(0.1, t_first_token_deadline - time.time())
                try:
                    response = await asyncio.wait_for(response_cm.__aenter__(), timeout=time_for_headers)
                except (asyncio.TimeoutError, TimeoutError):
                    duration_ms = int((time.time() - t_start) * 1000)
                    logger.warning(
                        f"AI Provider streaming first-token timeout on {model_name} after {duration_ms}ms "
                        f"(waiting for headers, threshold={first_token_timeout}s, attempt={attempt})"
                    )
                    raise httpx.ReadTimeout(f"First-token timeout after {first_token_timeout}s waiting for headers on {model_name}")

                ttfb = time.time() - t_start
                logger.info(
                    f"AI_STREAM_RESPONSE status={response.status_code} ttfb={int(ttfb*1000)}ms "
                    f"model={model_name} attempt={attempt}"
                )
                if response.status_code != 200:
                    await response.aread()
                    self._handle_http_errors(response, model_name, self.base_url)

                line_iter = response.aiter_lines().__aiter__()
                while True:
                    if first_token_time is None:
                        time_for_token = max(0.1, t_first_token_deadline - time.time())
                        try:
                            line = await asyncio.wait_for(line_iter.__anext__(), timeout=time_for_token)
                        except (asyncio.TimeoutError, TimeoutError):
                            duration_ms = int((time.time() - t_start) * 1000)
                            logger.warning(
                                f"AI Provider streaming first-token timeout on {model_name} after {duration_ms}ms "
                                f"(waiting for content, threshold={first_token_timeout}s, chunks={content_chunks_count}, attempt={attempt})"
                            )
                            raise httpx.ReadTimeout(f"First-token timeout after {first_token_timeout}s on {model_name}")
                        except StopAsyncIteration:
                            break
                    else:
                        if configured_inter_chunk is not None:
                            try:
                                line = await asyncio.wait_for(line_iter.__anext__(), timeout=configured_inter_chunk)
                            except (asyncio.TimeoutError, TimeoutError):
                                duration_ms = int((time.time() - t_start) * 1000)
                                logger.warning(
                                    f"AI Provider streaming inter-chunk timeout on {model_name} after {duration_ms}ms "
                                    f"(threshold={configured_inter_chunk}s)"
                                )
                                raise httpx.ReadTimeout(f"Inter-chunk timeout after {configured_inter_chunk}s on {model_name}")
                            except StopAsyncIteration:
                                break
                        else:
                            try:
                                line = await line_iter.__anext__()
                            except StopAsyncIteration:
                                break

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
                                self._handle_stream_error(chunk["error"], model_name)

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
                                        logger.info(f"AI_STREAM_FIRST_TOKEN ttft={int(first_token_time*1000)}ms model={model_name}")
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

                # Requirement: Distinguish reasoning metadata from user-facing answer content.
                if content_chunks_count == 0:
                    if accumulated_reasoning:
                        logger.warning(
                            f"Model {model_name} emitted only reasoning metadata ({reasoning_chunks_count} chunks) "
                            "without any user-facing answer content."
                        )
                        raise MalformedResponseError("Model emitted reasoning metadata without final answer content")
                    else:
                        logger.warning(f"Model {model_name} completed stream with 0 content chunks.")
                        raise MalformedResponseError("AI provider completed without emitting answer content")

                total_duration_ms = int((time.time() - t_start) * 1000)
                ttft_ms = int(first_token_time * 1000) if first_token_time else None
                logger.info(
                    f"AI_STREAM_METRICS model={model_name} duration_ms={total_duration_ms} "
                    f"ttfb_ms={int(ttfb*1000)} ttft_ms={ttft_ms} chunks={content_chunks_count} "
                    f"chars={content_chars_count} reasoning_chunks={reasoning_chunks_count} "
                    f"max_inter_chunk_ms={int(max_inter_chunk_gap*1000)} finish_reason={finish_reason}"
                )
            except BaseException as exc:
                exc_info = (type(exc), exc, exc.__traceback__)
                raise
            finally:
                if response is not None:
                    await response_cm.__aexit__(*exc_info)

        # Iterate over models with bounded transient 429 retry and safe pre-token fallback
        has_yielded = False
        for idx, model_to_use in enumerate(models_to_try):
            is_fallback = (idx > 0)
            is_last_model = (idx == len(models_to_try) - 1)
            attempt = 0

            while True:
                attempt += 1
                stream_gen = _stream_model(model_to_use, is_fallback=is_fallback, attempt=attempt)
                should_fallback = False
                retry_delay: Optional[float] = None

                try:
                    async for chunk in stream_gen:
                        has_yielded = True
                        try:
                            request.model_id = model_to_use
                        except Exception:
                            pass
                        yield chunk
                    # Stream completed successfully
                    return
                except RateLimitError as err:
                    if has_yielded:
                        logger.warning(f"AI Provider streaming rate limit after content yielded on {model_to_use}: {err}")
                        raise err
                    if err.quota_exhausted or not err.retryable:
                        logger.warning(
                            f"AI Provider quota exhausted or non-retryable 429 on '{model_to_use}': {err}. "
                            "Aborting without retry."
                        )
                        raise err
                    if err.retry_after is not None and err.retry_after > max_retry_wait:
                        if not is_last_model:
                            next_model = models_to_try[idx + 1]
                            logger.warning(
                                f"AI Provider 429 Retry-After={err.retry_after:.1f}s exceeds max_wait={max_retry_wait:.1f}s "
                                f"on '{model_to_use}'. Falling back to '{next_model}'."
                            )
                            should_fallback = True
                        else:
                            logger.warning(
                                f"AI Provider 429 Retry-After={err.retry_after:.1f}s exceeds max_wait={max_retry_wait:.1f}s "
                                f"on final model '{model_to_use}'. Aborting."
                            )
                            raise err
                    elif attempt <= max_429_retries:
                        jitter = random.uniform(0.1, 0.35)
                        if err.retry_after is not None:
                            delay = min(max_retry_wait, max(0.1, err.retry_after) + jitter)
                        else:
                            delay = min(max_retry_wait, (0.5 * (2 ** (attempt - 1))) + jitter)
                        logger.warning(
                            f"AI Provider transient 429 on '{model_to_use}' (attempt {attempt}/{max_429_retries + 1}, "
                            f"retry_after={err.retry_after}). Waiting {delay:.2f}s before retry."
                        )
                        retry_delay = delay
                    elif not is_last_model:
                        next_model = models_to_try[idx + 1]
                        logger.warning(
                            f"AI Provider 429 retries exhausted on '{model_to_use}' after {attempt} attempts. "
                            f"Falling back to '{next_model}'."
                        )
                        should_fallback = True
                    else:
                        logger.warning(
                            f"AI Provider 429 retries exhausted on final model '{model_to_use}' after {attempt} attempts."
                        )
                        raise err
                except (httpx.ReadTimeout, httpx.TimeoutException, httpx.ConnectTimeout, httpx.ConnectError,
                        httpx.RemoteProtocolError, httpx.ReadError, ProviderTimeoutError, ProviderUnavailableError, MalformedResponseError) as err:
                    # If chunks were already yielded to consumer, NEVER fallback (prevents duplicate response)
                    if has_yielded:
                        logger.warning(f"AI Provider streaming error after content yielded on {model_to_use}: {err}")
                        if isinstance(err, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException, ProviderTimeoutError)):
                            raise ProviderTimeoutError("The AI response took too long. Please try again.")
                        elif isinstance(err, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, ProviderUnavailableError)):
                            raise ProviderUnavailableError("AI provider connection was interrupted")
                        raise err

                    if not is_last_model:
                        next_model = models_to_try[idx + 1]
                        err_name = type(err).__name__
                        logger.warning(
                            f"AI Provider streaming failed on model '{model_to_use}' before emitting first token "
                            f"(reason={err_name}: {str(err)}). "
                            f"Attempting safe fallback to model '{next_model}'."
                        )
                        should_fallback = True
                    else:
                        logger.warning(f"AI Provider streaming failed on final model '{model_to_use}' (category={type(err).__name__}): {err}")
                        if isinstance(err, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException, ProviderTimeoutError)):
                            raise ProviderTimeoutError("The AI response took too long. Please try again.")
                        elif isinstance(err, (httpx.ConnectError, ProviderUnavailableError)):
                            raise ProviderUnavailableError("Unable to establish connection to AI provider")
                        elif isinstance(err, (httpx.RemoteProtocolError, httpx.ReadError)):
                            raise ProviderUnavailableError("AI provider connection was interrupted")
                        raise err
                except asyncio.CancelledError:
                    logger.info(f"AI Provider stream cancelled by caller on model {model_to_use}")
                    raise
                except Exception as e:
                    if isinstance(e, AIException):
                        raise e
                    logger.error(f"Unexpected streaming error on {model_to_use}: {str(e)}")
                    raise ProviderUnavailableError("Unexpected error communicating with AI provider")
                finally:
                    await stream_gen.aclose()

                if retry_delay is not None:
                    await asyncio.sleep(retry_delay)
                    continue
                if should_fallback:
                    break

    def _handle_stream_error(self, err_obj: dict, actual_model: str):
        err_msg = err_obj.get("message") or "Unknown AI provider error"
        err_code = err_obj.get("code") or err_obj.get("type")
        retry_after = self._parse_numeric_setting(err_obj.get("retry_after"), None)
        if retry_after is not None and retry_after < 0:
            retry_after = None
        logger.error(f"NVIDIA API streaming error event on {actual_model}: code={err_code} msg={err_msg}")
        if err_code == 429 or "rate limit" in str(err_msg).lower() or self._is_quota_exhausted_text(str(err_msg)):
            quota_exhausted = self._is_quota_exhausted_text(str(err_msg)) or str(err_code).lower() == "insufficient_quota"
            raise RateLimitError(
                err_msg,
                retry_after=retry_after,
                retryable=not quota_exhausted,
                quota_exhausted=quota_exhausted,
            )
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
        raw_text_attr = getattr(response, "text", None)
        is_real_text = isinstance(raw_text_attr, str)
        raw_text = raw_text_attr if is_real_text else str(raw_text_attr or "")
        sanitized_text = raw_text[:500]
        if self.api_key:
            sanitized_text = sanitized_text.replace(self.api_key, "[REDACTED]")
        sanitized_text = re.sub(r'nvapi-[A-Za-z0-9_-]+', '[REDACTED]', sanitized_text)
        sanitized_text = re.sub(r'Bearer\s+[A-Za-z0-9._-]+', 'Bearer [REDACTED]', sanitized_text)

        if status == 401 or status == 403:
            logger.error(f"NVIDIA API authentication failed: {sanitized_text}")
            raise ProviderAuthenticationError("AI provider authentication failed")
        elif status == 404:
            logger.error("NVIDIA API endpoint or model not found (404)")
            raise InvalidRequestError("AI provider endpoint or model not found")
        elif status == 429:
            retry_after = self._parse_retry_after(getattr(response, "headers", None))
            quota_exhausted = self._is_quota_exhausted_text(raw_text) if is_real_text else False
            logger.warning(
                f"NVIDIA API rate limit (429) on {actual_model}: "
                f"retry_after={retry_after}, quota_exhausted={quota_exhausted}, detail={sanitized_text[:200]}"
            )
            raise RateLimitError(
                "AI provider quota exhausted. Please check provider billing or try again later."
                if quota_exhausted
                else "AI provider rate limit exceeded",
                retry_after=retry_after,
                retryable=not quota_exhausted,
                quota_exhausted=quota_exhausted,
            )
        elif status == 400 or status == 422:
            logger.error(f"NVIDIA API invalid request: {sanitized_text}")
            raise InvalidRequestError("Invalid request to AI provider")
        elif status >= 500:
            logger.error(f"NVIDIA API provider error {status}: {sanitized_text}")
            raise ProviderUnavailableError("AI provider is temporarily unavailable")
        else:
            logger.error(f"NVIDIA API unexpected status {status}: {sanitized_text}")
            raise AIException(f"AI provider returned status {status}")

    def _parse_response(self, data: dict, model: Optional[str] = None) -> AIResponse:
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
                model=data.get("model") or model or self.model,
                usage=usage,
                finish_reason=finish_reason
            )
        except MalformedResponseError as e:
            raise e
        except Exception as e:
            logger.error(f"Error parsing NVIDIA response: {e.__class__.__name__}")
            raise MalformedResponseError("Failed to parse provider response")
