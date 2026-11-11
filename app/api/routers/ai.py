from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import time
from typing import Dict, Optional
import json
import logging
from app.schemas.ai import AIRequest, AIResponse, ResumePolishRequest, ResumePolishResponse
from app.services.nvidia_provider import NvidiaAIProvider
from app.services.ai_provider import (
    AIProvider,
    AIException,
    ProviderAuthenticationError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    InvalidRequestError,
    RateLimitError,
    MalformedResponseError
)
from app.core.config import settings
from app.api.deps import get_current_user_id
from app.core.rate_limiter import ai_rate_limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])

# Backwards compatible alias for tests and legacy callers
_rate_limits: Dict[str, list] = ai_rate_limiter.raw_history

def get_ai_provider() -> AIProvider:
    return NvidiaAIProvider()

def check_rate_limit(
    client_ip: Optional[str] = None,
    user_id: Optional[str] = None,
    request: Optional[Request] = None
):
    """
    Enforces per-user rate limit (falling back to client IP).
    Maintains full backwards compatibility with legacy callers passing a single argument.
    """
    ai_rate_limiter.check(user_id=user_id, client_ip=client_ip, request=request)

@router.post("/test", response_model=AIResponse)
async def test_ai_provider(
    request: Request,
    ai_request: AIRequest,
    user_id: str = Depends(get_current_user_id),
    provider: AIProvider = Depends(get_ai_provider)
):
    """
    Protected test endpoint for the AI provider abstraction.
    Requires authenticated user, enforces rate limits, payload size limits, and concurrency limits.
    """
    check_rate_limit(user_id=user_id, request=request)
    
    # Payload size validation (bounded prompt combination)
    if len(ai_request.system_prompt) + len(ai_request.user_prompt) > 15000:
        raise HTTPException(status_code=400, detail="Combined prompt length too large")

    await ai_rate_limiter.acquire_concurrency(user_id=user_id, request=request)
    try:
        response = await provider.generate(ai_request)
        return response
    except ProviderAuthenticationError:
        raise HTTPException(status_code=500, detail="AI Provider configuration error")
    except ProviderUnavailableError:
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
    except ProviderTimeoutError:
        raise HTTPException(status_code=504, detail="AI request timed out")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="AI provider rate limit exceeded")
    except InvalidRequestError:
        raise HTTPException(status_code=400, detail="Invalid request to AI service")
    except MalformedResponseError:
        raise HTTPException(status_code=502, detail="Bad response from AI service")
    except AIException as e:
        raise HTTPException(status_code=500, detail="An error occurred processing the AI request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await ai_rate_limiter.release_concurrency(user_id=user_id, request=request)

@router.post("/test-provider")
async def test_provider_connectivity(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    provider: AIProvider = Depends(get_ai_provider)
):
    """
    Isolated connectivity test for the NVIDIA provider.
    Requires authentication to protect against unauthenticated API credit exhaustion.
    """
    check_rate_limit(user_id=user_id, request=request)
    
    ai_req = AIRequest(
        system_prompt="Return the word OK.",
        user_prompt="Return the word OK.",
        temperature=0.1,
        max_tokens=16,
        model_id=settings.NVIDIA_MODEL
    )
    
    await ai_rate_limiter.acquire_concurrency(user_id=user_id, request=request)
    try:
        logger.info("=== NVIDIA CONNECTIVITY TEST ===")
        logger.info(f"user_id={user_id}")
        logger.info(f"endpoint={settings.NVIDIA_BASE_URL}")
        logger.info(f"model={settings.NVIDIA_MODEL}")
        
        response = await provider.generate(ai_req)
        
        logger.info("result=SUCCESS")
        return {"status": "success", "content": response.content, "model": response.model}
    except Exception as e:
        logger.info("result=FAIL")
        logger.info(f"error_type={e.__class__.__name__}")
        logger.error(f"Test provider failed: {e.__class__.__name__}")
        raise HTTPException(status_code=500, detail=f"Connectivity test failed: {e.__class__.__name__}")
    finally:
        await ai_rate_limiter.release_concurrency(user_id=user_id, request=request)

@router.post("/resume/polish", response_model=ResumePolishResponse)
async def polish_resume_section(
    request: Request,
    polish_req: ResumePolishRequest,
    user_id: str = Depends(get_current_user_id),
    provider: AIProvider = Depends(get_ai_provider)
):
    check_rate_limit(user_id=user_id, request=request)

    # Factuality constraints
    system_prompt = f"""
You are an expert resume writer. Your job is to improve the provided resume section.
CRITICAL RULES:
1. PRESERVE FACTUAL INFORMATION. DO NOT fabricate metrics, employers, dates, job titles, or technologies.
2. If the user asks you to add metrics or achievements, and the source text does not contain them, DO NOT invent them. Improve the wording only.
3. Return ONLY a valid JSON object matching this schema:
{{
  "suggested_content": "The improved text",
  "changes": ["List of changes made"]
}}
"""

    if polish_req.target_role:
        system_prompt += f"\nTarget Role for context: {polish_req.target_role}"

    user_prompt = f"Section Type: {polish_req.section_type}\n\nOriginal Content:\n{polish_req.original_content}"
    if polish_req.instruction:
        user_prompt += f"\n\nUser Instruction: {polish_req.instruction}"

    ai_req = AIRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3, # low temperature for factual consistency
        max_tokens=1024,
        model_id=settings.NVIDIA_MODEL
    )

    await ai_rate_limiter.acquire_concurrency(user_id=user_id, request=request)
    try:
        response = await provider.generate(ai_req)
        
        # Parse JSON safely after removing reasoning blocks
        try:
            import re
            content = response.content.strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise MalformedResponseError("AI returned non-object JSON")
            
            return ResumePolishResponse(
                original_content=polish_req.original_content,
                suggested_content=parsed.get("suggested_content", polish_req.original_content),
                section_type=polish_req.section_type,
                changes=parsed.get("changes", []),
                model=response.model
            )
        except json.JSONDecodeError:
            raise MalformedResponseError("AI returned invalid JSON")
            
    except ProviderAuthenticationError:
        raise HTTPException(status_code=500, detail="AI Provider configuration error")
    except ProviderUnavailableError:
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
    except ProviderTimeoutError:
        raise HTTPException(status_code=504, detail="AI request timed out")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="AI provider rate limit exceeded")
    except InvalidRequestError:
        raise HTTPException(status_code=400, detail="Invalid request to AI service")
    except MalformedResponseError:
        raise HTTPException(status_code=502, detail="Bad response from AI service")
    except AIException:
        raise HTTPException(status_code=500, detail="An error occurred processing the AI request")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await ai_rate_limiter.release_concurrency(user_id=user_id, request=request)
