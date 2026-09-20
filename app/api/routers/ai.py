from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import time
from typing import Dict
import json
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

router = APIRouter(prefix="/ai", tags=["ai"])

# Simple in-memory rate limiting (for demo/MVP purposes)
# In production, use Redis or a proper rate limiting library
_rate_limits: Dict[str, list] = {}

def get_ai_provider() -> AIProvider:
    return NvidiaAIProvider()

def check_rate_limit(client_ip: str):
    now = time.time()
    minute_ago = now - 60
    
    # Clean up old requests
    if client_ip in _rate_limits:
        _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if t > minute_ago]
    else:
        _rate_limits[client_ip] = []
        
    if len(_rate_limits[client_ip]) >= settings.AI_RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for AI requests")
        
    _rate_limits[client_ip].append(now)

@router.post("/test", response_model=AIResponse)
async def test_ai_provider(
    request: Request,
    ai_request: AIRequest,
    provider: AIProvider = Depends(get_ai_provider)
):
    """
    Test endpoint for the AI provider abstraction.
    NOTE: In a real environment, this should be protected by the auth middleware.
    """
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)
    
    # Payload size validation (basic)
    if len(ai_request.system_prompt) + len(ai_request.user_prompt) > 20000:
        raise HTTPException(status_code=400, detail="Prompt too large")

    try:
        response = await provider.generate(ai_request)
        return response
    except ProviderAuthenticationError:
        # Don't expose internal auth failures directly, but log securely internally
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
        # Fallback for unexpected errors
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/resume/polish", response_model=ResumePolishResponse)
async def polish_resume_section(
    request: Request,
    polish_req: ResumePolishRequest,
    provider: AIProvider = Depends(get_ai_provider)
):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

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
        max_tokens=1024
    )

    try:
        response = await provider.generate(ai_req)
        
        # Parse JSON
        try:
            # Strip potential markdown blocks if model misbehaves
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            parsed = json.loads(content)
            
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
