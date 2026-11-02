from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from app.core.config import settings

def get_allowed_models() -> set[str]:
    """Returns the set of server-permitted AI model IDs."""
    models = set()
    if getattr(settings, "NVIDIA_MODEL", None):
        models.add(settings.NVIDIA_MODEL.strip())
    if getattr(settings, "ATS_GROQ_MODEL", None):
        models.add(settings.ATS_GROQ_MODEL.strip())
    if getattr(settings, "ATS_GROK_MODEL", None):
        models.add(settings.ATS_GROK_MODEL.strip())
    # Baseline defaults in case config is sparse
    models.update({"openai/gpt-oss-20b", "openai/gpt-oss-120b", "grok-2-latest"})
    return models

class AIRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1, max_length=5000, description="System instruction prompt")
    user_prompt: str = Field(..., min_length=1, max_length=10000, description="User instruction or task input")
    history: List[Dict[str, str]] = Field(default_factory=list, description="List of previous messages: [{'role': 'user', 'content': '...'}, ...]")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=16, le=4096, description="Maximum output tokens (16 to 4096)")
    model_id: Optional[str] = Field(default=None, max_length=100, description="Allowed model identifier")
    timeout: Optional[float] = Field(default=None, ge=1.0, le=120.0)
    response_format: Optional[Dict[str, Any]] = None
    
    model_config = {"protected_namespaces": ()}

    @field_validator("system_prompt", "user_prompt")
    @classmethod
    def validate_non_empty_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or pure whitespace")
        return v

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        cleaned = v.strip()
        allowed = get_allowed_models()
        if cleaned not in allowed:
            raise ValueError(f"Invalid model_id '{cleaned}'. Allowed models: {sorted(allowed)}")
        return cleaned

    @field_validator("history")
    @classmethod
    def validate_history(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(v) > 30:
            raise ValueError("History cannot exceed 30 turns")
        valid_roles = {"system", "user", "assistant"}
        for idx, item in enumerate(v):
            if not isinstance(item, dict):
                raise ValueError(f"History turn {idx} must be an object")
            role = item.get("role")
            content = item.get("content")
            if role not in valid_roles:
                raise ValueError(f"History turn {idx} has invalid role: '{role}'")
            if not isinstance(content, str):
                raise ValueError(f"History turn {idx} content must be a string")
            if len(content) > 5000:
                raise ValueError(f"History turn {idx} content exceeds 5000 character limit")
        return v

class AIResponseUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

class AIResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Optional[AIResponseUsage] = None
    finish_reason: Optional[str] = None

class ResumePolishRequest(BaseModel):
    section_type: str = Field(..., min_length=1, max_length=50, description="e.g., summary, experience, project, skills")
    original_content: str = Field(..., min_length=1, max_length=5000, description="Resume section text to polish")
    target_role: Optional[str] = Field(None, max_length=100, description="Optional target role for contextual wording")
    instruction: Optional[str] = Field(None, max_length=500, description="Optional user directive")

    @field_validator("section_type", "original_content")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or pure whitespace")
        return v.strip()

    @field_validator("target_role", "instruction")
    @classmethod
    def clean_optional_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            return None
        return v.strip() if v else None

class ResumePolishResponse(BaseModel):
    original_content: str
    suggested_content: str
    section_type: str
    changes: List[str] = []
    model: str

class ChatMessageRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, max_length=100)
    message: str = Field(..., max_length=4000, description="User prompt message")
    model_id: Optional[str] = Field(None, max_length=100)
    model_config = {"protected_namespaces": ()}

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        cleaned = v.strip()
        allowed = get_allowed_models()
        if cleaned not in allowed:
            raise ValueError(f"Invalid model_id '{cleaned}'. Allowed models: {sorted(allowed)}")
        return cleaned
