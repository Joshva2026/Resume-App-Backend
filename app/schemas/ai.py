from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AIRequest(BaseModel):
    system_prompt: str = Field(..., max_length=5000)
    user_prompt: str = Field(..., max_length=15000)
    history: List[Dict[str, str]] = Field(default_factory=list, description="List of previous messages: [{'role': 'user', 'content': '...'}, ...]")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    model_id: Optional[str] = None

class AIResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class AIResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Optional[AIResponseUsage] = None
    finish_reason: Optional[str] = None

class ResumePolishRequest(BaseModel):
    section_type: str = Field(..., description="e.g., summary, experience, project, skills")
    original_content: str = Field(..., max_length=5000)
    target_role: Optional[str] = Field(None, max_length=100)
    instruction: Optional[str] = Field(None, max_length=500)

class ResumePolishResponse(BaseModel):
    original_content: str
    suggested_content: str
    section_type: str
    changes: List[str] = []
    model: str
