from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

class AtsParseResponse(BaseModel):
    document_id: str
    file_name: str
    parsed_structure: Dict[str, Any]

class AtsAnalyzeRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=100, description="UUID of the uploaded ATS document")
    target_role: Optional[str] = Field(None, max_length=100, description="Target job title")
    job_description: Optional[str] = Field(None, max_length=15000, description="Target job description for semantic match")
    bypass_cache: Optional[bool] = False

    @field_validator("document_id")
    @classmethod
    def validate_doc_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("document_id cannot be empty")
        return v.strip()

    @field_validator("target_role", "job_description")
    @classmethod
    def clean_optional_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            return None
        return v.strip() if v else None

class AtsReport(BaseModel):
    ats_document_id: str
    target_role: Optional[str] = None
    overall_score: int = 0
    section_scores: Dict[str, Optional[int]] = Field(default_factory=dict)
    keyword_analysis: Dict[str, List[str]] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    semantic_analysis_available: bool = True
    semantic_error: Optional[str] = None
    analysis_mode: Optional[str] = "ai"
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None
