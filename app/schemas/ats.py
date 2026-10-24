from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AtsParseResponse(BaseModel):
    document_id: str
    file_name: str
    parsed_structure: Dict[str, Any]

class AtsAnalyzeRequest(BaseModel):
    document_id: str
    target_role: Optional[str] = None
    job_description: Optional[str] = None

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
