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
    target_role: Optional[str]
    overall_score: int
    section_scores: Dict[str, int]
    keyword_analysis: Dict[str, List[str]]
    recommendations: List[str]
    strengths: List[str]
