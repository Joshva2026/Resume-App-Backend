from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from typing import Optional
import uuid

from app.schemas.ats import AtsParseResponse, AtsAnalyzeRequest, AtsReport
from app.services.ats_parser import AtsParser
from app.services.ats_scorer import AtsScorer
from app.services.ats_semantic_analyzer import AtsSemanticAnalyzer
from app.services.nvidia_provider import NvidiaAIProvider
from app.api.routers.ai import check_rate_limit
from app.api.deps import get_current_user_id, get_supabase_client
from supabase import Client

router = APIRouter(prefix="/ats", tags=["ats"])

@router.post("/parse", response_model=AtsParseResponse)
async def parse_resume(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    supabase: Client = Depends(get_supabase_client)
):
    check_rate_limit(request.client.host if request.client else "unknown")
    
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are currently supported")
        
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024: # 5MB limit
        raise HTTPException(status_code=400, detail="File too large")
        
    parser = AtsParser()
    try:
        if file.content_type == "application/pdf":
            raw_text = parser.parse_pdf(contents)
        else:
            raw_text = parser.parse_docx(contents)
            
        structured_data = parser.structure_text(raw_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse document: {str(e)}")
        
    # Persist to Supabase
    try:
        response = supabase.table("ats_documents").insert({
            "user_id": user_id,
            "file_name": file.filename,
            "file_type": file.content_type,
            "file_size": len(contents),
            "parsed_text": raw_text,
            "parsed_structure": structured_data
        }).execute()
        doc_id = response.data[0]["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to persist document to database")
        
    return AtsParseResponse(
        document_id=doc_id,
        file_name=file.filename,
        parsed_structure=structured_data
    )

@router.post("/analyze", response_model=AtsReport)
async def analyze_resume(
    request: Request,
    analyze_req: AtsAnalyzeRequest,
    user_id: str = Depends(get_current_user_id),
    supabase: Client = Depends(get_supabase_client)
):
    check_rate_limit(request.client.host if request.client else "unknown")
    
    # Fetch from DB
    try:
        doc_response = supabase.table("ats_documents").select("*").eq("id", analyze_req.document_id).eq("user_id", user_id).execute()
        if not doc_response.data:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        parsed_structure = doc_response.data[0]["parsed_structure"]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch document")
    
    # 1. PII Exclusion & Scoring Input Preparation
    scorer = AtsScorer()
    sanitized_text = scorer.sanitize_for_scoring(parsed_structure)
    
    if len(sanitized_text) < 50:
        raise HTTPException(status_code=400, detail="Document contains insufficient content for analysis")
        
    # 2. Deterministic Scoring
    score_result = scorer.calculate_score(sanitized_text, analyze_req.job_description)
    
    # 3. AI Semantic Analysis
    provider = NvidiaAIProvider()
    analyzer = AtsSemanticAnalyzer(provider)
    semantic_result = await analyzer.analyze(sanitized_text, analyze_req.job_description)
    
    report = AtsReport(
        ats_document_id=analyze_req.document_id,
        target_role=analyze_req.target_role,
        overall_score=score_result["overall"],
        section_scores=score_result["sections"],
        keyword_analysis={
            "matched": semantic_result["matched_keywords"],
            "missing": semantic_result["missing_keywords"]
        },
        recommendations=semantic_result["recommendations"],
        strengths=semantic_result["strengths"]
    )
    
    # Persist report
    try:
        supabase.table("ats_reports").insert({
            "user_id": user_id,
            "ats_document_id": analyze_req.document_id,
            "target_role": analyze_req.target_role,
            "job_description": analyze_req.job_description,
            "overall_score": score_result["overall"],
            "section_scores": score_result["sections"],
            "keyword_analysis": {
                "matched": semantic_result["matched_keywords"],
                "missing": semantic_result["missing_keywords"]
            },
            "recommendations": semantic_result["recommendations"]
        }).execute()
    except Exception as e:
        pass # We return the report anyway if DB fails at this non-critical step, but in production we might want to log this
    
    return report

@router.get("/history")
async def get_ats_history(
    user_id: str = Depends(get_current_user_id),
    supabase: Client = Depends(get_supabase_client)
):
    try:
        response = supabase.table("ats_reports").select(
            "id, ats_document_id, overall_score, target_role, created_at, ats_documents(file_name)"
        ).eq("user_id", user_id).order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

