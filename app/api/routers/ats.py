from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from typing import Optional, Dict, Any
import logging
import uuid

from app.schemas.ats import AtsParseResponse, AtsAnalyzeRequest, AtsReport
from app.services.ats_parser import AtsParser
from app.services.ats_scorer import AtsScorer
from app.services.ats_semantic_analyzer import AtsSemanticAnalyzer
from app.services.nvidia_provider import NvidiaAIProvider
from app.api.routers.ai import check_rate_limit
from app.api.deps import get_current_user_id, get_supabase_client
from supabase import Client

logger = logging.getLogger(__name__)

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
        raise HTTPException(status_code=422, detail="Only PDF and DOCX files are currently supported")
        
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024: # 5MB limit
        raise HTTPException(status_code=422, detail="File too large")
        
    parser = AtsParser()
    try:
        if file.content_type == "application/pdf":
            raw_text = parser.parse_pdf(contents)
        else:
            raw_text = parser.parse_docx(contents)
            
        structured_data = parser.structure_text(raw_text)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {str(e)}")
        
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

@router.post("/parse-resume/{resume_id}", response_model=AtsParseResponse)
async def parse_saved_resume(
    request: Request,
    resume_id: str,
    user_id: str = Depends(get_current_user_id),
    supabase: Client = Depends(get_supabase_client)
):
    """
    Directly converts an existing ResumeForge AI saved resume into an ATS document
    without requiring the user to download and re-upload files.
    """
    check_rate_limit(request.client.host if request.client else "unknown")
    
    try:
        resume_res = supabase.table("resumes").select("*").eq("id", resume_id).eq("user_id", user_id).execute()
        if not resume_res.data:
            raise HTTPException(status_code=404, detail="Resume not found or access denied")
        resume_data = resume_res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch resume: {str(e)}")

    resume_name = resume_data.get("resume_name") or "Saved Resume"
    
    structured_data: Dict[str, Any] = {
        "personal_info": "",
        "summary": resume_data.get("professional_summary") or "",
        "experience": "",
        "education": "",
        "skills": "",
        "projects": "",
        "certifications": "",
        "other": ""
    }
    
    exp_lines = []
    edu_lines = []
    skill_lines = []
    proj_lines = []
    cert_lines = []
    
    # Check resume_sections
    try:
        sections_res = supabase.table("resume_sections").select("*").eq("resume_id", resume_id).order("display_order").execute()
        if sections_res.data:
            for sec in sections_res.data:
                stype = (sec.get("section_type") or "").lower()
                content = sec.get("content")
                if not content or not isinstance(content, dict):
                    continue
                
                if stype in ["experience", "employment"]:
                    title = content.get("job_title") or content.get("title") or ""
                    comp = content.get("company") or content.get("employer") or ""
                    desc = content.get("description") or content.get("bullets") or ""
                    period = f"{content.get('start_date', '')} - {content.get('end_date', '')}".strip(" -")
                    exp_lines.append(f"{title} at {comp} ({period})\n{desc}".strip())
                elif stype in ["education", "academic"]:
                    deg = content.get("degree") or content.get("field_of_study") or ""
                    inst = content.get("institution") or content.get("school") or ""
                    period = f"{content.get('start_date', '')} - {content.get('end_date', '')}".strip(" -")
                    edu_lines.append(f"{deg}, {inst} ({period})".strip())
                elif stype in ["skill", "skills"]:
                    name = content.get("name") or content.get("skill") or ""
                    if name:
                        skill_lines.append(name)
                elif stype in ["project", "projects"]:
                    name = content.get("title") or content.get("name") or ""
                    desc = content.get("description") or ""
                    tools = content.get("technologies") or content.get("tools") or ""
                    proj_lines.append(f"{name}: {desc} ({tools})".strip())
                elif stype in ["certification", "certifications", "achievement", "award"]:
                    title = content.get("title") or content.get("name") or ""
                    issuer = content.get("subtitle") or content.get("custom1") or ""
                    cert_lines.append(f"{title} - {issuer}".strip(" -"))
                elif stype == "summary":
                    if not structured_data["summary"]:
                        structured_data["summary"] = content.get("text") or content.get("description") or ""
    except Exception as e:
        logger.warning(f"Failed to fetch resume_sections: {e}")

    # Fallback to legacy tables if sections were empty
    if not exp_lines:
        try:
            legacy_exp = supabase.table("experiences").select("*").eq("resume_id", resume_id).execute()
            for e in legacy_exp.data:
                exp_lines.append(f"{e.get('job_title', '')} at {e.get('company', '')}\n{e.get('description', '')}".strip())
        except Exception:
            pass

    if not edu_lines:
        try:
            legacy_edu = supabase.table("educations").select("*").eq("resume_id", resume_id).execute()
            for ed in legacy_edu.data:
                edu_lines.append(f"{ed.get('degree', '')}, {ed.get('institution', '')}".strip())
        except Exception:
            pass

    if exp_lines:
        structured_data["experience"] = "\n\n".join(exp_lines)
    if edu_lines:
        structured_data["education"] = "\n".join(edu_lines)
    if skill_lines:
        structured_data["skills"] = ", ".join(skill_lines)
    if proj_lines:
        structured_data["projects"] = "\n\n".join(proj_lines)
    if cert_lines:
        structured_data["certifications"] = "\n".join(cert_lines)

    raw_text = f"{resume_name}\n\n"
    for k, v in structured_data.items():
        if isinstance(v, str) and v.strip():
            raw_text += f"{k.title()}:\n{v.strip()}\n\n"

    try:
        response = supabase.table("ats_documents").insert({
            "user_id": user_id,
            "file_name": f"{resume_name}.pdf",
            "file_type": "application/json-resume",
            "file_size": len(raw_text.encode("utf-8")),
            "parsed_text": raw_text,
            "parsed_structure": structured_data
        }).execute()
        doc_id = response.data[0]["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist document to database: {str(e)}")

    return AtsParseResponse(
        document_id=doc_id,
        file_name=f"{resume_name}.pdf",
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch document")
    
    # 1. PII Exclusion & Scoring Input Preparation
    scorer = AtsScorer()
    sanitized_text = scorer.sanitize_for_scoring(parsed_structure)
    
    if len(sanitized_text) < 50:
        raise HTTPException(status_code=422, detail="Document contains insufficient content for analysis")
        
    # 2. Deterministic Scoring
    score_result = scorer.calculate_score(
        sanitized_text, 
        analyze_req.job_description, 
        analyze_req.target_role,
        parsed_structure=parsed_structure
    )
    
    # 3. AI Semantic Analysis (with deterministic fallback)
    provider = NvidiaAIProvider()
    analyzer = AtsSemanticAnalyzer(provider)
    semantic_result = await analyzer.analyze(
        sanitized_text, 
        analyze_req.job_description,
        analyze_req.target_role,
        section_scores=score_result["sections"],
        parsed_structure=parsed_structure
    )
    
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
        strengths=semantic_result["strengths"],
        semantic_analysis_available=semantic_result.get("semantic_analysis_available", True),
        semantic_error=semantic_result.get("semantic_error"),
        analysis_mode=semantic_result.get("analysis_mode", "ai")
    )
    
    # Persist report safely
    try:
        report_data = {
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
            "recommendations": semantic_result["recommendations"],
            "strengths": semantic_result["strengths"]
        }
        try:
            supabase.table("ats_reports").insert(report_data).execute()
        except Exception:
            # Fallback if strengths column has not yet been migrated in active database
            report_data.pop("strengths", None)
            supabase.table("ats_reports").insert(report_data).execute()
    except Exception as e:
        logger.warning(f"Failed to persist ATS report: {e}")
    
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
