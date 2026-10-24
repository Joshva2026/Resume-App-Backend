from fastapi.testclient import TestClient
from app.main import app
from app.api.deps import get_current_user_id, get_supabase_client
from app.api.routers.ats import AtsSemanticAnalyzer
from unittest.mock import patch, MagicMock, AsyncMock

client = TestClient(app)

def mock_get_user():
    return "fake_user"

def test_analyze_no_jd():
    mock_db = MagicMock()
    mock_db.table().select().eq().eq().execute.return_value = MagicMock(
        data=[{"parsed_structure": {"experience": "Developed software using Python and FastAPI. " * 5, "skills": "Python"}}]
    )
    mock_db.table().insert().execute.return_value = None

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    with patch.object(AtsSemanticAnalyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = {
            "matched_keywords": ["python"],
            "missing_keywords": ["java"],
            "recommendations": ["add java"],
            "strengths": ["python is good"]
        }

        # CASE A: No Target Role or JD
        response = client.post("/ats/analyze", json={
            "document_id": "test_doc_id"
        })
        
        assert response.status_code == 200
        report = response.json()
        assert report["ats_document_id"] == "test_doc_id"
        assert report["target_role"] is None
        assert report["section_scores"]["JD Match"] is None
        assert report["section_scores"]["Target Role"] is None
        assert isinstance(report["section_scores"]["Experience"], int)

def test_analyze_with_jd():
    mock_db = MagicMock()
    mock_db.table().select().eq().eq().execute.return_value = MagicMock(
        data=[{"parsed_structure": {"experience": "Developed software using Python and FastAPI. " * 5, "skills": "Python"}}]
    )
    mock_db.table().insert().execute.return_value = None

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    with patch.object(AtsSemanticAnalyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = {
            "matched_keywords": ["python"],
            "missing_keywords": ["java"],
            "recommendations": ["add java"],
            "strengths": ["python is good"],
            "analysis_mode": "ai",
            "semantic_analysis_available": True
        }

        # CASE B: With Target Role and JD
        response = client.post("/ats/analyze", json={
            "document_id": "test_doc_id",
            "target_role": "Software Engineer",
            "job_description": "We need someone who develops software with python"
        })
        
        assert response.status_code == 200
        report = response.json()
        assert report["ats_document_id"] == "test_doc_id"
        assert report["target_role"] == "Software Engineer"
        assert isinstance(report["section_scores"]["JD Match"], int)
        assert isinstance(report["section_scores"]["Target Role"], int)
        assert report["ai_provider"] in ["Groq", "xAI"]
        assert report["ai_model"] in ["openai/gpt-oss-120b", "grok-2-latest"]

def test_analyze_provider_timeout_fallback():
    mock_db = MagicMock()
    mock_db.table().select().eq().eq().execute.return_value = MagicMock(
        data=[{"parsed_structure": {
            "experience": "Developed software services using Python and FastAPI. Increased throughput by 35%. " * 3,
            "skills": "Python, SQL, AWS, Docker",
            "education": "BS in Computer Science",
            "projects": "Built cloud data pipelines using Python"
        }}]
    )
    mock_db.table().insert().execute.return_value = None

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    from app.services.ai_provider import ProviderTimeoutError
    from app.services.groq_provider import GroqAIProvider
    from app.services.grok_provider import XAIGrokAIProvider

    with patch.object(GroqAIProvider, 'generate', new_callable=AsyncMock) as mock_groq, \
         patch.object(XAIGrokAIProvider, 'generate', new_callable=AsyncMock) as mock_grok:
        mock_groq.side_effect = ProviderTimeoutError("The AI response took too long. Please try again.")
        mock_grok.side_effect = ProviderTimeoutError("The AI response took too long. Please try again.")

        response = client.post("/ats/analyze", json={
            "document_id": "test_doc_timeout",
            "target_role": "Backend Engineer",
            "job_description": "We are seeking a Backend Engineer with Python, SQL, and Kubernetes experience."
        })

        assert response.status_code == 200
        report = response.json()
        assert report["ats_document_id"] == "test_doc_timeout"
        assert report["target_role"] == "Backend Engineer"
        assert report["semantic_analysis_available"] is False
        assert report["analysis_mode"] == "deterministic_fallback"
        assert "Displaying local deterministic keyword match" in report["semantic_error"]
        
        # Verify matched and missing keywords are extracted deterministically
        matched = report["keyword_analysis"]["matched"]
        missing = report["keyword_analysis"]["missing"]
        assert "Python" in matched
        assert "SQL" in matched
        assert "Kubernetes" in missing
        
        # Verify score is dynamically calculated (not fixed 85 or 90)
        assert isinstance(report["overall_score"], int)
        assert 0 < report["overall_score"] < 100
        assert len(report["recommendations"]) > 0
        assert len(report["strengths"]) > 0

def test_parse_saved_resume():
    mock_db = MagicMock()
    mock_db.table("resumes").select().eq().eq().execute.return_value = MagicMock(
        data=[{"id": "res_123", "resume_name": "Software Engineer Resume", "professional_summary": "Experienced engineer."}]
    )
    mock_db.table("resume_sections").select().eq().order().execute.return_value = MagicMock(
        data=[
            {
                "section_type": "experience",
                "content": {"job_title": "Backend Dev", "company": "Tech Corp", "description": "Built Python APIs."}
            },
            {
                "section_type": "skill",
                "content": {"name": "Python"}
            }
        ]
    )
    mock_db.table("ats_documents").insert().execute.return_value = MagicMock(
        data=[{"id": "new_doc_456"}]
    )

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    response = client.post("/ats/parse-resume/res_123")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["document_id"] == "new_doc_456"
    assert res_data["file_name"] == "Software Engineer Resume.pdf"
    assert "experience" in res_data["parsed_structure"]
    assert "Backend Dev at Tech Corp" in res_data["parsed_structure"]["experience"]

def test_document_not_found_returns_404():
    mock_db = MagicMock()
    mock_db.table().select().eq().eq().execute.return_value = MagicMock(data=[])

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    response = client.post("/ats/analyze", json={"document_id": "non_existent_doc"})
    assert response.status_code == 404
    assert "not found or access denied" in response.json()["detail"].lower()

def test_insufficient_content_returns_422():
    mock_db = MagicMock()
    mock_db.table().select().eq().eq().execute.return_value = MagicMock(
        data=[{"parsed_structure": {"skills": "Hi"}}]
    )

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    response = client.post("/ats/analyze", json={"document_id": "sparse_doc"})
    assert response.status_code == 422
    assert "insufficient content" in response.json()["detail"].lower()

def test_fallback_persistence_when_strengths_column_unmigrated():
    mock_db = MagicMock()
    mock_db.table().select().eq().eq().execute.return_value = MagicMock(
        data=[{"parsed_structure": {
            "experience": "Senior Software Engineer. Developed cloud platforms with Python and Go. " * 3,
            "skills": "Python, Go, Docker"
        }}]
    )
    
    # First insert call (with strengths) raises exception simulating missing column in DB
    # Second insert call (fallback without strengths) succeeds
    mock_insert = MagicMock()
    call_count = 0
    def side_effect_insert(data):
        nonlocal call_count
        call_count += 1
        m = MagicMock()
        if "strengths" in data:
            m.execute.side_effect = Exception("column 'strengths' does not exist")
        else:
            m.execute.return_value = MagicMock(data=[{"id": "rep_1"}])
        return m

    mock_db.table("ats_reports").insert.side_effect = side_effect_insert

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    with patch.object(AtsSemanticAnalyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = {
            "matched_keywords": ["python"],
            "missing_keywords": [],
            "recommendations": ["keep learning"],
            "strengths": ["strong python expertise"]
        }

        response = client.post("/ats/analyze", json={"document_id": "test_doc_fallback"})
        assert response.status_code == 200
        # Analysis succeeds and returns report with strengths even if DB table hadn't been migrated yet
        report = response.json()
        assert report["strengths"] == ["strong python expertise"]
        assert call_count >= 2

def test_historical_report_deserialization():
    from app.schemas.ats import AtsReport
    # Historical DB record without strengths, target_role, or section_scores
    old_record = {
        "ats_document_id": "doc_old_123",
        "overall_score": 75
    }
    report = AtsReport.model_validate(old_record)
    assert report.ats_document_id == "doc_old_123"
    assert report.overall_score == 75
    assert report.strengths == []
    assert report.recommendations == []
    assert report.section_scores == {}
    assert report.target_role is None

def test_ats_history_multiple_records():
    mock_db = MagicMock()
    mock_records = [
        {
            "id": "rep_1",
            "ats_document_id": "doc_1",
            "overall_score": 88,
            "target_role": "Backend Engineer",
            "created_at": "2026-10-24T18:00:00Z",
            "ats_documents": {"file_name": "resume1.pdf"}
        },
        {
            "id": "rep_2",
            "ats_document_id": "doc_2",
            "overall_score": 62,
            "target_role": "Frontend Engineer",
            "created_at": "2026-10-23T12:00:00Z",
            "ats_documents": {"file_name": "resume2.docx"}
        }
    ]
    mock_db.table("ats_reports").select().eq().order().execute.return_value = MagicMock(data=mock_records)

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    response = client.get("/ats/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["overall_score"] == 88
    assert data[1]["overall_score"] == 62

def test_ats_analyze_ai_mode_provider_metadata_no_nameerror():
    """
    Regression test for BUG 1: Ensures that when AI semantic analysis succeeds
    (analysis_mode='ai'), the router accesses settings without raising NameError
    and returns correct provider and model metadata.
    """
    from app.services.groq_provider import GroqAIProvider
    from app.core.config import settings

    mock_db = MagicMock()
    mock_db.table().select().eq().eq().execute.return_value = MagicMock(
        data=[{"parsed_structure": {
            "experience": "Senior Backend Developer with 5 years of Python experience.",
            "skills": "Python, FastAPI, Docker, PostgreSQL",
            "education": "B.S. Computer Science"
        }}]
    )
    mock_db.table().insert().execute.return_value = None

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user_id] = mock_get_user
    app.dependency_overrides[get_supabase_client] = mock_get_db

    with patch.object(settings, "GROQ_API_KEY", "gsk_test_mock_key"), \
         patch.object(settings, "ATS_GROQ_MODEL", "openai/gpt-oss-120b"), \
         patch.object(AtsSemanticAnalyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:

        mock_analyze.return_value = {
            "matched_keywords": ["Python", "FastAPI"],
            "missing_keywords": ["Redis"],
            "missing_required": ["Redis"],
            "missing_preferred": [],
            "recommendations": ["Add Redis caching project"],
            "strengths": ["Strong Python experience"],
            "analysis_mode": "ai",
            "semantic_analysis_available": True
        }

        response = client.post("/ats/analyze", json={
            "document_id": "doc_regression_nameerror",
            "target_role": "Backend Engineer",
            "job_description": "Looking for Backend Engineer skilled in Python and Redis"
        })

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        report = response.json()
        assert report["ai_provider"] == "Groq"
        assert report["ai_model"] == "openai/gpt-oss-120b"
        assert report["analysis_mode"] == "ai"
        assert report["semantic_analysis_available"] is True

def test_groq_nvidia_provider_isolation():
    """
    Verifies that ATS AI analysis uses Groq with openai/gpt-oss-120b,
    while AI Assistant uses NVIDIA with openai/gpt-oss-20b.
    """
    from app.services.groq_provider import GroqAIProvider
    from app.services.nvidia_provider import NvidiaAIProvider
    from app.services.ats_semantic_analyzer import get_default_ats_provider
    from app.core.config import settings

    with patch.object(settings, "GROQ_API_KEY", "gsk_isolation_test_key"):
        ats_provider = get_default_ats_provider()
        assert isinstance(ats_provider, GroqAIProvider)
        assert ats_provider.model == "openai/gpt-oss-120b"

    chat_provider = NvidiaAIProvider()
    assert chat_provider.model == "openai/gpt-oss-20b"
    assert settings.NVIDIA_MODEL == "openai/gpt-oss-20b"

def test_different_job_descriptions_produce_distinct_cache_keys():
    """
    Verifies that different job descriptions for the same resume produce
    distinct cache keys so analysis results are never improperly shared.
    """
    from app.services.ats_cache import AtsAnalysisCache

    user_id = "test_user_unique_cache"
    resume = "Experienced software engineer with Python"
    role = "Software Engineer"
    jd1 = "Looking for a Python and Docker developer"
    jd2 = "Looking for a React and TypeScript frontend developer"

    key1 = AtsAnalysisCache.make_key(user_id, resume, jd1, role, "ats_v2.0")
    key2 = AtsAnalysisCache.make_key(user_id, resume, jd2, role, "ats_v2.0")

    assert key1 != key2, "Different job descriptions must generate distinct cache keys"


