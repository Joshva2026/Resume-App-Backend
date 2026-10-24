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
            "strengths": ["python is good"]
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
    from app.services.nvidia_provider import NvidiaAIProvider

    with patch.object(NvidiaAIProvider, 'generate', new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = ProviderTimeoutError("The AI response took too long. Please try again.")

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

