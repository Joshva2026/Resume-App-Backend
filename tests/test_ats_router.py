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
