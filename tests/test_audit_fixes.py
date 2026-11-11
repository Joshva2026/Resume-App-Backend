import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.api.deps import get_current_user_id, get_supabase_client
from app.api.routers.ai import get_ai_provider, _rate_limits
from app.core.rate_limiter import ai_rate_limiter
from app.schemas.ai import AIResponse, AIResponseUsage
from app.services.ats_cache import AtsAnalysisCache
from app.services.ats_parser import AtsParser
from app.services.ats_scorer import AtsScorer
from app.schemas.ats import AtsReport

client = TestClient(app)


def mock_user():
    return "audit-user-123"


@pytest.fixture(autouse=True)
def reset_state():
    app.dependency_overrides.clear()
    _rate_limits.clear()
    ai_rate_limiter.clear()
    AtsAnalysisCache.clear()
    yield
    app.dependency_overrides.clear()
    _rate_limits.clear()
    ai_rate_limiter.clear()
    AtsAnalysisCache.clear()


def test_ats_parse_rejects_invalid_magic_bytes_and_empty_file():
    app.dependency_overrides[get_current_user_id] = mock_user
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()

    # Empty PDF file
    empty_resp = client.post(
        "/ats/parse",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert empty_resp.status_code == 422
    assert "empty" in empty_resp.json()["detail"].lower()

    # Spoofed PDF magic bytes
    fake_pdf = client.post(
        "/ats/parse",
        files={"file": ("evil.pdf", b"NOT_A_REAL_PDF_FILE", "application/pdf")},
    )
    assert fake_pdf.status_code == 422
    assert "header" in fake_pdf.json()["detail"].lower()

    # Spoofed DOCX magic bytes
    fake_docx = client.post(
        "/ats/parse",
        files={
            "file": (
                "evil.docx",
                b"NOT_A_ZIP_ARCHIVE",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert fake_docx.status_code == 422
    assert "header" in fake_docx.json()["detail"].lower()


def test_ats_parse_rate_limited_by_user_id():
    app.dependency_overrides[get_current_user_id] = mock_user
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()

    client.post(
        "/ats/parse",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert "user:audit-user-123" in ai_rate_limiter.raw_history


def test_ats_parse_saved_resume_supports_legacy_title_column():
    app.dependency_overrides[get_current_user_id] = mock_user
    mock_db = MagicMock()
    tables = {}

    def table_side_effect(name):
        if name not in tables:
            tables[name] = MagicMock()
        return tables[name]

    mock_db.table.side_effect = table_side_effect
    tables["resumes"] = MagicMock()
    tables["resumes"].select().eq().eq().execute.return_value = MagicMock(
        data=[{"id": "res-legacy", "resume_name": "Legacy Resume", "professional_summary": "Summary"}]
    )
    tables["resume_sections"] = MagicMock()
    tables["resume_sections"].select().eq().order().execute.return_value = MagicMock(data=[])
    tables["experiences"] = MagicMock()
    tables["experiences"].select().eq().execute.return_value = MagicMock(
        data=[{"title": "Staff Engineer", "company": "Acme", "description": "Built APIs"}]
    )
    tables["educations"] = MagicMock()
    tables["educations"].select().eq().execute.return_value = MagicMock(data=[])
    tables["ats_documents"] = MagicMock()
    tables["ats_documents"].insert().execute.return_value = MagicMock(data=[{"id": "doc-legacy-1"}])

    app.dependency_overrides[get_supabase_client] = lambda: mock_db

    resp = client.post("/ats/parse-resume/res-legacy")
    assert resp.status_code == 200
    body = resp.json()
    assert "Staff Engineer" in body["parsed_structure"]["experience"]


def test_ats_scorer_handles_none_section_scores():
    scorer = AtsScorer()
    result = scorer.calculate_deterministic_semantics(
        text="Experienced software engineer using Python and FastAPI.",
        jd="Need Python developer",
        target_role="Backend Engineer",
        section_scores={
            "Completeness": None,
            "Education": None,
            "Projects": None,
            "Readability": None,
        },
    )
    assert result["analysis_mode"] == "deterministic_fallback"
    assert isinstance(result["strengths"], list)
    assert isinstance(result["recommendations"], list)


def test_ats_cache_purges_expired_before_eviction():
    AtsAnalysisCache.clear()
    dummy_report = AtsReport(
        ats_document_id="d1",
        overall_score=80,
        section_scores={},
        keyword_analysis={"matched": [], "missing": []},
        recommendations=[],
        strengths=[],
    )
    with patch.object(AtsAnalysisCache, "MAX_ENTRIES", 2):
        # Insert 1 already-expired entry and 1 fresh entry
        AtsAnalysisCache._cache["expired"] = (1.0, dummy_report)
        AtsAnalysisCache._cache["fresh"] = (9999999999.0, dummy_report)
        AtsAnalysisCache.set("new_key", dummy_report, ttl_seconds=60.0)
        assert "expired" not in AtsAnalysisCache._cache
        assert "fresh" in AtsAnalysisCache._cache
        assert "new_key" in AtsAnalysisCache._cache


def test_ats_parser_encrypted_pdf_raises_clear_error():
    parser = AtsParser()
    fake_reader = MagicMock()
    fake_reader.is_encrypted = True
    fake_reader.decrypt.return_value = 0

    with patch("pypdf.PdfReader", return_value=fake_reader):
        with pytest.raises(ValueError, match="password-protected"):
            parser.parse_pdf(b"%PDF-1.4 encrypted")


def test_chat_list_conversations_handles_null_model_id_and_delete_404():
    app.dependency_overrides[get_current_user_id] = mock_user
    mock_db = MagicMock()
    mock_db.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = MagicMock(
        data=[
            {
                "id": "c1",
                "title": "Legacy chat",
                "model_id": None,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
            }
        ]
    )
    app.dependency_overrides[get_supabase_client] = lambda: mock_db

    list_resp = client.get("/ai/conversations")
    assert list_resp.status_code == 200
    assert list_resp.json()[0]["model_id"] == "openai/gpt-oss-20b"

    # Missing conversation on DELETE must return 404 instead of 500
    mock_db.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[]
    )
    del_resp = client.delete("/ai/conversations/missing-conv-id")
    assert del_resp.status_code == 404


def test_chat_stream_releases_concurrency_on_404():
    app.dependency_overrides[get_current_user_id] = mock_user
    mock_db = MagicMock()
    mock_db.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[]
    )
    app.dependency_overrides[get_supabase_client] = lambda: mock_db
    app.dependency_overrides[get_ai_provider] = lambda: MagicMock()

    resp = client.post(
        "/ai/chat/stream",
        json={"message": "Hello", "conversation_id": "nonexistent"},
    )
    assert resp.status_code == 404
    assert ai_rate_limiter._active.get("user:audit-user-123", 0) == 0


def test_ai_polish_strips_think_block_before_json_parse():
    app.dependency_overrides[get_current_user_id] = mock_user
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(
        return_value=AIResponse(
            content='<think>Let me think {not json here}.</think>\n{"suggested_content": "Built scalable services.", "changes": ["Improved clarity"]}',
            provider="nvidia",
            model="openai/gpt-oss-20b",
            usage=AIResponseUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason="stop",
        )
    )
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    resp = client.post(
        "/ai/resume/polish",
        json={"section_type": "experience", "original_content": "Made services"},
    )
    assert resp.status_code == 200
    assert resp.json()["suggested_content"] == "Built scalable services."


def test_ai_test_provider_does_not_leak_exception_text():
    app.dependency_overrides[get_current_user_id] = mock_user
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(side_effect=RuntimeError("sensitive_internal_token_leak"))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    resp = client.post("/ai/test-provider")
    assert resp.status_code == 500
    assert "sensitive_internal_token_leak" not in resp.text


def test_rate_limiter_does_not_retain_empty_history_on_limit_error():
    limiter = ai_rate_limiter
    limiter.clear()
    limiter._history["user:old"] = [0.0]
    with patch("app.core.rate_limiter.settings.AI_RATE_LIMIT_PER_MINUTE", 0):
        with pytest.raises(Exception):
            limiter.check(user_id="old")
    assert "user:old" not in limiter.raw_history
