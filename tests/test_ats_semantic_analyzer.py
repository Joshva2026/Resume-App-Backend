import pytest
import json
import asyncio
from app.services.ats_semantic_analyzer import AtsSemanticAnalyzer
from app.schemas.ai import AIResponse, AIRequest
from app.services.ai_provider import AIProvider, ProviderTimeoutError

class MockProvider(AIProvider):
    def __init__(self, content: str, finish_reason: str = None, raise_exception=None):
        self.mock_content = content
        self.mock_finish_reason = finish_reason
        self.raise_exception = raise_exception
        self.last_request = None

    async def generate(self, request: AIRequest) -> AIResponse:
        self.last_request = request
        if self.raise_exception:
            raise self.raise_exception
        return AIResponse(
            content=self.mock_content,
            provider="mock",
            model="mock",
            finish_reason=self.mock_finish_reason
        )

@pytest.mark.asyncio
async def test_valid_json_with_jd():
    # 1. Complete valid JSON with JD mode (and JD is present)
    mock_json = {
        "matched_keywords": ["Python", "SQL"],
        "missing_keywords": ["Docker", "AWS"],
        "strengths": ["Strong problem solving"],
        "recommendations": ["Learn Docker"]
    }
    provider = MockProvider(content=json.dumps(mock_json))
    analyzer = AtsSemanticAnalyzer(provider)
    result = await analyzer.analyze("resume text", "jd text")
    
    assert result["semantic_analysis_available"] is True
    assert result["matched_keywords"] == ["Python", "SQL"]
    assert result["missing_keywords"] == ["Docker", "AWS"]

@pytest.mark.asyncio
async def test_markdown_wrapped_json():
    # 2. Markdown wrapped JSON
    mock_json = {
        "matched_keywords": ["Python"],
        "missing_keywords": [],
        "strengths": [],
        "recommendations": []
    }
    content = f"```json\n{json.dumps(mock_json)}\n```"
    provider = MockProvider(content=content)
    analyzer = AtsSemanticAnalyzer(provider)
    result = await analyzer.analyze("resume text", "jd text")
    
    assert result["semantic_analysis_available"] is True
    assert result["matched_keywords"] == ["Python"]

@pytest.mark.asyncio
async def test_malformed_and_truncated_json():
    # 4, 5, 6. Malformed JSON / Truncated JSON / Unterminated string
    content = '{\n  "matched_keywords": [\n    "Python"' # Incomplete
    provider = MockProvider(content=content)
    analyzer = AtsSemanticAnalyzer(provider)
    result = await analyzer.analyze("resume text", "jd text")
    
    assert result["semantic_analysis_available"] is False
    assert result["semantic_error"] == "AI recommendations are temporarily unavailable. Your ATS score is still available."
    assert result["matched_keywords"] == []

@pytest.mark.asyncio
async def test_provider_truncation():
    # 10. Provider truncation (finish_reason)
    mock_json = {
        "matched_keywords": ["Python"],
        "missing_keywords": [],
        "strengths": [],
        "recommendations": []
    }
    provider = MockProvider(content=json.dumps(mock_json), finish_reason="length")
    analyzer = AtsSemanticAnalyzer(provider)
    result = await analyzer.analyze("resume text", "jd text")
    
    assert result["semantic_analysis_available"] is False
    assert result["semantic_error"] == "AI recommendations are temporarily unavailable. Your ATS score is still available."

@pytest.mark.asyncio
async def test_no_jd_mode():
    # 11. No-JD mode (empty JD means no matched/missing keywords generated)
    mock_json = {
        "matched_keywords": ["Python"], # The AI might hallucinate these
        "missing_keywords": ["Docker"], # The AI might hallucinate these
        "strengths": ["Strong problem solving"],
        "recommendations": ["Learn Docker"]
    }
    provider = MockProvider(content=json.dumps(mock_json))
    analyzer = AtsSemanticAnalyzer(provider)
    result = await analyzer.analyze("resume text", None)
    
    assert result["semantic_analysis_available"] is True
    assert result["matched_keywords"] == ["Python"] # Kept as detected skills
    assert result["missing_keywords"] == [] # Must be forced to empty
    assert result["strengths"] == ["Strong problem solving"]

@pytest.mark.asyncio
async def test_array_and_string_limits():
    # 13, 14, 15. Enforcing max array items and max string lengths
    mock_json = {
        "matched_keywords": [f"Kw{i}" for i in range(50)],
        "missing_keywords": [f"Miss{i}" for i in range(50)],
        "strengths": ["A" * 200, "B", "C", "D", "E"], # String too long, list too long
        "recommendations": ["X", "Y", "Z", "1", "2", "3", "4"] # List too long
    }
    provider = MockProvider(content=json.dumps(mock_json))
    analyzer = AtsSemanticAnalyzer(provider)
    result = await analyzer.analyze("resume text", "jd text")
    
    assert result["semantic_analysis_available"] is True
    assert len(result["matched_keywords"]) == 8
    assert len(result["missing_keywords"]) == 8
    assert len(result["strengths"]) == 3
    assert len(result["strengths"][0]) == 120 # Truncated string
    assert len(result["recommendations"]) == 5

@pytest.mark.asyncio
async def test_provider_timeout():
    # 8. Provider timeout
    provider = MockProvider(content="", raise_exception=ProviderTimeoutError("Timeout"))
    analyzer = AtsSemanticAnalyzer(provider)
    result = await analyzer.analyze("resume text", "jd text")
    
    assert result["semantic_analysis_available"] is False
    assert "temporarily unavailable" in result["semantic_error"]
