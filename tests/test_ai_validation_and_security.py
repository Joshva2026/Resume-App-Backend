import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from pydantic import ValidationError
import asyncio

from app.main import app
from app.core.config import settings
from app.core.rate_limiter import ai_rate_limiter
from app.api.deps import get_current_user_id, get_supabase_client
from app.api.routers.ai import get_ai_provider
from app.schemas.ai import (
    AIRequest,
    ResumePolishRequest,
    ChatMessageRequest,
    get_allowed_models
)

client = TestClient(app)

TEST_USER_A = "11111111-1111-1111-1111-111111111111"
TEST_USER_B = "22222222-2222-2222-2222-222222222222"

@pytest.fixture(autouse=True)
def reset_state():
    app.dependency_overrides.clear()
    ai_rate_limiter.clear()
    yield
    app.dependency_overrides.clear()
    ai_rate_limiter.clear()

def mock_user_a():
    return TEST_USER_A

def mock_user_b():
    return TEST_USER_B

def build_mock_supabase():
    mock_sb = MagicMock()
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": "conv-123"}])
    mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[{"id": "conv-123", "user_id": TEST_USER_A}]
    )
    mock_table.update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": "conv-123"}])
    mock_sb.table.return_value = mock_table
    return mock_sb

# =========================================================================
# 1. MODEL_ID VALIDATION TESTS
# =========================================================================

def test_missing_model_id_defaults_safely():
    # In AIRequest, model_id is Optional and defaults to None
    req = AIRequest(system_prompt="System instructions", user_prompt="User input")
    assert req.model_id is None

    # In ChatMessageRequest, model_id is Optional and defaults to None
    chat_req = ChatMessageRequest(message="Hello AI")
    assert chat_req.model_id is None

def test_valid_model_id_accepted():
    allowed = list(get_allowed_models())
    assert len(allowed) > 0
    valid_model = allowed[0]
    
    req = AIRequest(system_prompt="System", user_prompt="User", model_id=valid_model)
    assert req.model_id == valid_model

def test_invalid_model_id_rejected():
    with pytest.raises(ValidationError) as exc:
        AIRequest(system_prompt="System", user_prompt="User", model_id="malicious-model-4o")
    assert "Invalid model_id 'malicious-model-4o'" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        ChatMessageRequest(message="Hello", model_id="gpt-super-expensive-unauthorized")
    assert "Invalid model_id" in str(exc.value)

# =========================================================================
# 2. MAX_TOKENS VALIDATION TESTS
# =========================================================================

def test_excessive_max_tokens_rejected():
    # Upper ceiling is 4096 tokens
    with pytest.raises(ValidationError) as exc:
        AIRequest(system_prompt="System", user_prompt="User", max_tokens=100000)
    assert "Input should be less than or equal to 4096" in str(exc.value)

def test_sub_minimum_max_tokens_rejected():
    # Lower floor is 16 tokens
    with pytest.raises(ValidationError) as exc:
        AIRequest(system_prompt="System", user_prompt="User", max_tokens=5)
    assert "Input should be greater than or equal to 16" in str(exc.value)

# =========================================================================
# 3. PROMPT & RESUME INPUT VALIDATION TESTS
# =========================================================================

def test_oversized_prompts_rejected():
    # System prompt > 5000 characters
    with pytest.raises(ValidationError) as exc:
        AIRequest(system_prompt="A" * 5001, user_prompt="Valid user prompt")
    assert "String should have at most 5000 characters" in str(exc.value)

    # User prompt > 10000 characters
    with pytest.raises(ValidationError) as exc:
        AIRequest(system_prompt="Valid system", user_prompt="A" * 10001)
    assert "String should have at most 10000 characters" in str(exc.value)

def test_oversized_resume_content_rejected():
    # Resume polish request > 5000 characters
    with pytest.raises(ValidationError) as exc:
        ResumePolishRequest(
            section_type="experience",
            original_content="Long text " * 600 # > 5000 chars
        )
    assert "String should have at most 5000 characters" in str(exc.value)

def test_empty_and_whitespace_inputs_rejected():
    # Pure whitespace prompts in AIRequest
    with pytest.raises(ValidationError) as exc:
        AIRequest(system_prompt="   \n\t  ", user_prompt="Valid prompt")
    assert "Prompt cannot be empty or pure whitespace" in str(exc.value)

    # Pure whitespace in ResumePolishRequest
    with pytest.raises(ValidationError) as exc:
        ResumePolishRequest(section_type="   ", original_content="Valid content")
    assert "Field cannot be empty or pure whitespace" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        ResumePolishRequest(section_type="summary", original_content="   \n\t ")
    assert "Field cannot be empty or pure whitespace" in str(exc.value)

# =========================================================================
# 4. AUTHENTICATION & ACCESS CONTROL TESTS
# =========================================================================

def test_unauthenticated_requests_blocked():
    # 1. /ai/resume/polish
    res1 = client.post("/ai/resume/polish", json={
        "section_type": "summary",
        "original_content": "Experienced Python Engineer"
    })
    assert res1.status_code in [401, 403], f"Expected 401/403 but got {res1.status_code}"

    # 2. /ai/test
    res2 = client.post("/ai/test", json={
        "system_prompt": "System",
        "user_prompt": "User"
    })
    assert res2.status_code in [401, 403], f"Expected 401/403 but got {res2.status_code}"

    # 3. /ai/test-provider
    res3 = client.post("/ai/test-provider")
    assert res3.status_code in [401, 403], f"Expected 401/403 but got {res3.status_code}"

    # 4. /ai/chat
    res4 = client.post("/ai/chat", json={"message": "Hello"})
    assert res4.status_code in [401, 403], f"Expected 401/403 but got {res4.status_code}"

def test_authenticated_resume_polish_allowed():
    app.dependency_overrides[get_current_user_id] = mock_user_a
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value=MagicMock(
        content='{"suggested_content": "Improved Summary", "changes": ["Enhanced action verbs"]}',
        model=settings.NVIDIA_MODEL
    ))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    res = client.post("/ai/resume/polish", json={
        "section_type": "summary",
        "original_content": "Built websites with Python and Django."
    })
    assert res.status_code == 200
    data = res.json()
    assert data["suggested_content"] == "Improved Summary"
    assert "Enhanced action verbs" in data["changes"]

# =========================================================================
# 5. PER-USER RATE LIMITING & CONCURRENCY TESTS
# =========================================================================

def test_per_user_rate_limit_enforcement():
    app.dependency_overrides[get_current_user_id] = mock_user_a
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value=MagicMock(
        content='{"suggested_content": "Clean", "changes": []}',
        model=settings.NVIDIA_MODEL
    ))
    app.dependency_overrides[get_ai_provider] = lambda: mock_provider

    payload = {"section_type": "summary", "original_content": "Software developer"}

    # Exhaust User A's limit (10 requests/min default)
    limit = settings.AI_RATE_LIMIT_PER_MINUTE
    for _ in range(limit):
        res = client.post("/ai/resume/polish", json=payload)
        assert res.status_code == 200

    # Request limit + 1 from User A MUST return HTTP 429
    res_exceeded = client.post("/ai/resume/polish", json=payload)
    assert res_exceeded.status_code == 429
    assert "Rate limit exceeded" in res_exceeded.json()["detail"]

    # ISOLATION CHECK: User B must NOT be blocked by User A's rate limit exhaustion!
    app.dependency_overrides[get_current_user_id] = mock_user_b
    res_user_b = client.post("/ai/resume/polish", json=payload)
    assert res_user_b.status_code == 200

def test_concurrent_request_abuse_protection():
    # Test acquire and release concurrency guard for the same user
    user_id = "test-concurrent-user"
    
    # Run async concurrency check
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def run_concurrency_test():
        # First request acquires in-flight generation slot
        await ai_rate_limiter.acquire_concurrency(user_id=user_id)
        
        # Second simultaneous request from same user MUST raise HTTP 429
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await ai_rate_limiter.acquire_concurrency(user_id=user_id)
        assert exc.value.status_code == 429
        assert "already running" in exc.value.detail

        # After releasing slot, subsequent request succeeds
        await ai_rate_limiter.release_concurrency(user_id=user_id)
        await ai_rate_limiter.acquire_concurrency(user_id=user_id)
        await ai_rate_limiter.release_concurrency(user_id=user_id)

    loop.run_until_complete(run_concurrency_test())
    loop.close()

# =========================================================================
# 6. MALICIOUS REQUEST PAYLOADS
# =========================================================================

def test_malicious_payloads_rejected_or_bounded():
    # 1. Path traversal in model_id rejected
    with pytest.raises(ValidationError):
        ChatMessageRequest(message="Hello", model_id="../../../etc/passwd")

    # 2. SQL injection string in model_id rejected
    with pytest.raises(ValidationError):
        ChatMessageRequest(message="Hello", model_id="'; DROP TABLE resumes; --")

    # 3. Malicious bloated conversation history rejected
    bloated_history = [{"role": "user", "content": "msg"}] * 35 # > 30 turns
    with pytest.raises(ValidationError) as exc:
        AIRequest(system_prompt="System", user_prompt="User", history=bloated_history)
    assert "History cannot exceed 30 turns" in str(exc.value)

    # 4. Message turn containing invalid role injection
    with pytest.raises(ValidationError) as exc:
        AIRequest(system_prompt="System", user_prompt="User", history=[{"role": "admin_override", "content": "hack"}])
    assert "invalid role" in str(exc.value)
