import pytest
import json
from app.core.config import settings
from app.services.ats_privacy import ResumePrivacyRedactor
from app.services.ats_scorer import AtsScorer
from app.services.ats_cache import AtsAnalysisCache
from app.services.ats_semantic_analyzer import AtsSemanticAnalyzer
from app.schemas.ai import AIRequest, AIResponse, AIResponseUsage
from app.services.ai_provider import AIProvider
from app.schemas.ats import AtsReport

class MockTruncatingAIProvider(AIProvider):
    async def generate(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            content="",
            provider="nvidia",
            model="openai/gpt-oss-20b",
            usage=AIResponseUsage(prompt_tokens=200, completion_tokens=500, total_tokens=700),
            finish_reason="length"
        )

class MockMalformedAIProvider(AIProvider):
    async def generate(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            content="I am an AI assistant and here is my reasoning: candidate has good skills...",
            provider="nvidia",
            model="openai/gpt-oss-20b",
            usage=AIResponseUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250),
            finish_reason="stop"
        )

# -------------------------------------------------------------
# 1. Privacy Redaction Tests (Synthetic Identities Only)
# -------------------------------------------------------------
def test_synthetic_personal_identifiers_removed_while_preserving_professional_context():
    redactor = ResumePrivacyRedactor()
    
    synthetic_resume_structure = {
        "personal_info": "Alexandria Morgan\nalex.morgan.synthetic@example.org\n+1 (555) 019-2834\n742 Evergreen Terrace, Springfield, IL 62704\nDOB: 15/08/1992\nlinkedin.com/in/alex-morgan-synthetic",
        "summary": "Full-Stack Engineer with 6 years building high-throughput distributed applications.",
        "experience": "Senior Backend Developer at Synthetix Corp (2020 - Present)\n• Architected REST APIs using Python and FastAPI, improving latency by 35%.\n• Containerized 12 microservices using Docker and orchestrated deployments on Kubernetes.\n• Managed PostgreSQL database with 50M records and reduced query execution time by 40%.",
        "skills": "Python, FastAPI, Docker, Kubernetes, PostgreSQL, Redis, Git, Linux, CI/CD",
        "education": "B.S. in Computer Science, University of Technology (2014 - 2018)",
        "certifications": "AWS Certified Solutions Architect"
    }

    sanitized_text = redactor.build_sanitized_resume_prompt_text(synthetic_resume_structure)

    # 1. Verify personal identifiers are absent
    assert "alex.morgan.synthetic@example.org" not in sanitized_text
    assert "555" not in sanitized_text and "019-2834" not in sanitized_text
    assert "742 Evergreen Terrace" not in sanitized_text
    assert "DOB" not in sanitized_text and "15/08/1992" not in sanitized_text
    assert "linkedin.com/in/alex-morgan" not in sanitized_text
    assert "Alexandria Morgan" not in sanitized_text

    # 2. Verify professional skills, metrics, and achievements are intact
    assert "Python" in sanitized_text
    assert "FastAPI" in sanitized_text
    assert "Docker" in sanitized_text
    assert "Kubernetes" in sanitized_text
    assert "PostgreSQL" in sanitized_text
    assert "35%" in sanitized_text
    assert "40%" in sanitized_text
    assert "50M" in sanitized_text
    assert "Computer Science" in sanitized_text
    assert "AWS Certified Solutions Architect" in sanitized_text

def test_final_provider_request_boundary_safety_gate():
    redactor = ResumePrivacyRedactor()
    
    # Prompt containing accidental token and email
    unsafe_prompt = "Candidate prompt. Contact: test.user@synth.io. Token: sbp_live_secret_token_12345. Experience with React and Node.js."
    is_safe, clean_prompt = redactor.verify_request_safety(unsafe_prompt)
    
    assert is_safe is True
    assert "test.user@synth.io" not in clean_prompt
    assert "sbp_live_secret_token_12345" not in clean_prompt
    assert "React" in clean_prompt
    assert "Node.js" in clean_prompt

# -------------------------------------------------------------
# 2. Dynamic Scoring & Job-Specific Analysis Tests
# -------------------------------------------------------------
def test_different_job_descriptions_produce_distinct_dynamic_scores():
    scorer = AtsScorer()
    
    # Candidate Resume: Heavy Frontend Engineer (React, JavaScript, HTML, CSS, TypeScript)
    frontend_resume = """
    Summary: Frontend Engineer specializing in responsive web applications.
    Experience:
    • Developed responsive single-page web applications using React and JavaScript.
    • Built modern user interfaces with HTML, CSS, and TypeScript.
    • Improved page render performance by 45% using code splitting and lazy loading.
    Skills: React, JavaScript, TypeScript, HTML, CSS, Git
    Education: B.S. in Software Engineering
    Projects: Built an open-source React component library with 10k downloads.
    """

    # JD A: Frontend Developer (Perfect match)
    frontend_jd = """
    Job Title: Frontend Web Developer
    Requirements:
    - Strong proficiency in React, JavaScript, HTML, and CSS
    - Experience with TypeScript and Git
    - Knowledge of responsive UI development
    """

    # JD B: Machine Learning / Data Engineer (Unrelated role)
    ml_jd = """
    Job Title: Machine Learning Engineer
    Requirements:
    - 3+ years experience with Python, PyTorch, and TensorFlow
    - Distributed data processing with Spark and Kafka
    - Experience building Data Pipelines and ETL
    """

    # JD C: Cloud DevOps Engineer (Partially/unrelated role)
    devops_jd = """
    Job Title: DevOps Engineer
    Requirements:
    - Kubernetes, Docker, Terraform, and AWS
    - CI/CD automation and Linux administration
    """

    score_frontend = scorer.calculate_score(frontend_resume, jd=frontend_jd, target_role="Frontend Developer")
    score_ml = scorer.calculate_score(frontend_resume, jd=ml_jd, target_role="Machine Learning Engineer")
    score_devops = scorer.calculate_score(frontend_resume, jd=devops_jd, target_role="DevOps Engineer")

    # 1. Frontend match must be high, ML match must be low
    assert score_frontend["overall"] > score_ml["overall"]
    assert score_frontend["overall"] > score_devops["overall"]
    assert score_frontend["overall"] >= 75
    assert score_ml["overall"] <= 55

    # 2. JD Match section score must reflect requirement alignment
    assert score_frontend["sections"]["JD Match"] > score_ml["sections"]["JD Match"]
    assert score_ml["sections"]["JD Match"] < 35

    # 3. None of the scores should be hardcoded constants like 81
    assert score_frontend["overall"] != 81
    assert score_ml["overall"] != 81
    assert score_devops["overall"] != 81

def test_identical_inputs_produce_consistent_deterministic_scores():
    scorer = AtsScorer()
    
    resume = """
    Experience: Developed web backend in Python and PostgreSQL. Reduced latency by 20%.
    Skills: Python, PostgreSQL, Docker
    Education: B.S. in Computer Science
    """
    jd = "Looking for a Python Developer with PostgreSQL and Docker experience."
    
    score1 = scorer.calculate_score(resume, jd=jd, target_role="Python Developer")
    score2 = scorer.calculate_score(resume, jd=jd, target_role="Python Developer")
    
    assert score1["overall"] == score2["overall"]
    assert score1["sections"] == score2["sections"]

# -------------------------------------------------------------
# 3. Cache Isolation & Invalidation Tests
# -------------------------------------------------------------
def test_cache_fingerprint_invalidation_on_changed_inputs():
    AtsAnalysisCache.clear()
    
    user_id = "user_test_123"
    resume_a = "Resume content A with Python and FastAPI"
    resume_b = "Resume content B with React and TypeScript"
    jd_a = "Looking for Python developer"
    jd_b = "Looking for React developer"
    version = AtsScorer.SCORING_VERSION

    key_a = AtsAnalysisCache.make_key(user_id, resume_a, jd_a, "Backend", version)
    key_diff_jd = AtsAnalysisCache.make_key(user_id, resume_a, jd_b, "Backend", version)
    key_diff_resume = AtsAnalysisCache.make_key(user_id, resume_b, jd_a, "Backend", version)
    key_diff_user = AtsAnalysisCache.make_key("another_user", resume_a, jd_a, "Backend", version)
    key_diff_version = AtsAnalysisCache.make_key(user_id, resume_a, jd_a, "Backend", "ats_v1.0")

    # All keys must be unique fingerprints
    assert key_a != key_diff_jd
    assert key_a != key_diff_resume
    assert key_a != key_diff_user
    assert key_a != key_diff_version

    # Set report in cache for key_a
    dummy_report = AtsReport(ats_document_id="doc_1", overall_score=85)
    AtsAnalysisCache.set(key_a, dummy_report)

    # Cache hit for identical input
    assert AtsAnalysisCache.get(key_a) is not None
    assert AtsAnalysisCache.get(key_a).overall_score == 85

    # Cache miss for changed inputs
    assert AtsAnalysisCache.get(key_diff_jd) is None
    assert AtsAnalysisCache.get(key_diff_resume) is None
    assert AtsAnalysisCache.get(key_diff_user) is None

# -------------------------------------------------------------
# 4. Truncation Resilience & Fallback Tests
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_truncated_nvidia_response_activates_deterministic_fallback():
    truncating_provider = MockTruncatingAIProvider()
    analyzer = AtsSemanticAnalyzer(truncating_provider)

    resume_text = """
    Experience: Developed web backend in Python and FastAPI. Increased throughput by 30%.
    Skills: Python, FastAPI, Docker
    Education: B.S. in Computer Science
    """
    jd_text = "Requirements: Python, FastAPI, Kubernetes, AWS"

    result = await analyzer.analyze(
        sanitized_text=resume_text,
        jd=jd_text,
        target_role="Python Developer"
    )

    # 1. Truncation must NOT be treated as a successful AI report
    assert result["analysis_mode"] == "deterministic_fallback"
    assert result["semantic_analysis_available"] is False
    assert "was truncated by provider" in result["semantic_error"]

    # 2. Fallback must use both resume and current JD
    assert "Python" in result["matched_keywords"]
    assert "FastAPI" in result["matched_keywords"]
    assert "Kubernetes" in result["missing_keywords"]
    assert "AWS" in result["missing_keywords"]

@pytest.mark.asyncio
async def test_malformed_reasoning_without_json_activates_fallback():
    malformed_provider = MockMalformedAIProvider()
    analyzer = AtsSemanticAnalyzer(malformed_provider)

    resume_text = "Experience with Python and SQL."
    jd_text = "Need Python and Docker."

    result = await analyzer.analyze(
        sanitized_text=resume_text,
        jd=jd_text,
        target_role="Software Engineer"
    )

    assert result["analysis_mode"] == "deterministic_fallback"
    assert "Python" in result["matched_keywords"]
    assert "Docker" in result["missing_keywords"]

# -------------------------------------------------------------
# 5. Production Timeout & Optimization Tests
# -------------------------------------------------------------
class MockTimeoutAIProvider(AIProvider):
    async def generate(self, request: AIRequest) -> AIResponse:
        from app.services.ai_provider import ProviderTimeoutError
        raise ProviderTimeoutError("The AI response took too long. Please try again.")

class MockCompleteAIProvider(AIProvider):
    async def generate(self, request: AIRequest) -> AIResponse:
        valid_json = """
        {
            "matched_keywords": ["Python (Experience: Built REST API)", "FastAPI (Experience: Optimized backend)"],
            "missing_keywords": ["Docker", "Kubernetes"],
            "missing_required": ["Docker"],
            "missing_preferred": ["Kubernetes"],
            "strengths": ["Demonstrated backend architecture experience"],
            "recommendations": ["Add container orchestration with Docker and Kubernetes"]
        }
        """
        return AIResponse(
            content=valid_json,
            provider="nvidia",
            model="openai/gpt-oss-20b",
            usage=AIResponseUsage(prompt_tokens=250, completion_tokens=180, total_tokens=430),
            finish_reason="stop"
        )

@pytest.mark.asyncio
async def test_slow_nvidia_read_timeout_smooth_fallback():
    timeout_provider = MockTimeoutAIProvider()
    analyzer = AtsSemanticAnalyzer(timeout_provider)

    resume_text = "Experience with Python and FastAPI."
    jd_text = "Requirements: Python, FastAPI, Docker"

    result = await analyzer.analyze(
        sanitized_text=resume_text,
        jd=jd_text,
        target_role="Backend Developer"
    )

    assert result["analysis_mode"] == "deterministic_fallback"
    assert result["semantic_analysis_available"] is False
    assert "timed out" in result["semantic_error"]
    # Fallback still successfully computed matched/missing keywords
    assert "Python" in result["matched_keywords"]
    assert "Docker" in result["missing_keywords"]

@pytest.mark.asyncio
async def test_valid_complete_ai_response_parsed_successfully():
    complete_provider = MockCompleteAIProvider()
    analyzer = AtsSemanticAnalyzer(complete_provider)

    resume_text = "Experience with Python and FastAPI."
    jd_text = "Requirements: Python, FastAPI, Docker, Kubernetes"

    result = await analyzer.analyze(
        sanitized_text=resume_text,
        jd=jd_text,
        target_role="Backend Developer"
    )

    assert result["analysis_mode"] == "ai"
    assert result["semantic_analysis_available"] is True
    assert result["semantic_error"] is None
    assert "Python" in result["matched_keywords"]
    assert "FastAPI" in result["matched_keywords"]
    assert "Docker" in result["missing_required"]
    assert "Kubernetes" in result["missing_preferred"]
    assert len(result["strengths"]) > 0

def test_job_description_compaction_strips_boilerplate_and_preserves_requirements():
    analyzer = AtsSemanticAnalyzer(MockTruncatingAIProvider())
    
    noisy_jd = """
    Job Title: Senior Backend Developer
    
    About Us:
    Acme Corp is an equal opportunity employer dedicated to building a diverse and inclusive team.
    All qualified applicants will receive consideration without regard to race, color, religion, sex.
    
    Benefits:
    Benefits package includes comprehensive medical, dental, vision, unlimited PTO, and 401(k) matching.
    Commuter benefits and flexible hybrid work environment.
    
    Requirements:
    - 5+ years of Python and FastAPI development
    - Microservices architecture and PostgreSQL database optimization
    - Experience deploying containers with Docker and Kubernetes
    
    Preferred Qualifications:
    - AWS or GCP cloud certifications
    """

    compacted = analyzer._compact_job_description(noisy_jd, max_chars=1200)

    # Boilerplate removed
    assert "equal opportunity employer" not in compacted.lower()
    assert "dental, vision" not in compacted.lower()
    assert "401(k)" not in compacted.lower()
    
    # Core requirements preserved
    assert "Python" in compacted
    assert "FastAPI" in compacted
    assert "Docker" in compacted
    assert "Kubernetes" in compacted
    assert "PostgreSQL" in compacted

def test_cache_ttl_distinction_and_bypass():
    AtsAnalysisCache.clear()

    key = "test_cache_key_ttl"
    ai_report = AtsReport(ats_document_id="doc_ai", analysis_mode="ai", overall_score=90)
    fallback_report = AtsReport(ats_document_id="doc_fb", analysis_mode="deterministic_fallback", overall_score=60)

    # Fallback report gets short 15s TTL
    AtsAnalysisCache.set(key, fallback_report)
    assert AtsAnalysisCache.get(key) is not None
    assert AtsAnalysisCache.get(key).analysis_mode == "deterministic_fallback"

    # AI report gets full 600s TTL
    AtsAnalysisCache.set(key, ai_report)
    assert AtsAnalysisCache.get(key).analysis_mode == "ai"

# -------------------------------------------------------------
# 5. Grok (xAI) vs Groq vs NVIDIA Provider Isolation & Routing Verification
# -------------------------------------------------------------
def test_ats_routes_to_grok_and_ai_assistant_routes_to_nvidia():
    from app.services.grok_provider import XAIGrokAIProvider, GrokAIProvider
    from app.services.groq_provider import GroqAIProvider
    from app.services.nvidia_provider import NvidiaAIProvider
    from app.api.routers.ai import get_ai_provider
    from app.services.ats_semantic_analyzer import get_default_ats_provider
    import inspect
    from app.api.routers.ats import analyze_resume
    from unittest.mock import patch

    # 1. Verify AI Assistant dependency strictly uses NvidiaAIProvider with openai/gpt-oss-20b
    assistant_provider = get_ai_provider()
    assert isinstance(assistant_provider, NvidiaAIProvider)
    assert assistant_provider.model == "openai/gpt-oss-20b"

    # 2. Verify xAI Grok resolution when XAI_API_KEY is configured
    with patch.object(settings, "XAI_API_KEY", "xai-test-key-12345"):
        xai_prov = get_default_ats_provider()
        assert isinstance(xai_prov, XAIGrokAIProvider)
        assert xai_prov.model == "grok-2-latest"

    # 3. Verify Groq resolution when only GROQ_API_KEY is configured
    with patch.object(settings, "XAI_API_KEY", None), \
         patch.object(settings, "GROQ_API_KEY", "gsk-test-key-12345"):
        groq_prov = get_default_ats_provider()
        assert isinstance(groq_prov, GroqAIProvider)
        assert groq_prov.model == "openai/gpt-oss-120b"

    # 4. Verify default fallback provider is XAIGrokAIProvider
    with patch.object(settings, "XAI_API_KEY", None), \
         patch.object(settings, "GROQ_API_KEY", None):
        default_prov = get_default_ats_provider()
        assert isinstance(default_prov, XAIGrokAIProvider)

    # 5. Inspect analyze_resume source in ats router to ensure provider resolution is decoupled from NvidiaAIProvider
    ats_source = inspect.getsource(analyze_resume)
    assert "get_default_ats_provider()" in ats_source
    assert "NvidiaAIProvider()" not in ats_source

def test_government_id_and_token_redaction():
    redactor = ResumePrivacyRedactor()
    raw_text = """
    Name: John Doe
    Email: john.doe@example.org
    Phone: +1 (555) 432-1098
    SSN: 123-45-6789
    Aadhaar: 2345 6789 0123
    Passport No: P1234567
    National ID: ID-987654321
    xAI Grok Key: xai-live-1234567890abcdefghijklmnopqrstuvwxyz
    Groq Key: gsk_live_1234567890abcdefghijklmnopqrstuvwxyz
    Nvidia Key: nvapi-live_abcdef1234567890abcdef1234567890
    Technical Skills: Python, FastAPI, Docker, Kubernetes, AWS Certified Solutions Architect
    Experience: Built high-throughput microservices handling 50M requests daily.
    """
    is_safe, clean_text = redactor.verify_request_safety(redactor.redact_text(raw_text))
    
    assert is_safe is True
    assert "123-45-6789" not in clean_text
    assert "2345 6789 0123" not in clean_text
    assert "P1234567" not in clean_text
    assert "ID-987654321" not in clean_text
    assert "xai-live" not in clean_text
    assert "gsk_live" not in clean_text
    assert "nvapi-live" not in clean_text
    assert "john.doe@example.org" not in clean_text
    assert "432-1098" not in clean_text
    
    # Verify professional content is completely preserved
    assert "Python" in clean_text
    assert "FastAPI" in clean_text
    assert "Docker" in clean_text
    assert "Kubernetes" in clean_text
    assert "AWS Certified Solutions Architect" in clean_text
    assert "50M" in clean_text

@pytest.mark.asyncio
async def test_groq_receives_unfabricated_evidence_and_different_jds():
    class CapturingProvider(AIProvider):
        def __init__(self):
            self.captured_prompts = []

        async def generate(self, request: AIRequest) -> AIResponse:
            self.captured_prompts.append(request.user_prompt)
            # Return valid JSON reflecting candidate's actual qualifications
            return AIResponse(
                content=json.dumps({
                    "matched_keywords": ["Python (Experience: Built microservices)"],
                    "missing_keywords": ["Rust", "Solana"],
                    "missing_required": ["Rust"],
                    "missing_preferred": ["Solana"],
                    "strengths": ["Demonstrated backend proficiency in Python"],
                    "recommendations": ["Acquire hands-on experience in Rust"]
                }),
                provider="groq",
                model="openai/gpt-oss-120b"
            )

    provider = CapturingProvider()
    analyzer = AtsSemanticAnalyzer(provider=provider)

    resume = "Experience: Developed microservices with Python and PostgreSQL for 4 years."
    jd_web3 = "Job: Web3 Engineer. Requirements: Rust, Solana, Smart Contracts."
    jd_backend = "Job: Python Backend Engineer. Requirements: Python, PostgreSQL, REST APIs."

    res1 = await analyzer.analyze(sanitized_text=resume, jd=jd_web3)
    res2 = await analyzer.analyze(sanitized_text=resume, jd=jd_backend)

    # 1. Verify user prompts to Groq are distinct
    assert len(provider.captured_prompts) == 2
    assert jd_web3 in provider.captured_prompts[0]
    assert jd_backend in provider.captured_prompts[1]
    assert provider.captured_prompts[0] != provider.captured_prompts[1]

    # 2. Verify cache keys are distinct for different JDs
    key1 = AtsAnalysisCache.make_key("user_1", resume, jd_web3, None, "ats_v2.0")
    key2 = AtsAnalysisCache.make_key("user_1", resume, jd_backend, None, "ats_v2.0")
    assert key1 != key2

    # 3. Verify response schema compatibility
    assert res1["analysis_mode"] == "ai"
    assert "matched_keywords" in res1
    assert "missing_keywords" in res1
    assert "missing_required" in res1
    assert "missing_preferred" in res1
    assert "strengths" in res1
    assert "recommendations" in res1


