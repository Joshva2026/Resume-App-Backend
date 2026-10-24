import pytest
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
