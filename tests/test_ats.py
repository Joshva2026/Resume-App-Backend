import pytest
from app.services.ats_scorer import AtsScorer
from app.services.ats_parser import AtsParser

def test_pii_does_not_affect_score():
    scorer = AtsScorer()
    
    # Same body, different PII
    resume_a = {
        "personal_info": ["John Doe", "john.doe@example.com", "555-0100"],
        "experience": "Developed Python backend using FastAPI.",
        "skills": "Python, SQL.",
        "education": "BS Computer Science."
    }
    
    resume_b = {
        "personal_info": ["Jane Smith", "jane.smith@another.com", "999-9999", "linkedin.com/in/jane"],
        "experience": "Developed Python backend using FastAPI.",
        "skills": "Python, SQL.",
        "education": "BS Computer Science."
    }
    
    text_a = scorer.sanitize_for_scoring(resume_a)
    text_b = scorer.sanitize_for_scoring(resume_b)
    
    score_a = scorer.calculate_score(text_a, target_role="Backend Developer")
    score_b = scorer.calculate_score(text_b, target_role="Backend Developer")
    
    # Assert deterministic equality regardless of PII
    assert score_a["overall"] == score_b["overall"]
    assert score_a["sections"] == score_b["sections"]

def test_docx_and_pdf_produce_normalized_structure():
    # Verify that parser normalization logic isolates PII identically when presented with a string.
    parser = AtsParser()
    
    raw_text = "John Doe\njohn@example.com\n\nExperience\nDeveloped app"
    structured = parser.structure_text(raw_text)
    
    assert "John Doe" in structured["personal_info"]
    assert "john@example.com" in structured["personal_info"]
    assert "Developed app" in structured["experience"]

def test_score_is_not_hardcoded_or_fabricated():
    scorer = AtsScorer()
    
    sparse_text = "Experience\nWorked at company."
    rich_text = """
    Experience
    Developed, managed, and led cloud microservices using Python, Go, and SQL.
    Optimized database performance by 45% and reduced latency by $2M annual cost savings.
    
    Education
    BS in Computer Science from University of Engineering.
    
    Skills
    Python, SQL, AWS, Docker, Kubernetes, React, Agile, Leadership.
    
    Projects
    Built high-scale portfolio analytics engine processing 10M daily events.
    
    Certifications
    AWS Certified Solutions Architect.
    """
    
    score_sparse = scorer.calculate_score(sparse_text, jd="Need Python developer", target_role="Backend Developer")
    score_rich = scorer.calculate_score(rich_text, jd="Need Python developer", target_role="Backend Developer")
    
    assert score_sparse["overall"] < score_rich["overall"]
    assert score_sparse["overall"] != 85
    assert score_sparse["overall"] != 90
    assert score_rich["overall"] > 70
    # Consistency
    assert score_rich["overall"] == scorer.calculate_score(rich_text, jd="Need Python developer", target_role="Backend Developer")["overall"]

def test_deterministic_semantics_no_hallucinations():
    scorer = AtsScorer()
    resume_text = "Experienced in Python and SQL with 5 years building REST APIs."
    jd_text = "Seeking engineer with Python, SQL, Docker, and Kubernetes expertise."
    
    result = scorer.calculate_deterministic_semantics(resume_text, jd_text, target_role="Backend Developer")
    
    assert result["semantic_analysis_available"] is False
    assert result["analysis_mode"] == "deterministic_fallback"
    # Python and SQL exist in both
    assert "Python" in result["matched_keywords"]
    assert "SQL" in result["matched_keywords"]
    # Docker and Kubernetes are in JD but missing in resume
    assert "Docker" in result["missing_keywords"]
    assert "Kubernetes" in result["missing_keywords"]
    # No unmentioned skill like "React" or "Java" should be present in matched or missing
    assert "React" not in result["matched_keywords"]
    assert "React" not in result["missing_keywords"]
    # Recommendations must contain priority missing keywords
    assert any("Docker" in r or "Kubernetes" in r for r in result["recommendations"])

def test_section_detection_with_parsed_structure():
    scorer = AtsScorer()
    parsed_structure = {
        "experience": "Software Engineer at Acme. Built cloud backend.",
        "education": "MIT, CS",
        "skills": "Python, Kotlin, Docker",
        "projects": "Open source mobile app",
        "certifications": "AWS Solutions Architect"
    }
    sanitized = scorer.sanitize_for_scoring(parsed_structure)
    score_res = scorer.calculate_score(sanitized, parsed_structure=parsed_structure)
    
    # All 5 sections should be detected
    assert score_res["sections"]["Completeness"] >= 95
    assert score_res["sections"]["Projects"] == 100
    assert score_res["sections"]["Education"] == 100
    assert score_res["sections"]["Certifications"] == 100

def test_java_vs_javascript_no_collision():
    scorer = AtsScorer()
    
    # Resume with Java only
    java_resume = "Experienced in Java, Spring Boot, and SQL."
    java_result = scorer.calculate_deterministic_semantics(java_resume, jd="Looking for JavaScript and React")
    assert "Java" not in java_result["matched_keywords"]
    assert "JavaScript" not in java_result["matched_keywords"]
    assert "JavaScript" in java_result["missing_keywords"]
    
    # Resume with JavaScript only
    js_resume = "Experienced in JavaScript, React, and Node.js."
    js_result = scorer.calculate_deterministic_semantics(js_resume, jd="Looking for Java and Spring Boot")
    assert "Java" not in js_result["matched_keywords"]
    assert "Java" in js_result["missing_keywords"]

def test_action_verb_led_does_not_match_handled_or_compiled():
    scorer = AtsScorer()
    text = "Handled customer inquiries, scheduled calendar appointments, and compiled quarterly summaries."
    
    # None of the action verbs should match
    res = scorer.calculate_score(text)
    # Experience score should not be boosted by false "led" match
    assert res["sections"]["Experience"] == 0

def test_dates_and_phone_numbers_do_not_inflate_metrics():
    scorer = AtsScorer()
    non_metric_text = "Software Engineer at Acme Corp from 2018 to 2023. Phone: 555-0123. Emergency: 911-0000. Version 3.2.1."
    metrics = scorer._extract_achievement_metrics(non_metric_text)
    assert len(metrics) == 0
    
    real_metric_text = "Optimized query latency by 45%, processing 50M records daily, resulting in $2M annual cost savings."
    real_metrics = scorer._extract_achievement_metrics(real_metric_text)
    assert len(real_metrics) >= 3

def test_stopwords_do_not_appear_as_keywords():
    scorer = AtsScorer()
    jd = "Seeking a candidate with strong background in Python and Docker to join our high throughput team."
    resume = "Experienced developer with Python. Looking for a new opportunity."
    
    semantics = scorer.calculate_deterministic_semantics(resume, jd=jd)
    matched = [k.lower() for k in semantics["matched_keywords"]]
    missing = [k.lower() for k in semantics["missing_keywords"]]
    
    stopwords = ["and", "seeking", "high", "throughput", "candidate", "strong", "team", "with"]
    for sw in stopwords:
        assert sw not in matched
        assert sw not in missing

def test_strong_vs_unrelated_jd_match():
    scorer = AtsScorer()
    jd = "Seeking a Senior Python Developer with FastAPI, Docker, Kubernetes, AWS, and PostgreSQL experience."
    
    strong_resume = """
    Experience
    Senior Python Developer. Built distributed microservices using Python and FastAPI.
    Containerized applications with Docker and managed Kubernetes clusters on AWS.
    Optimized database performance with PostgreSQL.
    """
    
    unrelated_resume = """
    Experience
    Warehouse Clerk. Handled customer calls, scheduled appointments, and verified inventory records.
    """
    
    score_strong = scorer.calculate_score(strong_resume, jd=jd, target_role="Senior Python Developer")
    score_unrelated = scorer.calculate_score(unrelated_resume, jd=jd, target_role="Senior Python Developer")
    
    assert score_strong["sections"]["JD Match"] > 70
    assert score_unrelated["sections"]["JD Match"] < 25
    assert score_strong["overall"] > score_unrelated["overall"]

def test_missing_jd_behavior():
    scorer = AtsScorer()
    resume = "Developed scalable web services using Python, SQL, and Docker. B.S. in Computer Science."
    
    res = scorer.calculate_score(resume, jd=None, target_role=None)
    assert res["sections"]["JD Match"] is None
    assert res["sections"]["Target Role"] is None
    assert 0 <= res["overall"] <= 100

