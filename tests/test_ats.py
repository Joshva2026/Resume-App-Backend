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
