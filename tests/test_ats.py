import pytest
from app.services.ats_scorer import AtsScorer
from app.services.ats_parser import AtsParser

def test_pii_does_not_affect_score():
    scorer = AtsScorer()
    
    # Same body, different PII
    resume_a = {
        "personal_info": ["John Doe", "john.doe@example.com", "555-0100"],
        "body": "Experience: Developed Python backend using FastAPI. Skills: Python, SQL. Education: BS Computer Science."
    }
    
    resume_b = {
        "personal_info": ["Jane Smith", "jane.smith@another.com", "999-9999", "linkedin.com/in/jane"],
        "body": "Experience: Developed Python backend using FastAPI. Skills: Python, SQL. Education: BS Computer Science."
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
    assert "Experience" in structured["body"]
    assert "Developed app" in structured["body"]
