from typing import Dict, Any, List
import re

class AtsScorer:
    def sanitize_for_scoring(self, parsed_structure: Dict[str, Any]) -> str:
        """
        Extracts only the body text for scoring, explicitly dropping 'personal_info'.
        """
        return parsed_structure.get("body", "")

    def calculate_score(self, text: str, jd: str = None, target_role: str = None) -> Dict[str, Any]:
        """
        Deterministic 100-point 9-factor scoring model.
        """
        text_lower = text.lower()
        score = 0
        section_scores = {}
        
        # 1. Section Completeness / Presence (15%)
        core_sections = ["experience", "education", "skills"]
        optional_sections = ["projects", "certifications", "achievements", "summary", "objective"]
        found_core = sum(1 for s in core_sections if s in text_lower)
        found_opt = sum(1 for s in optional_sections if s in text_lower)
        
        completeness = min(100, (found_core * 25) + (found_opt * 10))
        section_scores["Completeness"] = completeness
        score += completeness * 0.15
        
        # 2. Experience / Responsibilities (20%) - Action Verbs & Metrics
        action_verbs = ["developed", "managed", "led", "created", "designed", "implemented", "built", "optimized", "increased", "reduced"]
        verb_count = sum(1 for v in action_verbs if v in text_lower)
        metrics = len(re.findall(r"\d+%", text)) + len(re.findall(r"\$?\d+[MK]?", text))
        
        exp_score = min(100, (verb_count * 10) + (metrics * 15))
        section_scores["Experience"] = exp_score
        score += exp_score * 0.20
        
        # 3. Relevant Skills & Technologies (15%)
        # Look for a density of common technical/soft skills
        skills = ["python", "java", "agile", "sql", "aws", "react", "communication", "leadership", "management", "data", "analysis", "cloud"]
        skill_count = sum(1 for s in skills if s in text_lower)
        skill_score = min(100, skill_count * 15)
        section_scores["Skills"] = skill_score
        score += skill_score * 0.15
        
        # 4. Projects (10%)
        project_score = 100 if "projects" in text_lower or "portfolio" in text_lower else 0
        section_scores["Projects"] = project_score
        score += project_score * 0.10
        
        # 5. Education / Qualifications (10%)
        edu_score = 100 if "education" in text_lower or "university" in text_lower or "college" in text_lower or "bachelor" in text_lower else 0
        section_scores["Education"] = edu_score
        score += edu_score * 0.10
        
        # 6. Certifications / Achievements (5%)
        cert_score = 100 if "certifications" in text_lower or "achievements" in text_lower or "awards" in text_lower else 0
        section_scores["Certifications"] = cert_score
        score += cert_score * 0.05
        
        # 7. Target Role Alignment (10%)
        if target_role:
            role_words = set(re.findall(r"\w+", target_role.lower()))
            resume_words = set(re.findall(r"\w+", text_lower))
            overlap = role_words.intersection(resume_words)
            role_score = min(100, int((len(overlap) / max(1, len(role_words))) * 100))
            section_scores["Target Role"] = role_score
            score += role_score * 0.10
        else:
            section_scores["Target Role"] = "Not Performed"
        
        # 8. JD Semantic Match (10%)
        if jd:
            jd_words = set(re.findall(r"\w+", jd.lower()))
            resume_words = set(re.findall(r"\w+", text_lower))
            overlap = jd_words.intersection(resume_words)
            jd_score = min(100, int((len(overlap) / max(1, len(jd_words))) * 100 * 1.5)) # 1.5x multiplier for realistic density
            section_scores["JD Match"] = jd_score
            score += jd_score * 0.10
        else:
            section_scores["JD Match"] = "Not Performed"
            
        # 9. ATS Readability / Structure (5%)
        # Checks if length is reasonable (not too short, not a giant block of unbroken text)
        lines = text.split("\n")
        non_empty_lines = len([l for l in lines if l.strip()])
        readability = 100 if 20 < non_empty_lines < 200 else 50
        section_scores["Readability"] = readability
        score += readability * 0.05
        
        # Redistribute missing components
        used_weight = 1.0
        if not target_role:
            used_weight -= 0.10
        if not jd:
            used_weight -= 0.10
            
        final_score = score * (1.0 / used_weight)
        
        return {
            "overall": int(final_score),
            "sections": section_scores
        }
