from typing import Dict, Any, List
import re

class AtsScorer:
    def sanitize_for_scoring(self, parsed_structure: Dict[str, Any]) -> str:
        """
        Extracts only the body text for scoring, explicitly dropping 'personal_info'.
        """
        # Collect all parts except personal_info
        body_parts = []
        for key, value in parsed_structure.items():
            if key != "personal_info" and isinstance(value, str):
                body_parts.append(value)
        
        return "\n".join(body_parts)

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
            section_scores["Target Role"] = None
        
        # 8. JD Semantic Match (10%)
        if jd:
            jd_words = set(re.findall(r"\w+", jd.lower()))
            resume_words = set(re.findall(r"\w+", text_lower))
            overlap = jd_words.intersection(resume_words)
            jd_score = min(100, int((len(overlap) / max(1, len(jd_words))) * 100 * 1.5)) # 1.5x multiplier for realistic density
            section_scores["JD Match"] = jd_score
            score += jd_score * 0.10
        else:
            section_scores["JD Match"] = None
            
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

    def calculate_deterministic_semantics(
        self,
        text: str,
        jd: str = None,
        target_role: str = None,
        section_scores: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extracts verified matched/missing keywords, strengths, and actionable
        recommendations deterministically from resume text and job description/role.
        Never fabricates skills or credentials.
        """
        text_lower = text.lower()
        if section_scores is None:
            section_scores = self.calculate_score(text, jd, target_role)["sections"]

        # Curated technical and professional skills dictionary with proper casing
        SKILL_MAP = {
            "python": "Python", "java": "Java", "javascript": "JavaScript", "typescript": "TypeScript",
            "c++": "C++", "c#": "C#", "go": "Go", "golang": "Go", "rust": "Rust", "ruby": "Ruby",
            "php": "PHP", "swift": "Swift", "kotlin": "Kotlin", "sql": "SQL", "html": "HTML",
            "css": "CSS", "bash": "Bash", "react": "React", "angular": "Angular", "vue": "Vue.js",
            "next.js": "Next.js", "node.js": "Node.js", "nodejs": "Node.js", "express": "Express",
            "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "spring": "Spring",
            "spring boot": "Spring Boot", ".net": ".NET", "docker": "Docker", "kubernetes": "Kubernetes",
            "k8s": "Kubernetes", "terraform": "Terraform", "ci/cd": "CI/CD", "aws": "AWS",
            "gcp": "GCP", "azure": "Azure", "git": "Git", "linux": "Linux", "postgresql": "PostgreSQL",
            "postgres": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB", "redis": "Redis",
            "elasticsearch": "Elasticsearch", "graphql": "GraphQL", "rest api": "REST API",
            "microservices": "Microservices", "agile": "Agile", "scrum": "Scrum", "tdd": "TDD",
            "machine learning": "Machine Learning", "deep learning": "Deep Learning",
            "data analysis": "Data Analysis", "data pipeline": "Data Pipeline", "etl": "ETL",
            "system design": "System Design", "unit testing": "Unit Testing", "devops": "DevOps",
            "android": "Android", "jetpack compose": "Jetpack Compose", "ios": "iOS",
            "flutter": "Flutter", "project management": "Project Management", "leadership": "Leadership"
        }

        matched_keywords: List[str] = []
        missing_keywords: List[str] = []
        matched_seen = set()
        missing_seen = set()

        if jd and jd.strip():
            jd_lower = jd.lower()
            # 1. Check known skills against JD
            for skill_key, skill_display in SKILL_MAP.items():
                # Word boundary check for short words like go, r, c, etc.
                pattern = r'\b' + re.escape(skill_key) + r'\b'
                if re.search(pattern, jd_lower):
                    if re.search(pattern, text_lower):
                        if skill_display not in matched_seen:
                            matched_keywords.append(skill_display)
                            matched_seen.add(skill_display)
                    else:
                        if skill_display not in missing_seen:
                            missing_keywords.append(skill_display)
                            missing_seen.add(skill_display)
            
            # 2. Extract technical or capitalized candidate terms from JD (e.g. 4+ letters)
            jd_tokens = set(re.findall(r'\b[A-Za-z][A-Za-z0-9+#.-]{2,18}\b', jd))
            common_words = {
                "with", "that", "this", "from", "have", "will", "your", "work", "team", "year",
                "years", "role", "must", "good", "well", "plus", "such", "into", "some", "them",
                "then", "than", "each", "both", "their", "about", "other", "after", "first",
                "using", "able", "need", "needs", "skills", "experience", "required", "preferred",
                "responsibilities", "qualifications", "strong", "proven", "ability", "working",
                "candidate", "including", "across", "build", "create", "support", "deliver"
            }
            for token in jd_tokens:
                token_lower = token.lower()
                if token_lower in common_words or len(token) < 3 or token_lower in SKILL_MAP:
                    continue
                # If token appears in JD as technical term
                if re.search(r'\b' + re.escape(token_lower) + r'\b', text_lower):
                    display = token.capitalize()
                    if display not in matched_seen and len(matched_keywords) < 8:
                        matched_keywords.append(display)
                        matched_seen.add(display)
                else:
                    display = token.capitalize()
                    if display not in missing_seen and len(missing_keywords) < 8:
                        missing_keywords.append(display)
                        missing_seen.add(display)

        elif target_role and target_role.strip():
            role_lower = target_role.lower()
            for skill_key, skill_display in SKILL_MAP.items():
                pattern = r'\b' + re.escape(skill_key) + r'\b'
                if re.search(pattern, text_lower):
                    if skill_display not in matched_seen:
                        matched_keywords.append(skill_display)
                        matched_seen.add(skill_display)
            
            # If target role words are missing from resume
            for word in re.findall(r'\w+', role_lower):
                if len(word) > 2 and word not in text_lower and word not in missing_seen:
                    missing_keywords.append(word.capitalize())
                    missing_seen.add(word.capitalize())
        else:
            # No JD and no target role: extract detected resume skills only, missing is empty
            for skill_key, skill_display in SKILL_MAP.items():
                pattern = r'\b' + re.escape(skill_key) + r'\b'
                if re.search(pattern, text_lower):
                    if skill_display not in matched_seen:
                        matched_keywords.append(skill_display)
                        matched_seen.add(skill_display)
            missing_keywords = []

        # 3. Grounded Strengths
        strengths: List[str] = []
        action_verbs = ["developed", "managed", "led", "created", "designed", "implemented", "built", "optimized", "increased", "reduced"]
        verb_count = sum(1 for v in action_verbs if v in text_lower)
        metrics_count = len(re.findall(r"\d+%", text)) + len(re.findall(r"\$?\d+[MK]?", text))
        
        if metrics_count >= 1 and verb_count >= 3:
            strengths.append("Measurable achievements included with quantifiable metrics and action verbs.")
        if section_scores.get("Completeness", 0) >= 75:
            strengths.append("Standard ATS section structure present (Experience, Education, Skills).")
        if len(matched_keywords) >= 3:
            strengths.append(f"Identified {len(matched_keywords)} verified industry-relevant technical competencies.")
        elif section_scores.get("Education", 0) == 100:
            strengths.append("Academic qualifications and degrees clearly formatted.")
        if "projects" in text_lower or "portfolio" in text_lower:
            if len(strengths) < 3:
                strengths.append("Dedicated project section highlighting practical software engineering.")

        if not strengths:
            strengths.append("Document text parsed successfully with readable ATS section layout.")

        # 4. Grounded Actionable Recommendations
        recommendations: List[str] = []
        if missing_keywords:
            top_missing = ", ".join(missing_keywords[:3])
            recommendations.append(f"Add priority target keywords to Skills or Experience: {top_missing}.")
        if metrics_count < 2:
            recommendations.append("Add measurable outcomes (percentages, performance gains, revenue) to experience bullets.")
        if verb_count < 4:
            recommendations.append("Start bullet points with assertive action verbs (e.g., Developed, Engineered, Optimized).")
        if section_scores.get("Projects", 0) == 0:
            recommendations.append("Add a Projects section featuring real-world applications or open-source repositories.")
        if section_scores.get("Completeness", 0) < 75:
            recommendations.append("Ensure distinct section headings (Experience, Education, Skills) are explicitly labeled.")
        if section_scores.get("Readability", 0) < 100:
            recommendations.append("Format bullet points into 1-2 concise lines to avoid unparsed content blocks.")

        if not recommendations:
            recommendations.append("Tailor bullet points to specific impact statements reflecting the target job role.")

        return {
            "semantic_analysis_available": False,
            "semantic_error": "AI semantic analysis timed out. Displaying local deterministic keyword match and rule-based recommendations.",
            "analysis_mode": "deterministic_fallback",
            "matched_keywords": matched_keywords[:8],
            "missing_keywords": missing_keywords[:8],
            "strengths": strengths[:3],
            "recommendations": recommendations[:5]
        }
