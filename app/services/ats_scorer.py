from typing import Dict, Any, List, Optional, Set, Tuple
import re
from app.services.ats_privacy import ResumePrivacyRedactor

class AtsScorer:
    SCORING_VERSION = "ats_v2.0"

    # Curated technical and professional skills dictionary with proper casing and alias mappings
    SKILL_MAP = {
        "python": "Python", "java": "Java", "javascript": "JavaScript", "typescript": "TypeScript",
        "c++": "C++", "c#": "C#", "go": "Go", "golang": "Go", "rust": "Rust", "ruby": "Ruby",
        "php": "PHP", "swift": "Swift", "kotlin": "Kotlin", "sql": "SQL", "html": "HTML",
        "css": "CSS", "bash": "Bash", "react": "React", "react.js": "React", "angular": "Angular",
        "vue": "Vue.js", "vue.js": "Vue.js", "next.js": "Next.js", "node.js": "Node.js", "nodejs": "Node.js",
        "express": "Express", "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
        "spring": "Spring", "spring boot": "Spring Boot", ".net": ".NET", "dotnet": ".NET",
        "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes", "terraform": "Terraform",
        "ci/cd": "CI/CD", "cicd": "CI/CD", "aws": "AWS", "gcp": "GCP", "azure": "Azure",
        "git": "Git", "linux": "Linux", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
        "mysql": "MySQL", "mongodb": "MongoDB", "redis": "Redis", "elasticsearch": "Elasticsearch",
        "graphql": "GraphQL", "rest api": "REST API", "restful api": "REST API",
        "microservices": "Microservices", "agile": "Agile", "scrum": "Scrum", "tdd": "TDD",
        "machine learning": "Machine Learning", "deep learning": "Deep Learning", "pytorch": "PyTorch",
        "tensorflow": "TensorFlow", "nlp": "NLP", "llm": "LLM", "data analysis": "Data Analysis",
        "data pipeline": "Data Pipeline", "etl": "ETL", "system design": "System Design",
        "unit testing": "Unit Testing", "devops": "DevOps", "android": "Android",
        "jetpack compose": "Jetpack Compose", "ios": "iOS", "flutter": "Flutter",
        "project management": "Project Management", "leadership": "Leadership",
        "kafka": "Kafka", "rabbitmq": "RabbitMQ", "spark": "Spark", "hadoop": "Hadoop",
        "snowflake": "Snowflake", "databricks": "Databricks", "bigquery": "BigQuery"
    }

    COMMON_STOPWORDS: Set[str] = {
        "a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone",
        "along", "already", "also", "although", "always", "among", "an", "and", "another", "any",
        "anybody", "anyone", "anything", "anywhere", "apply", "are", "area", "areas", "around",
        "as", "ask", "asked", "asking", "asks", "at", "away", "b", "back", "backed", "backing",
        "backs", "be", "became", "because", "become", "becomes", "been", "before", "began",
        "behind", "being", "beings", "best", "better", "between", "big", "both", "build", "building",
        "built", "but", "by", "c", "came", "can", "cannot", "case", "cases", "certain", "certainly",
        "clear", "clearly", "come", "could", "create", "creating", "d", "deliver", "delivering",
        "did", "differ", "different", "differently", "director", "do", "does", "done", "down",
        "downed", "downing", "downs", "during", "e", "each", "early", "either", "end", "ended",
        "ending", "ends", "enough", "even", "evenly", "ever", "every", "everybody", "everyone",
        "everything", "everywhere", "experience", "experienced", "expertise", "f", "face", "faces",
        "fact", "facts", "far", "felt", "few", "fewer", "find", "finds", "first", "for", "four",
        "from", "full", "further", "furthered", "furthering", "furthers", "g", "gave", "general",
        "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods",
        "got", "great", "greater", "greatest", "group", "grouped", "grouping", "groups", "h", "had",
        "has", "have", "having", "he", "help", "helped", "helping", "helps", "her", "here", "hers",
        "herself", "high", "higher", "highest", "him", "himself", "his", "how", "however", "i", "if",
        "important", "in", "interest", "interested", "interesting", "interests", "into", "is", "it",
        "its", "itself", "j", "join", "joining", "just", "k", "keep", "keeps", "kind", "knew", "know",
        "known", "knows", "l", "large", "largely", "last", "later", "latest", "least", "less", "let",
        "lets", "like", "likely", "long", "longer", "longest", "look", "looking", "looks", "m", "made",
        "make", "making", "man", "many", "may", "me", "member", "members", "men", "might", "more",
        "most", "mostly", "mr", "mrs", "much", "must", "my", "myself", "n", "necessary", "need",
        "needed", "needing", "needs", "never", "new", "newer", "newest", "next", "no", "nobody",
        "non", "noone", "not", "nothing", "now", "nowhere", "number", "numbers", "o", "of", "off",
        "often", "old", "older", "oldest", "on", "once", "one", "only", "open", "opened", "opening",
        "opens", "opportunity", "or", "order", "ordered", "ordering", "orders", "other", "others",
        "our", "out", "over", "p", "part", "parted", "parting", "parts", "per", "place", "places",
        "point", "pointed", "pointing", "points", "possible", "present", "presented", "presenting",
        "presents", "problem", "problems", "program", "programs", "proven", "put", "puts", "q", "r",
        "rather", "really", "required", "requirements", "responsibilities", "right", "room", "rooms",
        "s", "said", "same", "saw", "say", "saying", "says", "second", "seconds", "see", "seeing",
        "seem", "seemed", "seeming", "seems", "sees", "seeking", "senior", "several", "shall", "she",
        "should", "show", "showed", "showing", "shows", "side", "sides", "since", "small", "smaller",
        "smallest", "so", "some", "somebody", "someone", "something", "somewhere", "state", "states",
        "still", "strong", "such", "support", "supporting", "sure", "t", "take", "taken", "team",
        "teams", "than", "that", "the", "their", "them", "then", "there", "therefore", "these",
        "they", "thing", "things", "think", "thinks", "this", "those", "though", "thought", "thoughts",
        "three", "through", "throughput", "thus", "to", "today", "together", "too", "took", "toward",
        "turn", "turned", "turning", "turns", "two", "u", "under", "until", "up", "upon", "us", "use",
        "used", "uses", "using", "v", "very", "w", "want", "wanted", "wanting", "wants", "was", "way",
        "ways", "we", "well", "wells", "went", "were", "what", "when", "where", "whether", "which",
        "while", "who", "whole", "whom", "whose", "why", "will", "with", "within", "without", "work",
        "worked", "working", "works", "would", "x", "y", "year", "years", "yet", "you", "young",
        "younger", "youngest", "your", "yours"
    }

    def _skill_regex(self, key: str) -> str:
        """Constructs an accurate regex boundary pattern for a skill key to avoid false positives."""
        if key == "java":
            return r'\bjava\b(?!script)'
        elif key == "javascript":
            return r'\b(?:javascript|js)\b'
        elif key == "c++":
            return r'(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9])'
        elif key == "c#":
            return r'(?<![a-zA-Z0-9])c#(?![a-zA-Z0-9])'
        elif key in [".net", "dotnet"]:
            return r'(?<![a-zA-Z0-9])(?:\.net|dotnet)(?![a-zA-Z0-9])'
        elif key in ["go", "golang"]:
            return r'\b(?:go|golang)\b'
        elif key == "r":
            return r'\br\b'
        elif key in ["ci/cd", "cicd"]:
            return r'(?<![a-zA-Z0-9])ci[\s\/-]?cd(?![a-zA-Z0-9])'
        elif key in ["k8s", "kubernetes"]:
            return r'\b(?:k8s|kubernetes)\b'
        elif key in ["postgres", "postgresql"]:
            return r'\b(?:postgres|postgresql)\b'
        elif key in ["node.js", "nodejs", "node"]:
            return r'\b(?:node\.js|nodejs|node)\b'
        elif key in ["react", "react.js", "reactjs"]:
            return r'\b(?:react\.js|reactjs|react)\b(?! native)'
        elif key == "data analysis":
            return r'\bdata analysis\b'
        elif key == "data pipeline":
            return r'\bdata pipelines?\b'
        else:
            return r'\b' + re.escape(key) + r'\b'

    def sanitize_for_scoring(self, parsed_structure: Dict[str, Any]) -> str:
        """
        Sanitizes resume structure for scoring and provider transmission.
        Completely drops personal identifying information (PII) using ResumePrivacyRedactor.
        Preserves all professional qualifications, technical skills, metrics, and experience.
        """
        redactor = ResumePrivacyRedactor()
        return redactor.build_sanitized_resume_prompt_text(parsed_structure)

    def extract_jd_requirements(self, jd: str) -> Dict[str, Any]:
        """
        Parses the job description to extract required skills, preferred skills,
        and domain technology keywords based on contextual sections.
        """
        if not jd or not jd.strip():
            return {
                "required_skills": [],
                "preferred_skills": [],
                "all_skills": []
            }

        jd_lower = jd.lower()

        # Split JD into sections if present
        # Detect sections like "Requirements", "Must Have", "Minimum Qualifications"
        # vs "Preferred", "Nice to Have", "Bonus", "Desired"
        required_patterns = [
            r'(?:requirements|required|must\s+have|minimum\s+qualifications|basic\s+qualifications|what\s+you(?:\'ll)?\s+need)([\s\S]*?)(?:preferred|nice\s+to\s+have|bonus|plus|desired|what\s+we\s+offer|benefits|$)',
        ]
        preferred_patterns = [
            r'(?:preferred|nice\s+to\s+have|bonus|plus|desired|additional\s+qualifications)([\s\S]*?)(?:what\s+we\s+offer|benefits|about\s+us|$)'
        ]

        req_section_text = ""
        for pat in required_patterns:
            m = re.search(pat, jd_lower)
            if m:
                req_section_text = m.group(1)
                break

        pref_section_text = ""
        for pat in preferred_patterns:
            m = re.search(pat, jd_lower)
            if m:
                pref_section_text = m.group(1)
                break

        all_detected_skills: List[str] = []
        required_skills: List[str] = []
        preferred_skills: List[str] = []

        for skill_key, skill_display in self.SKILL_MAP.items():
            pattern = self._skill_regex(skill_key)
            if re.search(pattern, jd_lower):
                if skill_display not in all_detected_skills:
                    all_detected_skills.append(skill_display)

                # Check if skill specifically falls in preferred or required text
                if pref_section_text and re.search(pattern, pref_section_text):
                    if skill_display not in preferred_skills:
                        preferred_skills.append(skill_display)
                elif req_section_text and re.search(pattern, req_section_text):
                    if skill_display not in required_skills:
                        required_skills.append(skill_display)

        # If no explicit sections were parsed, classify skills sensibly:
        # Core skills listed first are required; secondary/extra skills are preferred
        if not required_skills and not preferred_skills and all_detected_skills:
            if len(all_detected_skills) <= 4:
                required_skills = list(all_detected_skills)
                preferred_skills = []
            else:
                cutoff = max(2, int(len(all_detected_skills) * 0.65))
                required_skills = all_detected_skills[:cutoff]
                preferred_skills = all_detected_skills[cutoff:]
        elif not required_skills and all_detected_skills:
            required_skills = [s for s in all_detected_skills if s not in preferred_skills]

        return {
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "all_skills": all_detected_skills
        }

    def extract_skill_evidence(
        self,
        parsed_structure: Optional[Dict[str, Any]],
        text: str,
        skill_key: str,
        skill_display: str
    ) -> Optional[str]:
        """
        Extracts actual resume evidence for a matched skill (which section and sentence snippet).
        """
        pattern = self._skill_regex(skill_key)

        # 1. Search structured sections in order of evidence strength: experience, projects, summary, skills
        if parsed_structure:
            for sec_name in ["experience", "projects", "summary", "skills", "certifications"]:
                sec_val = parsed_structure.get(sec_name)
                if isinstance(sec_val, str) and sec_val.strip():
                    lines = [l.strip() for l in sec_val.split("\n") if l.strip()]
                    for line in lines:
                        if re.search(pattern, line.lower()):
                            clean_line = re.sub(r'^[•\-\*#\d\.\s]+', '', line).strip()
                            if len(clean_line) > 85:
                                clean_line = clean_line[:82] + "..."
                            return f"{sec_name.title()}: {clean_line}"

        # 2. Fallback to unstructured text search
        for line in text.split("\n"):
            line_str = line.strip()
            if line_str and re.search(pattern, line_str.lower()):
                clean_line = re.sub(r'^[•\-\*#\d\.\s]+', '', line_str).strip()
                if len(clean_line) > 85:
                    clean_line = clean_line[:82] + "..."
                return f"Resume: {clean_line}"

        return None

    def _extract_achievement_metrics(self, text: str) -> List[str]:
        """
        Identifies genuine quantitative results: percentages, currency, multipliers, scale indicators,
        and outcome statements. Explicitly ignores calendar years, dates, phone numbers, and zip codes.
        """
        cleaned = text
        cleaned = re.sub(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' [PHONE] ', cleaned)
        cleaned = re.sub(r'\b(?:19|20)\d{2}\s*[-–—to/]\s*(?:(?:19|20)\d{2}|present|current)\b', ' [DATERANGE] ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(?:19|20)\d{2}\b', ' [YEAR] ', cleaned)
        cleaned = re.sub(r'\b(?:v|version)?\s*\d+\.\d+(?:\.\d+)?\b', ' [VERSION] ', cleaned, flags=re.IGNORECASE)

        metrics = []
        for m in re.finditer(r'\b\d+(?:\.\d+)?\s*%', cleaned):
            metrics.append(m.group(0))
        for m in re.finditer(r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?\s*[kKmMbB]?\b|\b\d+(?:\.\d+)?\s*[kKmMbB]?\s*dollars?\b', cleaned):
            metrics.append(m.group(0))
        for m in re.finditer(r'\b\d+(?:\.\d+)?x\b', cleaned, flags=re.IGNORECASE):
            metrics.append(m.group(0))
        for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:k|m|million|billion|thousand)\b', cleaned, flags=re.IGNORECASE):
            metrics.append(m.group(0))
        for m in re.finditer(r'\b(?:team of|managed|led|supervised)\s+(\d+)\b', cleaned, flags=re.IGNORECASE):
            metrics.append(m.group(0))
        for m in re.finditer(r'\b(?:reduced|saved|cut|improved|uptime of)\s+[^.\n,;]*?(\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|minutes?|hours?|days?))\b', cleaned, flags=re.IGNORECASE):
            metrics.append(m.group(0))

        return metrics

    def calculate_score(
        self,
        text: str,
        jd: Optional[str] = None,
        target_role: Optional[str] = None,
        parsed_structure: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deterministic, explainable 100-point multi-factor scoring model.
        When a job description is supplied, scoring is directly anchored to that job's
        required and preferred qualifications (Required 30%, Preferred 15%, Experience 20%,
        Projects 10%, Education 10%, Readability 15%).
        Different job descriptions produce independently calculated results.
        Identical inputs produce identical scores.
        """
        text_lower = text.lower()
        section_scores: Dict[str, Optional[int]] = {}

        # -------------------------------------------------------------
        # 1. Section Completeness / Presence
        # -------------------------------------------------------------
        def check_section(sec_name: str, patterns: List[str]) -> bool:
            if parsed_structure and bool(parsed_structure.get(sec_name, "").strip()):
                return True
            for pat in patterns:
                if re.search(r'\b' + pat + r'\b', text_lower):
                    return True
            return False

        has_exp = check_section("experience", ["experience", "employment", "work history", "professional experience"])
        has_edu = check_section("education", ["education", "academic", "university", "college", "bachelor", "degree", r"b\.s\.", r"m\.s\."])
        has_skills = check_section("skills", ["skills", "technical skills", "core competencies", "technologies", "key skills"])
        has_projects = check_section("projects", ["projects", "personal projects", "portfolio", "key projects"])
        has_certs = check_section("certifications", ["certifications", "licenses", "credentials", "achievements", "awards"])
        has_summary = check_section("summary", ["summary", "profile", "objective", "about me"])

        found_core = sum([1 for x in [has_exp, has_edu, has_skills] if x])
        found_opt = sum([1 for x in [has_projects, has_certs, has_summary] if x])
        completeness = min(100, (found_core * 25) + (found_opt * 10))
        section_scores["Completeness"] = completeness

        # -------------------------------------------------------------
        # 2. Experience, Action Verbs & Metrics
        # -------------------------------------------------------------
        action_verbs = [
            "developed", "managed", "led", "created", "designed", "implemented",
            "built", "optimized", "increased", "reduced", "engineered", "architected",
            "delivered", "automated", "streamlined", "scaled", "spearheaded",
            "orchestrated", "executed", "collaborated"
        ]
        verb_count = sum(1 for v in action_verbs if re.search(r'\b' + re.escape(v) + r'\b', text_lower))
        achievement_metrics = self._extract_achievement_metrics(text)
        metrics_count = len(achievement_metrics)

        exp_base = 25 if has_exp else 0
        exp_score = min(100, exp_base + (verb_count * 10) + (metrics_count * 15))
        section_scores["Experience"] = exp_score

        # -------------------------------------------------------------
        # 3. Detect Skills Present in Candidate Resume
        # -------------------------------------------------------------
        detected_skills: Set[str] = set()
        for skill_key, skill_display in self.SKILL_MAP.items():
            pattern = self._skill_regex(skill_key)
            if re.search(pattern, text_lower):
                detected_skills.add(skill_display)

        # -------------------------------------------------------------
        # 4. Projects Score
        # -------------------------------------------------------------
        if has_projects:
            project_score = 100
        elif re.search(r'\b(open-source|github\.com|portfolio|project)\b', text_lower):
            project_score = 60
        else:
            project_score = 0
        section_scores["Projects"] = project_score

        # -------------------------------------------------------------
        # 5. Education & Certifications Score
        # -------------------------------------------------------------
        edu_score = 100 if has_edu else 0
        section_scores["Education"] = edu_score

        cert_score = 100 if has_certs else (50 if re.search(r'\b(certified|certification|aws|gcp|azure|pmp|cissp)\b', text_lower) else 0)
        section_scores["Certifications"] = cert_score

        # -------------------------------------------------------------
        # 6. Readability / Formatting
        # -------------------------------------------------------------
        lines = text.split("\n")
        non_empty_lines = len([l for l in lines if l.strip()])
        readability = 100 if 10 <= non_empty_lines <= 250 else 65
        section_scores["Readability"] = readability

        # -------------------------------------------------------------
        # 7. Job-Specific Evaluation (When JD is provided)
        # -------------------------------------------------------------
        has_jd = bool(jd and jd.strip())
        if has_jd:
            jd_reqs = self.extract_jd_requirements(jd or "")
            req_skills = set(jd_reqs["required_skills"])
            pref_skills = set(jd_reqs["preferred_skills"])

            # Required skill coverage (0 to 100)
            if req_skills:
                matched_req = detected_skills.intersection(req_skills)
                req_coverage_score = int(round((len(matched_req) / len(req_skills)) * 100))
            else:
                # If no specific required skills in taxonomy, check general skill density
                req_coverage_score = min(100, len(detected_skills) * 15)

            # Preferred skill coverage (0 to 100)
            if pref_skills:
                matched_pref = detected_skills.intersection(pref_skills)
                pref_coverage_score = int(round((len(matched_pref) / len(pref_skills)) * 100))
            else:
                # If no preferred skills specified in JD, tie preferred coverage directly to required match
                pref_coverage_score = req_coverage_score

            # Domain keyword match from JD text
            jd_keywords = set()
            for token in re.findall(r'\b[A-Za-z][A-Za-z0-9+#.-]{2,18}\b', jd or ""):
                t_low = token.lower()
                if t_low not in self.COMMON_STOPWORDS and len(token) > 2:
                    jd_keywords.add(t_low)

            if jd_keywords:
                matched_kw = sum(1 for kw in jd_keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
                keyword_match_ratio = min(1.0, (matched_kw / len(jd_keywords)) * 2.0)
            else:
                keyword_match_ratio = 1.0

            # Technical Skills Section Score: combines required & preferred coverage
            overall_skill_score = int(round((req_coverage_score * 0.70) + (pref_coverage_score * 0.30)))
            section_scores["Skills"] = overall_skill_score

            # JD Match metric: combines required skills (50%), preferred (25%), keywords (25%)
            jd_match_pct = int(round((req_coverage_score * 0.50) + (pref_coverage_score * 0.25) + (keyword_match_ratio * 25.0)))
            section_scores["JD Match"] = min(100, max(0, jd_match_pct))

            # Role Alignment
            if target_role and target_role.strip():
                role_tokens = [w for w in re.findall(r'\b\w+\b', target_role.lower()) if w not in self.COMMON_STOPWORDS and len(w) > 2]
                if role_tokens:
                    overlap = sum(1 for w in role_tokens if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
                    role_score = min(100, int((overlap / len(role_tokens)) * 100))
                else:
                    role_score = 100
                section_scores["Target Role"] = role_score
            else:
                section_scores["Target Role"] = None

            # DYNAMIC WEIGHTED SCORING FORMULA (Anchored to Job Requirements):
            # Required Skill Coverage: 30%
            # Preferred Skills / Stack: 15%
            # Experience & Impact: 20%
            # Project Alignment: 10%
            # Education & Certifications: 10%
            # ATS Structure & Readability: 15%
            edu_cert_combined = (edu_score * 0.6) + (cert_score * 0.4)
            ats_combined = (completeness * 0.65) + (readability * 0.35)

            final_calculated = (
                (req_coverage_score * 0.30) +
                (pref_coverage_score * 0.15) +
                (exp_score * 0.20) +
                (project_score * 0.10) +
                (edu_cert_combined * 0.10) +
                (ats_combined * 0.15)
            )

            # Bonus for strong role alignment if specified
            if section_scores["Target Role"] is not None:
                role_bonus = (section_scores["Target Role"] - 50) * 0.05
                final_calculated += role_bonus

            final_score = min(100, max(15, int(round(final_calculated))))

        else:
            # General Baseline Resume Audit (No Job Description provided)
            skill_score = min(100, len(detected_skills) * 15)
            section_scores["Skills"] = skill_score
            section_scores["JD Match"] = None

            if target_role and target_role.strip():
                role_tokens = [w for w in re.findall(r'\b\w+\b', target_role.lower()) if w not in self.COMMON_STOPWORDS and len(w) > 2]
                if role_tokens:
                    overlap = sum(1 for w in role_tokens if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
                    role_score = min(100, int((overlap / len(role_tokens)) * 100))
                else:
                    role_score = 100
                section_scores["Target Role"] = role_score
            else:
                section_scores["Target Role"] = None

            # Weights: Completeness 20%, Experience 25%, Skills 20%, Projects 15%, Edu/Certs 10%, Readability 10%
            edu_cert_combined = (edu_score * 0.6) + (cert_score * 0.4)
            final_calculated = (
                (completeness * 0.20) +
                (exp_score * 0.25) +
                (skill_score * 0.20) +
                (project_score * 0.15) +
                (edu_cert_combined * 0.10) +
                (readability * 0.10)
            )
            final_score = min(100, max(20, int(round(final_calculated))))

        return {
            "overall": final_score,
            "sections": section_scores
        }

    def calculate_deterministic_semantics(
        self,
        text: str,
        jd: Optional[str] = None,
        target_role: Optional[str] = None,
        section_scores: Optional[Dict[str, Any]] = None,
        parsed_structure: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates matched keywords with resume evidence, missing required/preferred skills,
        and job-specific recommendations deterministically from resume and job description.
        Distinguishes missing required skills from preferred skills.
        Clearly labels output as deterministic fallback.
        """
        text_lower = text.lower()
        if section_scores is None:
            section_scores = self.calculate_score(text, jd, target_role, parsed_structure=parsed_structure)["sections"]

        matched_keywords: List[str] = []
        matched_with_evidence: List[str] = []
        missing_required: List[str] = []
        missing_preferred: List[str] = []
        missing_keywords: List[str] = []

        matched_seen = set()
        missing_seen = set()

        if jd and jd.strip():
            jd_reqs = self.extract_jd_requirements(jd)
            req_skills = jd_reqs["required_skills"]
            pref_skills = jd_reqs["preferred_skills"]
            all_jd_skills = jd_reqs["all_skills"]

            # Map display name to key
            disp_to_key = {}
            for k, v in self.SKILL_MAP.items():
                if v not in disp_to_key:
                    disp_to_key[v] = k

            for skill_disp in all_jd_skills:
                skill_k = disp_to_key.get(skill_disp, skill_disp.lower())
                pat = self._skill_regex(skill_k)

                if re.search(pat, text_lower):
                    if skill_disp not in matched_seen:
                        matched_keywords.append(skill_disp)
                        evidence = self.extract_skill_evidence(parsed_structure, text, skill_k, skill_disp)
                        if evidence:
                            matched_with_evidence.append(f"{skill_disp} ({evidence})")
                        else:
                            matched_with_evidence.append(skill_disp)
                        matched_seen.add(skill_disp)
                else:
                    if skill_disp not in missing_seen:
                        missing_seen.add(skill_disp)
                        missing_keywords.append(skill_disp)
                        if skill_disp in req_skills:
                            missing_required.append(skill_disp)
                        else:
                            missing_preferred.append(skill_disp)

        elif target_role and target_role.strip():
            role_lower = target_role.lower()
            for skill_key, skill_display in self.SKILL_MAP.items():
                pattern = self._skill_regex(skill_key)
                if re.search(pattern, text_lower):
                    if skill_display not in matched_seen:
                        matched_keywords.append(skill_display)
                        evidence = self.extract_skill_evidence(parsed_structure, text, skill_key, skill_display)
                        matched_with_evidence.append(f"{skill_display} ({evidence})" if evidence else skill_display)
                        matched_seen.add(skill_display)

            for word in re.findall(r'\b\w+\b', role_lower):
                if len(word) > 2 and word not in self.COMMON_STOPWORDS and not re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    display = word.capitalize()
                    if display not in missing_seen:
                        missing_keywords.append(display)
                        missing_seen.add(display)
        else:
            # Baseline detection
            for skill_key, skill_display in self.SKILL_MAP.items():
                pattern = self._skill_regex(skill_key)
                if re.search(pattern, text_lower):
                    if skill_display not in matched_seen:
                        matched_keywords.append(skill_display)
                        evidence = self.extract_skill_evidence(parsed_structure, text, skill_key, skill_display)
                        matched_with_evidence.append(f"{skill_display} ({evidence})" if evidence else skill_display)
                        matched_seen.add(skill_display)
            missing_keywords = []

        # Grounded Strengths
        strengths: List[str] = []
        action_verbs = [
            "developed", "managed", "led", "created", "designed", "implemented",
            "built", "optimized", "increased", "reduced", "engineered", "architected"
        ]
        verb_count = sum(1 for v in action_verbs if re.search(r'\b' + re.escape(v) + r'\b', text_lower))
        achievement_metrics = self._extract_achievement_metrics(text)

        if len(achievement_metrics) >= 1 and verb_count >= 2:
            strengths.append(f"Includes quantifiable achievements ({', '.join(achievement_metrics[:2])}) with action verbs.")
        if section_scores.get("Completeness", 0) >= 75:
            strengths.append("Standard ATS section structure present (Experience, Education, Skills).")
        if len(matched_keywords) >= 3:
            strengths.append(f"Demonstrates {len(matched_keywords)} verified competencies aligned with role requirements.")
        elif section_scores.get("Education", 0) == 100:
            strengths.append("Academic qualifications and degrees clearly formatted.")
        if section_scores.get("Projects", 0) == 100:
            strengths.append("Dedicated project section highlighting practical engineering experience.")

        if not strengths:
            strengths.append("Document text parsed successfully with readable ATS section layout.")

        # Grounded Actionable Recommendations
        recommendations: List[str] = []
        if missing_required:
            top_req = ", ".join(missing_required[:3])
            recommendations.append(f"Target job requires {top_req}: add practical experience or projects demonstrating these.")
        if missing_preferred:
            top_pref = ", ".join(missing_preferred[:2])
            recommendations.append(f"Incorporate preferred technologies ({top_pref}) to stand out against competing candidates.")
        if len(achievement_metrics) < 2:
            recommendations.append("Add measurable outcomes (percentages, performance gains, revenue) to experience bullets.")
        if verb_count < 3:
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
            "matched_with_evidence": matched_with_evidence[:8],
            "missing_keywords": missing_keywords[:8],
            "missing_required": missing_required[:8],
            "missing_preferred": missing_preferred[:8],
            "strengths": strengths[:3],
            "recommendations": recommendations[:5]
        }
