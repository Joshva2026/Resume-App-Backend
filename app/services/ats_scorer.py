from typing import Dict, Any, List, Optional, Set
import re

class AtsScorer:
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
        Extracts only the body text for scoring, explicitly dropping 'personal_info'.
        Includes section headings so scoring models recognize structured sections clearly.
        """
        body_parts = []
        for key, value in parsed_structure.items():
            if key != "personal_info" and isinstance(value, str) and value.strip():
                heading = key.replace('_', ' ').title()
                body_parts.append(f"{heading}:\n{value.strip()}")
        
        return "\n\n".join(body_parts)

    def _extract_achievement_metrics(self, text: str) -> List[str]:
        """
        Identifies genuine quantitative results: percentages, currency, multipliers, scale indicators,
        and outcome statements. Explicitly ignores calendar years (19xx, 20xx), dates, phone numbers,
        version numbers, and zip codes.
        """
        # Step 1: Strip obvious date ranges, 4-digit calendar years, phone numbers, and version numbers
        # to avoid false positive counting
        cleaned = text
        # Remove phone numbers e.g. 555-0199, (555) 012-3456
        cleaned = re.sub(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' [PHONE] ', cleaned)
        # Remove date ranges e.g. 2018 - 2023, 2018-present
        cleaned = re.sub(r'\b(?:19|20)\d{2}\s*[-–—to/]\s*(?:(?:19|20)\d{2}|present|current)\b', ' [DATERANGE] ', cleaned, flags=re.IGNORECASE)
        # Remove standalone calendar years (1900-2099)
        cleaned = re.sub(r'\b(?:19|20)\d{2}\b', ' [YEAR] ', cleaned)
        # Remove version strings e.g. v3.2.1 or version 2.0
        cleaned = re.sub(r'\b(?:v|version)?\s*\d+\.\d+(?:\.\d+)?\b', ' [VERSION] ', cleaned, flags=re.IGNORECASE)

        metrics = []

        # 1. Percentages (e.g. 45%, 99.9%, 20 %)
        for m in re.finditer(r'\b\d+(?:\.\d+)?\s*%', cleaned):
            metrics.append(m.group(0))

        # 2. Currency gains or savings (e.g. $2M, $500K, $10,000, 50K dollars)
        for m in re.finditer(r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?\s*[kKmMbB]?\b|\b\d+(?:\.\d+)?\s*[kKmMbB]?\s*dollars?\b', cleaned):
            metrics.append(m.group(0))

        # 3. Multipliers (e.g. 10x, 3x faster, 2.5x throughput)
        for m in re.finditer(r'\b\d+(?:\.\d+)?x\b', cleaned, flags=re.IGNORECASE):
            metrics.append(m.group(0))

        # 4. Scale volumes (e.g. 50M records, 100k users, 2 million events)
        for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:k|m|million|billion|thousand)\b', cleaned, flags=re.IGNORECASE):
            metrics.append(m.group(0))

        # 5. Quantified performance or team size in context
        # e.g. "team of 6 engineers", "led 8 developers", "reduced latency by 120ms"
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
        Deterministic 100-point multi-factor scoring model.
        Accurately verifies section presence using parsed_structure or explicit headings.
        Eliminates false metric counting and substring verb false positives.
        """
        text_lower = text.lower()
        score = 0.0
        section_scores: Dict[str, Optional[int]] = {}

        # -------------------------------------------------------------
        # 1. Section Completeness / Presence (15%)
        # -------------------------------------------------------------
        # Check both parsed_structure (if passed) and text boundaries
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

        # 3 core sections = 75 pts. 1 opt = 85, 2 opt = 95, 3 opt = 100
        completeness = min(100, (found_core * 25) + (found_opt * 10))
        section_scores["Completeness"] = completeness
        score += completeness * 0.15

        # -------------------------------------------------------------
        # 2. Experience / Action Verbs & True Metrics (20%)
        # -------------------------------------------------------------
        action_verbs = [
            "developed", "managed", "led", "created", "designed", "implemented",
            "built", "optimized", "increased", "reduced", "engineered", "architected",
            "delivered", "automated", "streamlined", "scaled", "spearheaded",
            "orchestrated", "executed", "collaborated"
        ]
        # Use word boundaries so "led" does NOT match "handled", "scheduled", "compiled"
        verb_count = sum(1 for v in action_verbs if re.search(r'\b' + re.escape(v) + r'\b', text_lower))
        
        # Genuine achievement metrics
        achievement_metrics = self._extract_achievement_metrics(text)
        metrics_count = len(achievement_metrics)

        # Baseline experience has points from structure if experience exists
        exp_base = 20 if has_exp else 0
        exp_score = min(100, exp_base + (verb_count * 10) + (metrics_count * 15))
        section_scores["Experience"] = exp_score
        score += exp_score * 0.20

        # -------------------------------------------------------------
        # 3. Relevant Skills & Technologies (15%)
        # -------------------------------------------------------------
        detected_skills = set()
        for skill_key, skill_display in self.SKILL_MAP.items():
            pattern = self._skill_regex(skill_key)
            if re.search(pattern, text_lower):
                detected_skills.add(skill_display)

        # If JD is provided, skill score reflects JD skill match density
        jd_skills_in_jd = set()
        if jd and jd.strip():
            jd_lower = jd.lower()
            for skill_key, skill_display in self.SKILL_MAP.items():
                pattern = self._skill_regex(skill_key)
                if re.search(pattern, jd_lower):
                    jd_skills_in_jd.add(skill_display)

        if jd_skills_in_jd:
            matched_jd_skills = detected_skills.intersection(jd_skills_in_jd)
            jd_skill_ratio = len(matched_jd_skills) / max(1, len(jd_skills_in_jd))
            skill_score = min(100, int(jd_skill_ratio * 100))
        else:
            # General tech density (7+ skills = 100%)
            skill_score = min(100, len(detected_skills) * 15)

        section_scores["Skills"] = skill_score
        score += skill_score * 0.15

        # -------------------------------------------------------------
        # 4. Projects (10%)
        # -------------------------------------------------------------
        if has_projects:
            project_score = 100
        elif re.search(r'\b(open-source|github\.com|portfolio|project)\b', text_lower):
            project_score = 60
        else:
            project_score = 0
        section_scores["Projects"] = project_score
        score += project_score * 0.10

        # -------------------------------------------------------------
        # 5. Education / Qualifications (10%)
        # -------------------------------------------------------------
        edu_score = 100 if has_edu else 0
        section_scores["Education"] = edu_score
        score += edu_score * 0.10

        # -------------------------------------------------------------
        # 6. Certifications / Achievements (5%)
        # -------------------------------------------------------------
        cert_score = 100 if has_certs else (50 if re.search(r'\b(certified|certification|aws|gcp|azure|pmp|cissp)\b', text_lower) else 0)
        section_scores["Certifications"] = cert_score
        score += cert_score * 0.05

        # -------------------------------------------------------------
        # 7. Target Role Alignment (10%)
        # -------------------------------------------------------------
        if target_role and target_role.strip():
            role_tokens = [w for w in re.findall(r'\b\w+\b', target_role.lower()) if w not in self.COMMON_STOPWORDS and len(w) > 2]
            if role_tokens:
                overlap = sum(1 for w in role_tokens if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
                role_score = min(100, int((overlap / len(role_tokens)) * 100))
            else:
                role_score = 100
            section_scores["Target Role"] = role_score
            score += role_score * 0.10
        else:
            section_scores["Target Role"] = None

        # -------------------------------------------------------------
        # 8. Job Description Match Score (10%)
        # -------------------------------------------------------------
        if jd and jd.strip():
            # 1. Technical Skills match component (50%)
            if jd_skills_in_jd:
                matched_jd_skills = detected_skills.intersection(jd_skills_in_jd)
                skills_match_ratio = len(matched_jd_skills) / max(1, len(jd_skills_in_jd))
            else:
                skills_match_ratio = min(1.0, len(detected_skills) / 5.0)

            # 2. Key requirements/domain keywords match (35%)
            jd_keywords = set()
            for token in re.findall(r'\b[A-Za-z][A-Za-z0-9+#.-]{2,18}\b', jd):
                token_lower = token.lower()
                if token_lower not in self.COMMON_STOPWORDS and len(token) > 2:
                    jd_keywords.add(token_lower)

            if jd_keywords:
                matched_kw = sum(1 for kw in jd_keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
                # Reasonable density factor: matching 40%+ of unique keywords represents a strong match
                keyword_ratio = min(1.0, (matched_kw / len(jd_keywords)) * 2.2)
            else:
                keyword_ratio = 1.0

            # 3. Experience & Education alignment (15%)
            context_match = 0.5
            if has_exp:
                context_match += 0.3
            if has_edu:
                context_match += 0.2
            context_match = min(1.0, context_match)

            jd_total = (skills_match_ratio * 50) + (keyword_ratio * 35) + (context_match * 15)
            jd_score = min(100, max(0, int(round(jd_total))))
            section_scores["JD Match"] = jd_score
            score += jd_score * 0.10
        else:
            section_scores["JD Match"] = None

        # -------------------------------------------------------------
        # 9. ATS Readability / Formatting (5%)
        # -------------------------------------------------------------
        lines = text.split("\n")
        non_empty_lines = len([l for l in lines if l.strip()])
        readability = 100 if 10 <= non_empty_lines <= 250 else 60
        section_scores["Readability"] = readability
        score += readability * 0.05

        # -------------------------------------------------------------
        # Weight Redistribution for Omitted Role / JD
        # -------------------------------------------------------------
        used_weight = 1.0
        if not target_role:
            used_weight -= 0.10
        if not jd:
            used_weight -= 0.10

        final_score = min(100, max(0, int(round(score * (1.0 / used_weight)))))

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
        Extracts verified matched/missing keywords, strengths, and actionable
        recommendations deterministically from resume text and job description/role.
        Never fabricates skills or credentials. Strictly filters generic stopwords.
        """
        text_lower = text.lower()
        if section_scores is None:
            section_scores = self.calculate_score(text, jd, target_role, parsed_structure=parsed_structure)["sections"]

        matched_keywords: List[str] = []
        missing_keywords: List[str] = []
        matched_seen = set()
        missing_seen = set()

        if jd and jd.strip():
            jd_lower = jd.lower()
            # 1. Match known skills from taxonomy
            for skill_key, skill_display in self.SKILL_MAP.items():
                pattern = self._skill_regex(skill_key)
                if re.search(pattern, jd_lower):
                    if re.search(pattern, text_lower):
                        if skill_display not in matched_seen:
                            matched_keywords.append(skill_display)
                            matched_seen.add(skill_display)
                    else:
                        if skill_display not in missing_seen:
                            missing_keywords.append(skill_display)
                            missing_seen.add(skill_display)

            # 2. Extract technical acronyms or specific technical candidate tokens (all-caps acronyms e.g. SDK, RPC)
            jd_tokens = set(re.findall(r'\b[A-Za-z][A-Za-z0-9+#.-]{1,18}\b', jd))
            for token in jd_tokens:
                token_lower = token.lower()
                if (
                    token_lower in self.COMMON_STOPWORDS 
                    or len(token) < 3 
                    or token_lower in self.SKILL_MAP
                    or any(s.lower() == token_lower for s in matched_seen)
                    or any(s.lower() == token_lower for s in missing_seen)
                ):
                    continue

                # Only include terms that are explicitly all-caps acronyms or recognized technical terms
                is_technical_candidate = token.isupper() and 2 <= len(token) <= 5
                if is_technical_candidate:
                    if re.search(r'\b' + re.escape(token_lower) + r'\b', text_lower):
                        if token not in matched_seen and len(matched_keywords) < 8:
                            matched_keywords.append(token)
                            matched_seen.add(token)
                    else:
                        if token not in missing_seen and len(missing_keywords) < 8:
                            missing_keywords.append(token)
                            missing_seen.add(token)

        elif target_role and target_role.strip():
            role_lower = target_role.lower()
            for skill_key, skill_display in self.SKILL_MAP.items():
                pattern = self._skill_regex(skill_key)
                if re.search(pattern, text_lower):
                    if skill_display not in matched_seen:
                        matched_keywords.append(skill_display)
                        matched_seen.add(skill_display)

            # If target role words are missing from resume (excluding stopwords)
            for word in re.findall(r'\b\w+\b', role_lower):
                if len(word) > 2 and word not in self.COMMON_STOPWORDS and not re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    display = word.capitalize()
                    if display not in missing_seen:
                        missing_keywords.append(display)
                        missing_seen.add(display)
        else:
            # No JD and no target role: extract detected resume skills only, missing is empty
            for skill_key, skill_display in self.SKILL_MAP.items():
                pattern = self._skill_regex(skill_key)
                if re.search(pattern, text_lower):
                    if skill_display not in matched_seen:
                        matched_keywords.append(skill_display)
                        matched_seen.add(skill_display)
            missing_keywords = []

        # 3. Grounded Strengths
        strengths: List[str] = []
        action_verbs = [
            "developed", "managed", "led", "created", "designed", "implemented",
            "built", "optimized", "increased", "reduced", "engineered", "architected"
        ]
        verb_count = sum(1 for v in action_verbs if re.search(r'\b' + re.escape(v) + r'\b', text_lower))
        achievement_metrics = self._extract_achievement_metrics(text)

        if len(achievement_metrics) >= 1 and verb_count >= 2:
            strengths.append("Measurable achievements included with quantifiable metrics and action verbs.")
        if section_scores.get("Completeness", 0) >= 75:
            strengths.append("Standard ATS section structure present (Experience, Education, Skills).")
        if len(matched_keywords) >= 3:
            strengths.append(f"Identified {len(matched_keywords)} verified industry-relevant technical competencies.")
        elif section_scores.get("Education", 0) == 100:
            strengths.append("Academic qualifications and degrees clearly formatted.")
        if section_scores.get("Projects", 0) == 100:
            strengths.append("Dedicated project section highlighting practical engineering experience.")

        if not strengths:
            strengths.append("Document text parsed successfully with readable ATS section layout.")

        # 4. Grounded Actionable Recommendations
        recommendations: List[str] = []
        if missing_keywords:
            top_missing = ", ".join(missing_keywords[:3])
            recommendations.append(f"Add priority target keywords to Skills or Experience: {top_missing}.")
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
            "missing_keywords": missing_keywords[:8],
            "strengths": strengths[:3],
            "recommendations": recommendations[:5]
        }
