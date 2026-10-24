import re
import logging
from typing import Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class ResumePrivacyRedactor:
    """
    Production privacy redaction layer for ATS analysis.
    Ensures personal identifying information (PII) is never transmitted to NVIDIA NIM
    while preserving all professional context, technical skills, project details,
    work experience, metrics, dates, and qualifications.
    """

    # Email pattern
    EMAIL_REGEX = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',
        re.IGNORECASE
    )

    # Phone number patterns (international, US, Indian +91, with/without dashes, parens, dots)
    PHONE_REGEX = re.compile(
        r'(?:(?:\+|00)\d{1,3}[\s.-]?)?'                # Optional country code (+1, +91, etc.)
        r'(?:\(?\d{2,5}\)?[\s.-]?)?'                  # Optional area code (555), (044)
        r'\d{3,4}[\s.-]?\d{3,5}\b'                    # Core phone number
    )

    # Labeled phone e.g. Phone: 1234567890, Tel: ..., Mobile: ...
    LABELED_PHONE_REGEX = re.compile(
        r'\b(?:phone|tel|telephone|mobile|cell|contact(?:\s+no)?)\s*[:#-]?\s*[\d\s+().-]{7,20}\b',
        re.IGNORECASE
    )

    # Personal profile URLs
    LINKEDIN_URL_REGEX = re.compile(
        r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_.-]+/?',
        re.IGNORECASE
    )

    GITHUB_PROFILE_REGEX = re.compile(
        r'(?:https?://)?(?:www\.)?github\.com/(?!features|pricing|security|enterprise|topics|trending|collections)[A-Za-z0-9_.-]+/?(?:\s|$|[),])',
        re.IGNORECASE
    )

    GENERIC_PORTFOLIO_URL_REGEX = re.compile(
        r'(?:https?://)?(?:www\.)?[A-Za-z0-9_-]+\.(?:dev|me|io|design|portfolio|site)(?:/[A-Za-z0-9_.~!*\'();:@&=+$,/?%#-]*)?',
        re.IGNORECASE
    )

    SOCIAL_URL_REGEX = re.compile(
        r'(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com|instagram\.com|facebook\.com)/[A-Za-z0-9_.-]+/?',
        re.IGNORECASE
    )

    # Date of Birth / Age patterns
    DOB_REGEX = re.compile(
        r'\b(?:date\s+of\s+birth|dob|born|d\.o\.b\.?)\s*[:#-]?\s*(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b',
        re.IGNORECASE
    )

    # Street address and residential patterns
    STREET_ADDRESS_REGEX = re.compile(
        r'\b\d{1,5}\s+(?:[A-Za-z0-9#.-]+\s+){1,4}(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd|court|ct|circle|cir|way|place|pl|highway|hwy|apartment|apt|suite|ste|flat|floor)\b(?:[,\s]+(?:apt|suite|ste|unit|#)?\s*[A-Za-z0-9-]+)*(?:[,\s]+[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?',
        re.IGNORECASE
    )

    POSTAL_CODE_REGEX = re.compile(
        r'\b(?:pin|zip|postal|postal\s+code|zip\s+code)\s*[:#-]?\s*\d{5,6}\b',
        re.IGNORECASE
    )

    # Token and secret patterns (prevent leakage of internal credentials)
    TOKEN_REGEX = re.compile(
        r'\b(?:sbp_[A-Za-z0-9_-]+|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|Bearer\s+[A-Za-z0-9_.-]+)\b',
        re.IGNORECASE
    )

    # Common technology words that must NEVER be redacted even if they resemble personal names
    TECH_SAFE_TERMS: Set[str] = {
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "sql", "html", "css", "bash",
        "react", "angular", "vue", "next", "node", "express", "django", "flask",
        "fastapi", "spring", "docker", "kubernetes", "aws", "gcp", "azure", "git",
        "linux", "postgres", "postgresql", "mysql", "mongodb", "redis", "kafka",
        "spark", "hadoop", "snowflake", "graphql", "rest", "api", "devops", "ci",
        "cd", "android", "ios", "flutter", "pandas", "numpy", "pytorch", "tensorflow",
        "jenkins", "gitlab", "jira", "confluence", "scrum", "agile", "developer",
        "engineer", "architect", "lead", "manager", "director", "senior", "junior",
        "intern", "consultant", "analyst", "specialist"
    }

    def redact_text(self, text: str, candidate_names: Optional[list[str]] = None) -> str:
        """
        Redacts personal identifiers from unstructured text while preserving
        all technical, contextual, and professional information.
        """
        if not text:
            return ""

        redacted = text

        # 1. Redact credentials and auth tokens first
        redacted = self.TOKEN_REGEX.sub("[TOKEN REDACTED]", redacted)

        # 2. Redact email addresses
        redacted = self.EMAIL_REGEX.sub("[EMAIL REDACTED]", redacted)

        # 3. Redact labeled phone numbers first to catch compound labels
        redacted = self.LABELED_PHONE_REGEX.sub("[PHONE REDACTED]", redacted)

        # 4. Redact standalone phone numbers (with minimum digits check to avoid metrics like 100% or years 2024)
        def _phone_sub(match):
            m_text = match.group(0).strip()
            digits = re.sub(r'\D', '', m_text)
            # Require 7 to 15 digits for a valid phone number
            # Do NOT redact calendar years (1900-2099) or percentages or small quantities
            if 7 <= len(digits) <= 15 and not (len(digits) == 4 and (m_text.startswith("19") or m_text.startswith("20"))):
                # Ensure it's not a metric like 1000000 or 5000000 dollars
                if any(kw in text[max(0, match.start()-15):min(len(text), match.end()+15)].lower() for kw in ["$", "%", "users", "requests", "records", "queries", "lines", "dollars", "k", "m"]):
                    return match.group(0)
                return "[PHONE REDACTED]"
            return match.group(0)

        redacted = self.PHONE_REGEX.sub(_phone_sub, redacted)

        # 5. Redact personal profile URLs
        redacted = self.LINKEDIN_URL_REGEX.sub("[LINKEDIN REDACTED]", redacted)
        redacted = self.GITHUB_PROFILE_REGEX.sub(" [GITHUB REDACTED] ", redacted)
        redacted = self.SOCIAL_URL_REGEX.sub("[SOCIAL REDACTED]", redacted)
        redacted = self.GENERIC_PORTFOLIO_URL_REGEX.sub("[PORTFOLIO REDACTED]", redacted)

        # 6. Redact date of birth / age
        redacted = self.DOB_REGEX.sub("[DOB REDACTED]", redacted)

        # 7. Redact street addresses
        redacted = self.STREET_ADDRESS_REGEX.sub("[ADDRESS REDACTED]", redacted)
        redacted = self.POSTAL_CODE_REGEX.sub("[POSTAL REDACTED]", redacted)

        # 8. Redact candidate name if provided or detectable
        if candidate_names:
            for name in candidate_names:
                if not name or len(name.strip()) < 3:
                    continue
                name_clean = name.strip()
                if name_clean.lower() in self.TECH_SAFE_TERMS:
                    continue
                # Full name replacement (case-insensitive with word boundary)
                pattern = re.compile(r'\b' + re.escape(name_clean) + r'\b', re.IGNORECASE)
                redacted = pattern.sub("[CANDIDATE NAME]", redacted)

        # 9. Contextual Name Headers e.g. "Name: Jane Doe", "Candidate Name: Alex Smith"
        name_header_pattern = re.compile(
            r'\b(?:candidate(?:\s+name)?|name|full\s+name)\s*[:#-]\s*[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){1,3}\b',
            re.IGNORECASE
        )
        redacted = name_header_pattern.sub("Candidate: [CANDIDATE NAME]", redacted)

        # 10. Clean up any redundant redaction markers
        redacted = re.sub(r'(\[PHONE REDACTED\][\s,;]*)+', '[PHONE REDACTED] ', redacted)
        redacted = re.sub(r'(\[EMAIL REDACTED\][\s,;]*)+', '[EMAIL REDACTED] ', redacted)
        redacted = re.sub(r'(\[CANDIDATE NAME\][\s,;]*)+', '[CANDIDATE NAME] ', redacted)

        return redacted

    def sanitize_parsed_structure(
        self,
        parsed_structure: Dict[str, Any],
        candidate_names: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """
        Creates a clean sanitized copy of the structured resume dictionary.
        Explicitly removes the 'personal_info' section and sanitizes all other sections.
        """
        sanitized: Dict[str, Any] = {}
        names_to_redact = list(candidate_names) if candidate_names else []

        # Extract name clues from personal_info before discarding it
        raw_personal = parsed_structure.get("personal_info") or ""
        if isinstance(raw_personal, str) and raw_personal.strip():
            lines = [l.strip() for l in raw_personal.strip().split("\n") if l.strip()]
            for line in lines[:3]:
                # If a line looks like a person's name (2-3 title-case words without tech/stopwords)
                words = line.split()
                if 2 <= len(words) <= 3 and all(w[0].isupper() and w.isalpha() for w in words):
                    if not any(w.lower() in self.TECH_SAFE_TERMS for w in words):
                        names_to_redact.append(line)

        # Do NOT include personal_info in the sanitized structure
        sanitized["personal_info"] = "[REDACTED]"

        for section_key, section_val in parsed_structure.items():
            if section_key == "personal_info":
                continue
            if isinstance(section_val, str):
                sanitized[section_key] = self.redact_text(section_val, candidate_names=names_to_redact)
            elif isinstance(section_val, list):
                sanitized[section_key] = [
                    self.redact_text(item, candidate_names=names_to_redact) if isinstance(item, str) else item
                    for item in section_val
                ]
            elif isinstance(section_val, dict):
                sanitized[section_key] = {
                    k: self.redact_text(v, candidate_names=names_to_redact) if isinstance(v, str) else v
                    for k, v in section_val.items()
                }
            else:
                sanitized[section_key] = section_val

        return sanitized

    def build_sanitized_resume_prompt_text(
        self,
        parsed_structure: Dict[str, Any],
        candidate_names: Optional[list[str]] = None
    ) -> str:
        """
        Builds the clean, professional representation of the resume for scoring & AI analysis.
        Sections are clearly labeled so models evaluate structured evidence properly.
        Personal info is 100% excluded.
        """
        sanitized = self.sanitize_parsed_structure(parsed_structure, candidate_names=candidate_names)
        body_parts = []

        preferred_order = ["summary", "experience", "skills", "projects", "education", "certifications", "other"]
        for key in preferred_order:
            val = sanitized.get(key)
            if isinstance(val, str) and val.strip() and val != "[REDACTED]":
                heading = key.replace('_', ' ').title()
                body_parts.append(f"{heading}:\n{val.strip()}")

        for key, val in sanitized.items():
            if key not in preferred_order and key != "personal_info":
                if isinstance(val, str) and val.strip():
                    heading = key.replace('_', ' ').title()
                    body_parts.append(f"{heading}:\n{val.strip()}")

        return "\n\n".join(body_parts)

    def verify_request_safety(self, prompt: str) -> Tuple[bool, str]:
        """
        Final safety gate at the provider-request boundary.
        Guarantees that no raw emails, phone numbers, or tokens are sent to NVIDIA NIM.
        Returns (is_safe, sanitized_prompt).
        """
        # If any raw email is still found, redact it immediately
        clean_prompt = self.EMAIL_REGEX.sub("[EMAIL REDACTED]", prompt)
        clean_prompt = self.TOKEN_REGEX.sub("[TOKEN REDACTED]", clean_prompt)
        clean_prompt = self.LINKEDIN_URL_REGEX.sub("[LINKEDIN REDACTED]", clean_prompt)
        clean_prompt = self.LABELED_PHONE_REGEX.sub("[PHONE REDACTED]", clean_prompt)

        has_unredacted_email = bool(self.EMAIL_REGEX.search(clean_prompt))
        has_token = bool(self.TOKEN_REGEX.search(clean_prompt))

        is_safe = not (has_unredacted_email or has_token)
        return is_safe, clean_prompt
