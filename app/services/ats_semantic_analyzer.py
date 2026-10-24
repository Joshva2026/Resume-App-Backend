import json
import logging
import re
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.services.ai_provider import AIProvider, ProviderTimeoutError, ProviderUnavailableError
from app.services.grok_provider import XAIGrokAIProvider, GrokAIProvider
from app.services.groq_provider import GroqAIProvider
from app.schemas.ai import AIRequest
from app.services.ats_scorer import AtsScorer
from app.services.ats_privacy import ResumePrivacyRedactor

logger = logging.getLogger(__name__)

def get_default_ats_provider() -> AIProvider:
    """
    Selects the ATS AI Provider.
    1. If XAI_API_KEY is configured with a genuine xAI key (starts with 'xai-'), selects XAIGrokAIProvider with grok-2-latest.
    2. If GROQ_API_KEY is configured (or a gsk_ key is present), selects GroqAIProvider with openai/gpt-oss-120b.
    3. Keeps xAI support separate and optional. Never passes a Groq key to the xAI endpoint.
    """
    xai_key = (getattr(settings, "XAI_API_KEY", None) or "").strip()
    groq_key = (getattr(settings, "GROQ_API_KEY", None) or "").strip()

    if xai_key.startswith("xai-"):
        return XAIGrokAIProvider()

    if groq_key or xai_key.startswith("gsk_"):
        return GroqAIProvider()

    if xai_key:
        return XAIGrokAIProvider()

    return GroqAIProvider() if groq_key else XAIGrokAIProvider()

class AtsSemanticAnalyzer:
    def __init__(self, provider: Optional[AIProvider] = None):
        self.provider = provider or get_default_ats_provider()
        self.privacy_redactor = ResumePrivacyRedactor()

    def _compact_job_description(self, jd: Optional[str], max_chars: int = 6000) -> Optional[str]:
        """
        Extracts core requirements, responsibilities, and qualifications from the JD,
        preserving the exact user-provided requirements while stripping boilerplate HR perks
        and EEOC statements.
        """
        if not jd:
            return ""

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', jd) if p.strip()]
        relevant_paragraphs = []

        boilerplate_patterns = [
            r'\bequal\s+opportunity\s+employer\b',
            r'\bbenefits\s*(?:include|package|\:)\b',
            r'\b401\(?k\)?\b',
            r'\bhealth,\s*dental,\s*vision\b',
            r'\bunlimited\s+pto\b',
            r'\bcommuter\s+benefits\b',
            r'\bdiversity\s+and\s+inclusion\b',
            r'\ball\s+qualified\s+applicants\s+will\s+receive\b',
            r'\baccommodations\s+for\s+disabilities\b',
            r'\bsalary\s+range\b'
        ]

        for p in paragraphs:
            p_low = p.lower()
            if any(re.search(bp, p_low) for bp in boilerplate_patterns):
                continue
            relevant_paragraphs.append(p)

        compacted = "\n\n".join(relevant_paragraphs)
        if len(compacted) > max_chars:
            compacted = compacted[:max_chars].rsplit(" ", 1)[0] + "..."
        return compacted if compacted.strip() else jd[:max_chars]

    def _compact_resume_text(self, text: str, max_chars: int = 8000) -> str:
        """
        Preserves all relevant professional resume content (summary, experience,
        skills, projects, education, certifications, metrics, and dates)
        while providing a safe boundary against unbounded inputs.
        """
        if not text or len(text) <= max_chars:
            return text or ""
        return text[:max_chars].rsplit("\n", 1)[0]

    async def analyze(
        self,
        sanitized_text: str,
        jd: Optional[str] = None,
        target_role: Optional[str] = None,
        section_scores: Optional[Dict[str, Any]] = None,
        parsed_structure: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Queries Groq AI for semantic ATS feedback using openai/gpt-oss-120b.
        Guarantees that personal identifying information is never transmitted to the provider.
        Extracts required vs preferred qualifications, maps evidence, and generates actionable improvements.
        If AI times out, truncates, or is unavailable, falls back smoothly to deterministic analysis.
        """
        def get_fallback(reason: str) -> Dict[str, Any]:
            logger.warning(f"Groq ATS AI semantic analysis fallback activated. Reason: {reason}")
            scorer = AtsScorer()
            fallback_data = scorer.calculate_deterministic_semantics(
                text=sanitized_text,
                jd=jd,
                target_role=target_role,
                section_scores=section_scores,
                parsed_structure=parsed_structure
            )
            fallback_data["semantic_error"] = f"AI semantic analysis {reason}. Displaying local deterministic keyword match and rule-based recommendations."
            return fallback_data

        system_prompt = """You are an ATS Semantic Analyzer and expert technical recruiter.
Evaluate candidate qualifications against the exact job description accurately, objectively, and strictly based on the evidence in the resume.

EVALUATION RULES:
1. Extract required and preferred qualifications directly from the supplied job description.
2. Identify concrete resume evidence matching each qualification.
3. Distinguish explicitly listed skills from skills demonstrated through projects, experience, or measurable outcomes.
   For matched skills, cite brief evidence in parentheses, e.g. "Python (Experience: Built REST API)", "Docker (Project: Microservices deployment)".
4. Identify missing requirements (missing_required) and missing preferred skills (missing_preferred).
5. Highlight relevant project and work experience alignment in candidate strengths.
6. Generate specific, practical, actionable improvement recommendations tailored to this exact job.
7. NEVER invent skills, qualifications, experience, certifications, or achievements.
8. NEVER assume a missing qualification is present if not stated in the resume.
9. Personal identifying information (name, address, email, phone) has been redacted for privacy. NEVER penalize the candidate or treat absent personal info as an ATS deficiency.
10. Return ONLY a valid JSON object matching this schema without markdown fences, explanation, or commentary:
{
  "matched_keywords": [],
  "missing_keywords": [],
  "missing_required": [],
  "missing_preferred": [],
  "strengths": [],
  "recommendations": []
}

STRICT SCHEMA LIMITS:
- matched_keywords: maximum 8 items with brief evidence citation
- missing_keywords: maximum 8 items (required or preferred skills from the job description not found in resume)
- missing_required: maximum 5 mandatory skills from the job description not found in resume
- missing_preferred: maximum 5 nice-to-have or preferred skills from the job description not found in resume
- strengths: maximum 3 items (maximum 120 characters each, highlighting demonstrated qualifications)
- recommendations: maximum 4 items (maximum 140 characters each, specific actionable improvements for this job)
- If no job description is provided, missing_keywords, missing_required, and missing_preferred must be [].
- Output pure JSON without markdown fences, explanation, or commentary."""

        # Input preservation with generous safety boundaries
        compact_text = self._compact_resume_text(sanitized_text, max_chars=8000)
        compact_jd = self._compact_job_description(jd, max_chars=6000)
        bounded_role = target_role[:100].strip() if target_role else None

        user_prompt = f"--- START SANITIZED RESUME TEXT ---\n{compact_text}\n--- END SANITIZED RESUME TEXT ---"
        if bounded_role:
            user_prompt += f"\n\n--- TARGET ROLE ---\n{bounded_role}\n--- END TARGET ROLE ---"
        if compact_jd and compact_jd.strip():
            user_prompt += f"\n\n--- START JOB DESCRIPTION ---\n{compact_jd}\n--- END JOB DESCRIPTION ---"

        # FINAL PRIVACY BOUNDARY VERIFICATION
        # Verify that no personal information or tokens can ever reach Groq
        is_safe, safe_user_prompt = self.privacy_redactor.verify_request_safety(user_prompt)
        if not is_safe:
            logger.warning("Privacy boundary detected potential identifier; applying deep scrub before provider request.")
            safe_user_prompt = self.privacy_redactor.redact_text(safe_user_prompt)

        timeout_val = float(getattr(settings, "ATS_AI_TIMEOUT_SECONDS", 35.0))
        if isinstance(self.provider, XAIGrokAIProvider):
            selected_model = getattr(settings, "ATS_GROK_MODEL", "grok-2-latest")
        elif isinstance(self.provider, GroqAIProvider):
            selected_model = getattr(settings, "ATS_GROQ_MODEL", "openai/gpt-oss-120b")
        else:
            selected_model = getattr(settings, "ATS_GROK_MODEL", "grok-2-latest")

        req = AIRequest(
            system_prompt=system_prompt,
            user_prompt=safe_user_prompt,
            temperature=0.1,
            max_tokens=1600,
            model_id=selected_model,
            timeout=timeout_val
        )

        try:
            response = await self.provider.generate(req)

            # Detect truncation by provider
            if response.finish_reason in ["length", "max_tokens", "truncated", "incomplete"]:
                logger.warning(f"ATS AI response was truncated by provider (finish_reason={response.finish_reason}).")
                return get_fallback("was truncated by provider")

            content = (response.content or "").strip()

            if not content:
                logger.warning("ATS AI response contained empty content.")
                return get_fallback("returned empty content")

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            elif "{" in content:
                content = content[content.find("{"):]
            else:
                return get_fallback("returned no valid JSON structure")

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"ATS AI JSONDecodeError. Content snippet: {content[:200]}")
                return get_fallback("returned malformed JSON")

            # If no JD and no target role, force missing keywords to empty
            if (not jd or not jd.strip()) and (not target_role or not target_role.strip()):
                matched = parsed.get("matched_keywords", [])
                missing = []
                missing_req = []
                missing_pref = []
            else:
                matched = parsed.get("matched_keywords", [])
                missing = parsed.get("missing_keywords", [])
                missing_req = parsed.get("missing_required", [])
                missing_pref = parsed.get("missing_preferred", [])

                # If missing_keywords is empty but separated lists exist, combine them
                if not missing and (missing_req or missing_pref):
                    missing = [f"[Required] {s}" for s in missing_req] + [f"[Preferred] {s}" for s in missing_pref]

            strengths = parsed.get("strengths", [])
            recommendations = parsed.get("recommendations", [])

            # Enforce Server-Side Limits & String length
            def truncate_list(lst, max_items, max_len):
                if not isinstance(lst, list):
                    return []
                limited = lst[:max_items]
                return [str(item)[:max_len] for item in limited if item]

            # If matched items contain evidence in parentheses, provide both clean skill and evidence
            clean_matched = [re.sub(r'\s*\(.*?\)', '', str(m)).strip() for m in matched if m]
            matched_evidence = [str(m) for m in matched if m]

            return {
                "semantic_analysis_available": True,
                "semantic_error": None,
                "analysis_mode": "ai",
                "matched_keywords": truncate_list(clean_matched, 8, 80),
                "matched_with_evidence": truncate_list(matched_evidence, 8, 120),
                "missing_keywords": truncate_list(missing, 8, 100),
                "missing_required": truncate_list(missing_req, 5, 80),
                "missing_preferred": truncate_list(missing_pref, 5, 80),
                "strengths": truncate_list(strengths, 3, 120),
                "recommendations": truncate_list(recommendations, 5, 160)
            }
        except (ProviderTimeoutError, ProviderUnavailableError) as e:
            logger.error(f"ATS AI Provider Error: {str(e)}")
            return get_fallback("timed out or was temporarily unavailable")
        except Exception as e:
            logger.error(f"ATS AI Parsing/Unknown Error: {str(e)}")
            return get_fallback("encountered an unexpected error")
