import hashlib
import time
from typing import Dict, Tuple, Optional
from app.schemas.ats import AtsReport

class AtsAnalysisCache:
    """
    In-memory isolated cache for ATS reports.
    Invalidates immediately whenever resume content, job description,
    target role, scoring version, or user changes.
    """
    _cache: Dict[str, Tuple[float, AtsReport]] = {}
    TTL_SECONDS = 600  # 10 minutes
    MAX_ENTRIES = 500

    @classmethod
    def make_key(
        cls,
        user_id: str,
        resume_text: str,
        job_description: Optional[str],
        target_role: Optional[str],
        scoring_version: str
    ) -> str:
        resume_fp = hashlib.sha256(resume_text.strip().encode("utf-8")).hexdigest()
        jd_fp = hashlib.sha256((job_description or "").strip().encode("utf-8")).hexdigest()
        role_fp = hashlib.sha256((target_role or "").strip().lower().encode("utf-8")).hexdigest()
        raw = f"{user_id}:{resume_fp}:{jd_fp}:{role_fp}:{scoring_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def get(cls, key: str) -> Optional[AtsReport]:
        now = time.time()
        if key in cls._cache:
            ts, report = cls._cache[key]
            if now - ts <= cls.TTL_SECONDS:
                return report
            else:
                del cls._cache[key]
        return None

    @classmethod
    def set(cls, key: str, report: AtsReport) -> None:
        if len(cls._cache) >= cls.MAX_ENTRIES:
            oldest_key = min(cls._cache.keys(), key=lambda k: cls._cache[k][0])
            del cls._cache[oldest_key]
        cls._cache[key] = (time.time(), report)

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()
