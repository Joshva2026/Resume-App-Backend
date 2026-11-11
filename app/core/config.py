from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_JWT_SECRET: str
    
    # NVIDIA Config
    NVIDIA_API_KEY: Optional[str] = None
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "openai/gpt-oss-20b"
    NVIDIA_FALLBACK_MODEL: Optional[str] = "nvidia/nemotron-3.5-lightning-30b-a3b"
    
    # AI Limits
    AI_RATE_LIMIT_PER_MINUTE: int = 10
    AI_CHAT_RATE_LIMIT_PER_MINUTE: int = 20
    AI_MAX_CONCURRENT_REQUESTS: int = 1
    AI_REQUEST_TIMEOUT_SECONDS: int = 70
    AI_CONNECT_TIMEOUT_SECONDS: float = 10.0
    AI_FIRST_TOKEN_TIMEOUT_SECONDS: float = 30.0
    AI_FIRST_TOKEN_TIMEOUT_MAX_SECONDS: float = 60.0
    AI_STREAM_READ_TIMEOUT_SECONDS: Optional[float] = 100.0
    AI_STREAM_READ_TIMEOUT_MAX_SECONDS: float = 105.0
    AI_INTER_CHUNK_TIMEOUT_SECONDS: Optional[float] = 30.0
    AI_STREAM_KEEPALIVE_INTERVAL_SECONDS: int = 15
    AI_STREAM_429_MAX_RETRIES: int = 2
    AI_PROVIDER_RETRY_MAX_WAIT_SECONDS: float = 8.0
    AI_MAX_TOKENS: int = 1536
    ATS_AI_TIMEOUT_SECONDS: int = 35
    
    # xAI Grok Config (Dedicated for ATS Analysis)
    XAI_API_KEY: Optional[str] = None
    XAI_BASE_URL: str = "https://api.x.ai/v1"
    ATS_GROK_MODEL: str = "grok-2-latest"
    
    # Groq Config (Dedicated for ATS Analysis)
    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    ATS_GROQ_MODEL: str = "openai/gpt-oss-120b"

    # CORS Origins (comma-separated list)
    ALLOWED_ORIGINS: str = ""
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
