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
    
    # AI Limits
    AI_RATE_LIMIT_PER_MINUTE: int = 10
    AI_REQUEST_TIMEOUT_SECONDS: int = 70
    AI_CONNECT_TIMEOUT_SECONDS: int = 10
    # Streaming read/idle timeout is disabled (None) to allow long generations to complete
    AI_STREAM_READ_TIMEOUT_SECONDS: Optional[int] = None
    AI_INTER_CHUNK_TIMEOUT_SECONDS: Optional[int] = None
    AI_STREAM_KEEPALIVE_INTERVAL_SECONDS: int = 15
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
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
