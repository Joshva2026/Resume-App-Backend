from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.routers import ai, ats, chat

load_dotenv()

app = FastAPI(
    title="ResumeForge AI Backend",
    description="Backend for AI Resume Builder and ATS Analyzer",
    version="1.0.0"
)

app.include_router(ai.router)
app.include_router(ats.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"status": "ok", "service": "ResumeForge AI Backend"}

@app.get("/health")
def health_check():
    from app.core.config import settings
    ai_configured = bool(settings.NVIDIA_API_KEY)
    return {
        "status": "ok", 
        "message": "Backend is running securely.",
        "ai_provider": "nvidia" if ai_configured else "none",
        "configured": ai_configured
    }
