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

@app.on_event("startup")
async def startup_event():
    from app.core.config import settings
    
    # Check if pydantic fix is loaded
    pydantic_loaded = False
    try:
        from app.schemas.ai import AIRequest
        pydantic_loaded = hasattr(AIRequest, 'model_config') and 'protected_namespaces' in AIRequest.model_config
    except Exception:
        pass
        
    print("=== STARTUP DIAGNOSTIC ===", flush=True)
    print("COMMIT / SOURCE: 14f3059 (or latest)", flush=True)
    print(f"SUPABASE_URL_CONFIGURED: {'true' if settings.SUPABASE_URL else 'false'}", flush=True)
    print(f"SUPABASE_JWT_SECRET_CONFIGURED: {'true' if settings.SUPABASE_JWT_SECRET else 'false'}", flush=True)
    print(f"SUPABASE_KEY_CONFIGURED: {'true' if settings.SUPABASE_KEY else 'false'}", flush=True)
    print("JWT_ALGORITHM: HS256", flush=True)
    print("DIAGNOSTIC_EXECUTED: true", flush=True)
    print(f"PYDANTIC_FIX_LOADED: {'true' if pydantic_loaded else 'false'}", flush=True)
    print("=== END STARTUP DIAGNOSTIC ===", flush=True)

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
