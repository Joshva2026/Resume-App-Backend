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
    import logging
    from app.core.config import settings
    logger = logging.getLogger("startup_diagnostic")
    
    logger.info("=== STARTUP DIAGNOSTIC ===")
    
    # 1. SUPABASE_URL Check
    url = settings.SUPABASE_URL
    if url:
        hostname = url.split("//")[-1].split(".")[0]
        logger.info(f"SUPABASE_URL project reference: {hostname}")
    else:
        logger.info("SUPABASE_URL present: false")
        
    # 2. SUPABASE_KEY Check
    key = settings.SUPABASE_KEY
    logger.info(f"SUPABASE_KEY present: {bool(key)}")
    if key:
        # Detect type safely: Service role keys usually contain 'service_role' or start with 'sb_secret'
        if "service_role" in key or "sb_secret" in key:
            logger.info("SUPABASE_KEY format/type: Service Role Key (detected)")
        elif "anon" in key or "sb_anon" in key:
            logger.info("SUPABASE_KEY format/type: Anon Key (detected)")
        else:
            logger.info("SUPABASE_KEY format/type: Unknown Format")
            
    # 3. SUPABASE_JWT_SECRET Check
    secret = settings.SUPABASE_JWT_SECRET
    logger.info(f"SUPABASE_JWT_SECRET present: {bool(secret)}")
    
    # 4. JWT verification algorithm
    logger.info("JWT verification algorithm expected: HS256 (Legacy JWT Secret)")
    
    # 5. Pydantic Warning Check
    try:
        from app.schemas.ai import AIRequest
        logger.info(f"AIRequest protected_namespaces override present: {hasattr(AIRequest, 'model_config') and 'protected_namespaces' in AIRequest.model_config}")
    except Exception as e:
        logger.info(f"Failed to inspect AIRequest: {e}")

    logger.info("==========================")

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
