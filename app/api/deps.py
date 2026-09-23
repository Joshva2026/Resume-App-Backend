from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.core.config import settings
from supabase import create_client, Client

security = HTTPBearer()

def get_supabase_client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

import logging
logger = logging.getLogger(__name__)

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Validates the Supabase JWT securely on the backend using the JWT secret.
    Returns the user's UUID string.
    """
    token = credentials.credentials
    logger.info(f"[AUTH DIAGNOSTIC] authorization header present = True")
    logger.info(f"[AUTH DIAGNOSTIC] scheme = {credentials.scheme}")
    
    try:
        # Decode header safely
        unverified_headers = jwt.get_unverified_header(token)
        logger.info(f"[AUTH DIAGNOSTIC] JWT decoded header algorithm = {unverified_headers.get('alg', 'unknown')}")
        
        # Decode payload safely to check standard claims before verification
        unverified_claims = jwt.get_unverified_claims(token)
        logger.info(f"[AUTH DIAGNOSTIC] JWT issuer = {unverified_claims.get('iss', 'unknown')}")
        logger.info(f"[AUTH DIAGNOSTIC] JWT audience = {unverified_claims.get('aud', 'unknown')}")
        logger.info(f"[AUTH DIAGNOSTIC] JWT subject/user ID = {unverified_claims.get('sub', 'unknown')}")
        
        # Supabase uses HS256 for its JWT tokens signed with the JWT Secret
        payload = jwt.decode(
            token, 
            settings.SUPABASE_JWT_SECRET, 
            algorithms=["HS256"],
            options={"verify_aud": False} # Default Supabase audience is 'authenticated'
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials: No sub claim",
            )
        logger.info(f"[AUTH DIAGNOSTIC] token expiration status = VALID")
        return user_id
    except jwt.ExpiredSignatureError:
        logger.warning(f"[AUTH DIAGNOSTIC] token expiration status = EXPIRED")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        logger.warning(f"[AUTH DIAGNOSTIC] JWT verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
