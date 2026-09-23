from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.core.config import settings
from supabase import create_client, Client

security = HTTPBearer()

def get_supabase_client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

import urllib.request
import json
import logging
logger = logging.getLogger(__name__)

_jwks = None

def get_jwks():
    global _jwks
    if _jwks is None:
        try:
            jwks_url = f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json"
            req = urllib.request.Request(jwks_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                _jwks = json.loads(response.read())
        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load authentication keys"
            )
    return _jwks

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Validates the Supabase JWT securely on the backend using the JWKS public keys.
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
        
        # Verify the token using the Supabase JWKS (ES256)
        jwks = get_jwks()
        payload = jwt.decode(
            token, 
            jwks, 
            algorithms=["ES256"],
            options={"verify_aud": False}, # Default Supabase audience is 'authenticated'
            issuer=f"{settings.SUPABASE_URL}/auth/v1"
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
