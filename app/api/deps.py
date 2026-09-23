from fastapi import Depends, HTTPException, status, Request
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

def get_current_user_id(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Validates the Supabase JWT securely on the backend using dynamic algorithm selection.
    Returns the user's UUID string.
    """
    token = credentials.credentials
    
    try:
        # Decode header safely
        unverified_headers = jwt.get_unverified_header(token)
        alg = unverified_headers.get('alg', 'unknown')
        kid = unverified_headers.get('kid', 'none')
        typ = unverified_headers.get('typ', 'unknown')
        
        # User requested explicit EXACT safe diagnostics:
        logger.info("AUTH PATH:")
        logger.info(f"endpoint={request.url.path}")
        logger.info(f"verifier={'SUPABASE_LEGACY_HS256' if alg == 'HS256' else 'SUPABASE_JWKS'}")
        logger.info(f"algorithm={alg}")
        
        logger.info("JWT HEADER:")
        logger.info(f"alg={alg}")
        logger.info(f"kid={kid}")
        logger.info(f"typ={typ}")
        
        # Dynamic Verification Branching
        if alg == 'HS256':
            # CASE A: Legacy Project Configuration
            payload = jwt.decode(
                token, 
                settings.SUPABASE_JWT_SECRET, 
                algorithms=["HS256"],
                options={"verify_aud": False}
            )
        elif alg in ['ES256', 'RS256']:
            # CASE B: Modern JWKS Asymmetric Verification
            jwks = get_jwks()
            payload = jwt.decode(
                token, 
                jwks, 
                algorithms=[alg],
                options={"verify_aud": False},
                issuer=f"{settings.SUPABASE_URL}/auth/v1"
            )
        else:
            logger.warning(f"[AUTH DIAGNOSTIC] Unsupported algorithm: {alg}")
            raise JWTError("Unsupported JWT algorithm")
            
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials: No sub claim",
            )
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
