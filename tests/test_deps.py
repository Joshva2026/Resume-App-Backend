import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt, jwk
from app.api.deps import get_current_user_id
import app.api.deps as deps
from unittest.mock import patch, MagicMock

# Dummy secrets and keys for testing
DUMMY_SECRET = "supersecret_legacy_key"

# Actually generate a valid RSA keypair for testing ES256/RS256
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption()
)
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Convert to JWK dictionary format for python-jose
from jose import jwk
test_jwk = jwk.RSAKey(algorithm=jwk.ALGORITHMS.RS256, key=public_pem.decode('utf-8')).to_dict()
test_jwk["kid"] = "test-kid"

@pytest.fixture
def mock_jwks():
    with patch("app.api.deps.get_jwks", return_value={"keys": [test_jwk]}):
        yield

@pytest.fixture
def mock_settings():
    with patch("app.api.deps.settings") as mock_settings:
        mock_settings.SUPABASE_JWT_SECRET = DUMMY_SECRET
        mock_settings.SUPABASE_URL = "https://test.supabase.co"
        yield mock_settings

def create_mock_request():
    request = MagicMock()
    request.url.path = "/test"
    return request

def test_valid_hs256_token(mock_settings):
    token = jwt.encode({"sub": "user-123"}, DUMMY_SECRET, algorithm="HS256")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user_id = get_current_user_id(create_mock_request(), creds)
    assert user_id == "user-123"

def test_valid_rs256_token(mock_settings, mock_jwks):
    token = jwt.encode(
        {"sub": "user-123", "iss": "https://test.supabase.co/auth/v1"}, 
        private_pem.decode('utf-8'), 
        algorithm="RS256", 
        headers={"kid": "test-kid"}
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user_id = get_current_user_id(create_mock_request(), creds)
    assert user_id == "user-123"

def test_unsupported_algorithm(mock_settings):
    # Using HS512, which is unsupported
    token = jwt.encode({"sub": "user-123"}, DUMMY_SECRET, algorithm="HS512")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(create_mock_request(), creds)
    assert exc.value.status_code == 401

def test_invalid_hs256_signature(mock_settings):
    token = jwt.encode({"sub": "user-123"}, "wrong-secret", algorithm="HS256")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(create_mock_request(), creds)
    assert exc.value.status_code == 401

def test_invalid_rs256_signature(mock_settings, mock_jwks):
    wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    wrong_pem = wrong_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    token = jwt.encode(
        {"sub": "user-123", "iss": "https://test.supabase.co/auth/v1"}, 
        wrong_pem.decode('utf-8'), 
        algorithm="RS256", 
        headers={"kid": "test-kid"}
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(create_mock_request(), creds)
    assert exc.value.status_code == 401

def test_unknown_jwks_kid(mock_settings, mock_jwks):
    token = jwt.encode(
        {"sub": "user-123", "iss": "https://test.supabase.co/auth/v1"}, 
        private_pem.decode('utf-8'), 
        algorithm="RS256", 
        headers={"kid": "wrong-kid"}
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(create_mock_request(), creds)
    assert exc.value.status_code == 401

def test_expired_token(mock_settings):
    import time
    token = jwt.encode({"sub": "user-123", "exp": int(time.time()) - 3600}, DUMMY_SECRET, algorithm="HS256")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(create_mock_request(), creds)
    assert exc.value.status_code == 401
    assert "expired" in exc.value.detail.lower()

def test_wrong_issuer(mock_settings, mock_jwks):
    token = jwt.encode(
        {"sub": "user-123", "iss": "https://wrong.supabase.co/auth/v1"}, 
        private_pem.decode('utf-8'), 
        algorithm="RS256", 
        headers={"kid": "test-kid"}
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(create_mock_request(), creds)
    assert exc.value.status_code == 401
