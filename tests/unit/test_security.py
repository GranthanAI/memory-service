"""
tests/unit/test_security.py

Unit tests for service-to-service security (app/core/security.py).
Verifies:
  - Base64url encoding and decoding.
  - HS256 JWT token generation.
  - JWT signature verification and payload decoding.
  - Expired JWT token rejection.
  - Tampered JWT token signature rejection.
  - FastAPI verify_service_auth dependency verification.
"""

import time
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core.security import (
    base64url_encode,
    base64url_decode,
    generate_jwt,
    verify_jwt,
    verify_service_auth,
)


def test_base64url_helpers():
    """Verify base64url_encode and base64url_decode are correct and padding-robust."""
    test_bytes = b"hello world!!"
    encoded = base64url_encode(test_bytes)
    assert "=" not in encoded
    decoded = base64url_decode(encoded)
    assert decoded == test_bytes


def test_generate_and_verify_jwt_success():
    """Verify that a generated JWT is successfully decoded and matches the payload."""
    payload = {"sub": "test-service", "iss": "graphgpt"}
    secret = "my-secret-key"
    
    token = generate_jwt(payload, secret, expires_in=60)
    decoded = verify_jwt(token, secret)
    
    assert decoded is not None
    assert decoded["sub"] == "test-service"
    assert decoded["iss"] == "graphgpt"
    assert "exp" in decoded


def test_verify_jwt_invalid_signature():
    """Verify that an altered signature or wrong secret returns None."""
    payload = {"sub": "test-service"}
    secret = "correct-secret"
    
    token = generate_jwt(payload, secret)
    
    # 1. Decode with wrong secret
    assert verify_jwt(token, "wrong-secret") is None
    
    # 2. Alter payload segment slightly
    parts = token.split(".")
    altered_payload = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
    altered_token = f"{parts[0]}.{altered_payload}.{parts[2]}"
    assert verify_jwt(altered_token, secret) is None


def test_verify_jwt_expired():
    """Verify that expired JWT tokens are rejected."""
    payload = {"sub": "test-service"}
    secret = "secret"
    
    # Generate token with negative expiry (already expired)
    token = generate_jwt(payload, secret, expires_in=-10)
    assert verify_jwt(token, secret) is None


@pytest.mark.asyncio
async def test_verify_service_auth_api_key():
    """Verify that verify_service_auth permits valid API keys."""
    # API key matching settings.API_KEY
    res = await verify_service_auth(api_key="graphgpt-memory-secret")
    assert res["method"] == "api_key"
    assert res["identity"] == "internal_service"


@pytest.mark.asyncio
async def test_verify_service_auth_jwt():
    """Verify that verify_service_auth permits valid bearer JWT tokens."""
    token = generate_jwt({"sub": "conversation-service"}, secret_key="graphgpt-jwt-secret")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    res = await verify_service_auth(api_key=None, bearer=credentials)
    assert res["method"] == "jwt"
    assert res["identity"] == "conversation-service"


@pytest.mark.asyncio
async def test_verify_service_auth_unauthorized():
    """Verify that verify_service_auth raises 401 when both inputs are invalid/missing."""
    with pytest.raises(HTTPException) as exc:
        await verify_service_auth(api_key="invalid-key", bearer=None)
    assert exc.value.status_code == 401
    
    with pytest.raises(HTTPException) as exc:
        await verify_service_auth(api_key=None, bearer=None)
    assert exc.value.status_code == 401
