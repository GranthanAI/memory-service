"""
app/core/security.py

Service-to-Service Security and authentication handler.
Implements API Key validation and JWT (JSON Web Token) verification using only the Python Standard Library.
"""

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

logger = logging.getLogger("memory_service.core.security")

# Security schemes
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


def base64url_encode(data: bytes) -> str:
    """Helper to base64url-encode bytes data without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def base64url_decode(s: str) -> bytes:
    """Helper to decode base64url-encoded string, adding padding if needed."""
    padding = "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def generate_jwt(payload: Dict[str, Any], secret_key: str, expires_in: int = 3600) -> str:
    """
    Generates a signed HS256 JWT token using standard Python libraries.
    """
    header = {"alg": "HS256", "typ": "JWT"}
    full_payload = dict(payload)
    full_payload["exp"] = int(time.time()) + expires_in

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_json = json.dumps(full_payload, separators=(",", ":")).encode("utf-8")

    parts = base64url_encode(header_json) + "." + base64url_encode(payload_json)
    signature = hmac.new(
        secret_key.encode("utf-8"),
        parts.encode("utf-8"),
        hashlib.sha256
    ).digest()

    return parts + "." + base64url_encode(signature)


def verify_jwt(token: str, secret_key: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and verifies an HS256 JWT token using standard Python libraries.
    Returns the parsed payload if valid, None otherwise.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        
        header_segment, payload_segment, signature_segment = parts
        
        # 1. Re-calculate signature to verify payload integrity
        signing_input = (header_segment + "." + payload_segment).encode("utf-8")
        expected_sig = hmac.new(
            secret_key.encode("utf-8"),
            signing_input,
            hashlib.sha256
        ).digest()
        actual_sig = base64url_decode(signature_segment)
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.warning("JWT validation failed: signature mismatch")
            return None
            
        # 2. Decode payload and check expiration
        payload = json.loads(base64url_decode(payload_segment).decode("utf-8"))
        exp = payload.get("exp")
        if exp and exp < time.time():
            logger.warning("JWT validation failed: token has expired")
            return None
            
        return payload
    except Exception as e:
        logger.error(f"Error parsing/validating JWT token: {e}")
        return None


async def verify_service_auth(
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
) -> Dict[str, Any]:
    """
    FastAPI security dependency protecting internal endpoints.
    Allows request if a valid API Key (X-API-Key) or a valid HS256 Bearer JWT token is provided.
    Raises HTTP 401 if unauthorized.
    """
    # 1. Check API Key authentication
    if api_key and api_key == settings.API_KEY:
        return {"method": "api_key", "identity": "internal_service"}

    # 2. Check JWT Bearer token authentication
    if bearer:
        payload = verify_jwt(bearer.credentials, settings.JWT_SECRET_KEY)
        if payload:
            return {"method": "jwt", "payload": payload, "identity": payload.get("sub", "unknown")}

    logger.warning("Service-to-service authentication request failed: missing or invalid credentials")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized service access. Provide a valid X-API-Key or Bearer token.",
    )
