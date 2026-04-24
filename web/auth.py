"""
Authentication System for Web API

Provides API Key and JWT authentication.
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Set
import hashlib
import hmac
import os
import time
import secrets
import warnings


class APIKeyAuth:
    """API Key based authentication"""

    def __init__(self, api_keys: Set[str] = None):
        self.api_keys = api_keys or set()
        self.header_name = "X-API-Key"
        self.security = APIKeyHeader(name=self.header_name, auto_error=False)

    async def __call__(self, request: Request) -> Optional[str]:
        """Validate API Key from request header"""
        if not self.api_keys:
            return None  # No keys configured, skip validation

        api_key = await self.security(request)
        if api_key is None or api_key not in self.api_keys:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "API-Key"}
            )
        return api_key

    def add_key(self, key: str):
        """Add a new API key"""
        self.api_keys.add(key)

    def remove_key(self, key: str):
        """Remove an API key"""
        self.api_keys.discard(key)

    def generate_key(self) -> str:
        """Generate a new random API key"""
        return secrets.token_urlsafe(32)


class JWTAuth:
    """JWT Token based authentication using python-jose library."""

    # Supported algorithms (security best practice: whitelist allowed algorithms)
    SUPPORTED_ALGORITHMS = {"HS256", "HS384", "HS512"}

    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        expire_hours: int = 24
    ):
        # Validate algorithm (prevent algorithm confusion attacks)
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                             f"Supported: {self.SUPPORTED_ALGORITHMS}")

        # Use environment variable for secret key (security best practice)
        if secret_key is None:
            secret_key = os.environ.get("JWT_SECRET_KEY", "change-me-in-production")

        # Warn if using default secret key
        if secret_key == "change-me-in-production":
            warnings.warn(
                "SECURITY WARNING: Using default JWT secret key. "
                "Set the JWT_SECRET_KEY environment variable in production!",
                UserWarning
            )

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expire_hours = expire_hours

        # Initialize python-jose components
        from jose import jwt, JWTError
        self._jwt = jwt
        self._JWTError = JWTError

    def create_token(self, user_id: str, extra_claims: dict = None) -> str:
        """Create a JWT token for a user."""
        now = int(time.time())
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self.expire_hours * 3600,
        }
        if extra_claims:
            payload.update(extra_claims)

        return self._jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm
        )

    def verify_token(self, token: str) -> dict:
        """Verify a JWT token and return its payload."""
        try:
            # Decode and verify with algorithm whitelist
            payload = self._jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]  # Only allow configured algorithm
            )

            # Additional validation: ensure 'sub' claim exists
            if "sub" not in payload:
                raise ValueError("Token missing 'sub' claim")

            return payload

        except self._JWTError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}
            )

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[dict]:
        """FastAPI dependency to get current user from JWT"""
        if credentials is None:
            return None

        return self.verify_token(credentials.credentials)


# Password hashing with bcrypt (more secure than SHA256)
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)

    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its bcrypt hash."""
        try:
            return pwd_context.verify(password, hashed)
        except Exception:
            return False

except ImportError:
    # Fallback to SHA256 if bcrypt not available (not recommended for production)
    warnings.warn(
        "bcrypt not installed. Using SHA256 for password hashing (not recommended). "
        "Install passlib[bcrypt] for secure password hashing.",
        UserWarning
    )

    def hash_password(password: str) -> str:
        """Hash a password using SHA256 (fallback, less secure than bcrypt)."""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its SHA256 hash."""
        return hmac.compare_digest(hash_password(password), hashed)