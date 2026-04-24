"""
Web Package

FastAPI web application and supporting modules for the Password Guesser
penetration testing framework.

Modules:
- app: Main FastAPI application
- auth: Authentication and authorization
- tasks: Async task management
- websocket: WebSocket support
- pentest_api: Penetration testing API endpoints
- rate_limit: Rate limiting middleware
"""

from web.app import app, state
from web.auth import APIKeyAuth, JWTAuth

__all__ = ["app", "state", "APIKeyAuth", "JWTAuth"]
