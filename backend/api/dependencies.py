"""
API Dependencies
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional

from .config import settings
from ..config.database import get_database_session

# Security scheme
security = HTTPBearer()

def get_db() -> Session:
    """Get database session"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current authenticated user"""
    # TODO: Implement JWT token validation
    # For now, return a mock user
    return {
        "id": 1,
        "username": "admin",
        "email": "admin@openpolicy.com",
        "role": "admin"
    }

def require_admin(current_user = Depends(get_current_user)):
    """Require admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user
