"""
Shared dependencies for API endpoints.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Optional, Any
import jwt
from datetime import datetime, timedelta

from src.config import settings

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# App state dependency
async def get_app_state():
    """Get the application state from the FastAPI app"""
    from src.main import app_state
    return app_state

# User authentication dependencies
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Get the current authenticated user from the JWT token.
    
    Args:
        token: JWT token from the Authorization header
    
    Returns:
        Dict containing user information
    
    Raises:
        HTTPException: If the token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        
        # Extract user ID from token
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Check token expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
            raise credentials_exception
        
        # Return user information
        return {
            "user_id": user_id,
            "username": payload.get("username", ""),
            "email": payload.get("email", ""),
            "is_admin": payload.get("is_admin", False),
            "exp": exp
        }
    except jwt.PyJWTError:
        raise credentials_exception

async def get_admin_user(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the current user and verify they have admin privileges.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        Dict containing admin user information
    
    Raises:
        HTTPException: If the user is not an admin
    """
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action"
        )
    
    return current_user
