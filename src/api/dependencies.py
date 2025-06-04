"""
Shared dependencies for API endpoints.
"""

from fastapi import Depends, HTTPException, status
import asyncio
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Optional, Any
import jwt
from datetime import datetime, timedelta
import numpy as np

from src.config import settings
from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager
from src.core.centroid_builder import CentroidConceptBuilder
from src.core.relative_type_system import RelativeTypeSystem
from src.core.existence_types import ExistenceTypeRegistry
from src.core.existence_primitives import ExistencePrimitiveEngine
from src.neural.spherical_vectorizer import SphericalRelationshipVectorizer
from src.services.sphere_visualization import SphericalUniverseVisualizer
from src.llm.coree_interface import COREEInterface

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

# Spherical universe dependencies
async def get_universe() -> BlochSphereUniverse:
    """
    Get the BlochSphereUniverse instance.
    
    Returns:
        BlochSphereUniverse instance
    """
    app_state = await get_app_state()
    
    if not hasattr(app_state, "universe"):
        # Create universe if it doesn't exist
        app_state.universe = BlochSphereUniverse()
    
    return app_state.universe

async def get_null_gradient_manager() -> NullGradientManager:
    """
    Get the NullGradientManager instance.
    
    Returns:
        NullGradientManager instance
    """
    app_state = await get_app_state()
    universe = await get_universe()
    
    if not hasattr(app_state, "null_gradient_manager"):
        # Create null gradient manager if it doesn't exist
        app_state.null_gradient_manager = NullGradientManager(universe)
    
    return app_state.null_gradient_manager

async def get_centroid_builder() -> CentroidConceptBuilder:
    """
    Get the CentroidConceptBuilder instance.
    
    Returns:
        CentroidConceptBuilder instance
    """
    app_state = await get_app_state()
    universe = await get_universe()
    
    if not hasattr(app_state, "centroid_builder"):
        # Create centroid builder if it doesn't exist
        app_state.centroid_builder = CentroidConceptBuilder(universe)
    
    return app_state.centroid_builder

async def get_relative_type_system() -> RelativeTypeSystem:
    """
    Get the RelativeTypeSystem instance.
    
    Returns:
        RelativeTypeSystem instance
    """
    app_state = await get_app_state()
    universe = await get_universe()
    
    if not hasattr(app_state, "relative_type_system"):
        # Create relative type system if it doesn't exist
        app_state.relative_type_system = RelativeTypeSystem(universe)
    
    return app_state.relative_type_system

async def get_existence_type_system() -> ExistenceTypeRegistry:
    """
    Get the ExistenceTypeRegistry instance.
    
    Returns:
        ExistenceTypeRegistry instance
    """
    app_state = await get_app_state()
    
    if not hasattr(app_state, "existence_type_system"):
        # Create existence type system if it doesn't exist
        app_state.existence_type_system = ExistenceTypeRegistry()
    
    return app_state.existence_type_system

async def get_existence_primitive_engine() -> ExistencePrimitiveEngine:
    """
    Get the ExistencePrimitiveEngine instance.
    
    Returns:
        ExistencePrimitiveEngine instance
    """
    app_state = await get_app_state()
    
    if not hasattr(app_state, "existence_primitive_engine"):
        # Create existence primitive engine if it doesn't exist
        app_state.existence_primitive_engine = ExistencePrimitiveEngine()
    
    return app_state.existence_primitive_engine

async def get_spherical_vectorizer() -> SphericalRelationshipVectorizer:
    """
    Get the SphericalRelationshipVectorizer instance.
    
    Returns:
        SphericalRelationshipVectorizer instance
    """
    app_state = await get_app_state()
    universe = await get_universe()
    null_gradient = await get_null_gradient()
    
    if not hasattr(app_state, "spherical_vectorizer"):
        # Create spherical vectorizer if it doesn't exist
        app_state.spherical_vectorizer = SphericalRelationshipVectorizer(universe, null_gradient)
    
    return app_state.spherical_vectorizer

async def get_null_gradient() -> NullGradientManager:
    """
    Get the NullGradientManager instance.
    
    Returns:
        NullGradientManager instance
    """
    app_state = await get_app_state()
    
    if not hasattr(app_state, "null_gradient"):
        # Create null gradient if it doesn't exist
        app_state.null_gradient = NullGradientManager()
    
    return app_state.null_gradient

async def get_type_system() -> RelativeTypeSystem:
    """
    Get the RelativeTypeSystem instance.
    
    Returns:
        RelativeTypeSystem instance
    """
    app_state = await get_app_state()
    universe = await get_universe()
    null_gradient = await get_null_gradient()
    
    if not hasattr(app_state, "type_system"):
        # Create type system if it doesn't exist
        app_state.type_system = RelativeTypeSystem(universe, null_gradient)
    
    return app_state.type_system

async def get_visualizer() -> SphericalUniverseVisualizer:
    """
    Get the SphericalUniverseVisualizer instance.
    
    Returns:
        SphericalUniverseVisualizer instance
    """
    app_state = await get_app_state()
    universe = await get_universe()
    null_gradient = await get_null_gradient()
    vectorizer = await get_spherical_vectorizer()
    type_system = await get_type_system()
    
    if not hasattr(app_state, "visualizer"):
        # Create visualizer if it doesn't exist
        app_state.visualizer = SphericalUniverseVisualizer(universe, null_gradient, vectorizer, type_system)
    
    return app_state.visualizer

async def get_sphere_visualizer() -> SphericalUniverseVisualizer:
    """
    Get the SphericalUniverseVisualizer instance.
    
    Returns:
        SphericalUniverseVisualizer instance
    """
    return await get_visualizer()

async def get_coree_interface() -> COREEInterface:
    """
    Get the COREEInterface instance.
    
    Returns:
        COREEInterface instance
    """
    app_state = await get_app_state()
    
    if "coree_interface" not in app_state:
        # Create COREE interface if it doesn't exist
        app_state["coree_interface"] = COREEInterface()
        # Initialize in background to avoid blocking
        asyncio.create_task(app_state["coree_interface"].initialize())
    
    return app_state["coree_interface"]
