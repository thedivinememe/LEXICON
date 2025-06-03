"""
Spherical Relationship Vectorizer.

This module provides a vectorizer for concepts in the spherical universe,
allowing for the creation of concept vectors based on relationships
and constraints.
"""

import asyncio
import logging
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Set, Union

from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager

logger = logging.getLogger(__name__)

class SphericalRelationshipVectorizer:
    """
    Vectorizer for concepts in the spherical universe.
    
    Creates concept vectors based on relationships and constraints.
    """
    
    def __init__(self, 
                universe: Optional[BlochSphereUniverse] = None,
                null_gradient: Optional[NullGradientManager] = None):
        """
        Initialize the vectorizer.
        
        Args:
            universe: Spherical universe
            null_gradient: Null gradient manager
        """
        self.universe = universe if universe else BlochSphereUniverse()
        self.null_gradient = null_gradient if null_gradient else NullGradientManager(self.universe)
    
    async def vectorize_concept(self, concept: str) -> SphericalCoordinate:
        """
        Vectorize a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            Spherical coordinate for the concept
        """
        logger.info(f"Vectorizing concept '{concept}'")
        
        # Check if concept already exists
        if concept in self.universe.concepts:
            logger.info(f"Concept '{concept}' already exists")
            return self.universe.get_concept_position(concept)
        
        # Generate random position
        r = random.uniform(0.2, 0.4)
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        
        position = SphericalCoordinate(r=r, theta=theta, phi=phi)
        
        # Add concept to universe
        self.universe.add_concept(concept, position)
        
        logger.info(f"Vectorized concept '{concept}' to position {position}")
        
        return position
    
    async def vectorize_concepts(self, concepts: List[str]) -> Dict[str, SphericalCoordinate]:
        """
        Vectorize multiple concepts.
        
        Args:
            concepts: List of concept names
            
        Returns:
            Dictionary mapping concept names to positions
        """
        logger.info(f"Vectorizing {len(concepts)} concepts")
        
        positions = {}
        
        for concept in concepts:
            position = await self.vectorize_concept(concept)
            positions[concept] = position
        
        return positions
    
    async def vectorize_with_constraints(self, 
                                       concept: str,
                                       constraints: Dict[str, Any]) -> SphericalCoordinate:
        """
        Vectorize a concept with constraints.
        
        Args:
            concept: Concept name
            constraints: Dictionary of constraints
                - min_radius: Minimum radius
                - max_radius: Maximum radius
                - near_concept: Concept to be near
                - max_angle: Maximum angle from near_concept
                - opposite_concept: Concept to be opposite to
                - relationship_type: Relationship type with near_concept
            
        Returns:
            Spherical coordinate for the concept
        """
        logger.info(f"Vectorizing concept '{concept}' with constraints")
        
        # Check if concept already exists
        if concept in self.universe.concepts:
            logger.info(f"Concept '{concept}' already exists")
            return self.universe.get_concept_position(concept)
        
        # Get constraints
        min_radius = constraints.get("min_radius", 0.2)
        max_radius = constraints.get("max_radius", 0.4)
        near_concept = constraints.get("near_concept")
        max_angle = constraints.get("max_angle")
        opposite_concept = constraints.get("opposite_concept")
        relationship_type = constraints.get("relationship_type")
        
        # Generate position based on constraints
        if near_concept and opposite_concept:
            # Can't be near and opposite at the same time
            logger.warning("Can't be near and opposite at the same time, ignoring opposite_concept")
            opposite_concept = None
        
        if near_concept and near_concept in self.universe.concepts:
            # Position near another concept
            near_position = self.universe.get_concept_position(near_concept)
            
            # Generate random angle
            if max_angle is None:
                max_angle = np.pi / 4  # Default to 45 degrees
            
            # Generate random direction
            random_direction = np.random.randn(3)
            random_direction = random_direction / np.linalg.norm(random_direction)
            
            # Convert near position to Cartesian
            near_cart = near_position.to_cartesian()
            
            # Make random direction orthogonal to near_cart
            random_direction = random_direction - np.dot(random_direction, near_cart) * near_cart
            random_direction = random_direction / np.linalg.norm(random_direction)
            
            # Generate random angle within max_angle
            angle = random.uniform(0, max_angle)
            
            # Rotate near_cart towards random_direction by angle
            rotation_axis = np.cross(near_cart, random_direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # Rodrigues rotation formula
            rotated = near_cart * np.cos(angle) + np.cross(rotation_axis, near_cart) * np.sin(angle)
            
            # Generate random radius
            r = random.uniform(min_radius, max_radius)
            
            # Scale to radius
            cart = rotated * r / np.linalg.norm(rotated)
            
            # Convert to spherical
            position = SphericalCoordinate.from_cartesian(cart)
            
        elif opposite_concept and opposite_concept in self.universe.concepts:
            # Position opposite to another concept
            opposite_position = self.universe.get_concept_position(opposite_concept)
            
            # Generate position in opposite direction
            cart = -opposite_position.to_cartesian()
            
            # Generate random radius
            r = random.uniform(min_radius, max_radius)
            
            # Scale to radius
            cart = cart * r / np.linalg.norm(cart)
            
            # Convert to spherical
            position = SphericalCoordinate.from_cartesian(cart)
            
        else:
            # Generate random position
            r = random.uniform(min_radius, max_radius)
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            
            position = SphericalCoordinate(r=r, theta=theta, phi=phi)
        
        # Add concept to universe
        self.universe.add_concept(concept, position)
        
        # Add relationship if specified
        if near_concept and relationship_type:
            self.universe.add_relationship(concept, near_concept, relationship_type)
        
        logger.info(f"Vectorized concept '{concept}' to position {position}")
        
        return position
    
    async def get_nearest_concepts(self, 
                                 position: SphericalCoordinate,
                                 count: int = 5) -> List[Tuple[str, float]]:
        """
        Get the nearest concepts to a position.
        
        Args:
            position: Position to check
            count: Number of nearest concepts to return
            
        Returns:
            List of (concept_name, distance) tuples, sorted by distance
        """
        logger.info(f"Getting nearest concepts to position {position}")
        
        # Calculate distances to all concepts
        distances = []
        
        for concept, concept_position in self.universe.concepts.items():
            distance = self.universe.calculate_angular_distance(position, concept_position)
            distances.append((concept, distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Return top count
        return distances[:count]
    
    async def interpolate_concepts(self, 
                                 concept1: str,
                                 concept2: str,
                                 weight: float = 0.5) -> SphericalCoordinate:
        """
        Interpolate between two concepts.
        
        Args:
            concept1: First concept name
            concept2: Second concept name
            weight: Interpolation weight (0 = concept1, 1 = concept2)
            
        Returns:
            Interpolated position
            
        Raises:
            ValueError: If concepts not found
        """
        logger.info(f"Interpolating between '{concept1}' and '{concept2}' with weight {weight}")
        
        # Check if concepts exist
        if concept1 not in self.universe.concepts:
            raise ValueError(f"Concept '{concept1}' not found")
        
        if concept2 not in self.universe.concepts:
            raise ValueError(f"Concept '{concept2}' not found")
        
        # Get positions
        pos1 = self.universe.get_concept_position(concept1)
        pos2 = self.universe.get_concept_position(concept2)
        
        # Convert to Cartesian
        cart1 = pos1.to_cartesian()
        cart2 = pos2.to_cartesian()
        
        # Interpolate
        interp_cart = cart1 * (1 - weight) + cart2 * weight
        
        # Normalize to sphere surface
        norm = np.linalg.norm(interp_cart)
        if norm > 0:
            # Use average radius
            avg_r = pos1.r * (1 - weight) + pos2.r * weight
            interp_cart = interp_cart / norm * avg_r
        
        # Convert back to spherical
        interp_pos = SphericalCoordinate.from_cartesian(interp_cart)
        
        logger.info(f"Interpolated position: {interp_pos}")
        
        return interp_pos
    
    async def find_concept_by_position(self, 
                                     position: SphericalCoordinate,
                                     max_distance: float = 0.1) -> Optional[str]:
        """
        Find a concept near a position.
        
        Args:
            position: Position to check
            max_distance: Maximum angular distance
            
        Returns:
            Concept name, or None if no concept found
        """
        logger.info(f"Finding concept near position {position}")
        
        # Get nearest concepts
        nearest = await self.get_nearest_concepts(position, 1)
        
        if nearest and nearest[0][1] <= max_distance:
            return nearest[0][0]
        
        return None
    
    async def create_concept_from_interpolation(self, 
                                             concept1: str,
                                             concept2: str,
                                             weight: float = 0.5,
                                             new_concept: Optional[str] = None) -> Tuple[str, SphericalCoordinate]:
        """
        Create a new concept from interpolation.
        
        Args:
            concept1: First concept name
            concept2: Second concept name
            weight: Interpolation weight (0 = concept1, 1 = concept2)
            new_concept: Optional name for the new concept
            
        Returns:
            Tuple of (concept_name, position)
            
        Raises:
            ValueError: If concepts not found
        """
        logger.info(f"Creating concept from interpolation of '{concept1}' and '{concept2}'")
        
        # Generate interpolated position
        position = await self.interpolate_concepts(concept1, concept2, weight)
        
        # Generate name if not provided
        if new_concept is None:
            new_concept = f"{concept1}_{concept2}_{weight:.2f}"
        
        # Add concept
        self.universe.add_concept(new_concept, position)
        
        # Add relationships
        self.universe.add_relationship(new_concept, concept1, "and")
        self.universe.add_relationship(new_concept, concept2, "and")
        
        logger.info(f"Created concept '{new_concept}' at position {position}")
        
        return new_concept, position
    
    async def create_opposite_concept(self, 
                                    concept: str,
                                    opposite_name: Optional[str] = None) -> Tuple[str, SphericalCoordinate]:
        """
        Create a concept opposite to another.
        
        Args:
            concept: Concept name
            opposite_name: Optional name for the opposite concept
            
        Returns:
            Tuple of (concept_name, position)
            
        Raises:
            ValueError: If concept not found
        """
        logger.info(f"Creating opposite concept for '{concept}'")
        
        # Check if concept exists
        if concept not in self.universe.concepts:
            raise ValueError(f"Concept '{concept}' not found")
        
        # Get position
        position = self.universe.get_concept_position(concept)
        
        # Create opposite position
        cart = position.to_cartesian()
        opposite_cart = -cart
        
        # Convert to spherical
        opposite_position = SphericalCoordinate.from_cartesian(opposite_cart)
        
        # Generate name if not provided
        if opposite_name is None:
            opposite_name = f"not_{concept}"
        
        # Add concept
        self.universe.add_concept(opposite_name, opposite_position)
        
        # Add relationship
        self.universe.add_relationship(opposite_name, concept, "not")
        
        logger.info(f"Created opposite concept '{opposite_name}' at position {opposite_position}")
        
        return opposite_name, opposite_position
