"""
Spherical Universe for the LEXICON system.
Defines the spherical coordinate system and universe.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
import numpy as np
import math

@dataclass
class SphericalCoordinate:
    """
    Spherical coordinate in the Bloch sphere.
    
    Attributes:
        r: Radius (0 to 0.5)
        theta: Azimuthal angle (0 to 2π)
        phi: Polar angle (0 to π)
    """
    r: float
    theta: float
    phi: float
    
    def to_cartesian(self) -> np.ndarray:
        """Convert to Cartesian coordinates"""
        x = self.r * np.sin(self.phi) * np.cos(self.theta)
        y = self.r * np.sin(self.phi) * np.sin(self.theta)
        z = self.r * np.cos(self.phi)
        return np.array([x, y, z])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        cart = self.to_cartesian()
        return {
            "r": float(self.r),
            "theta": float(self.theta),
            "phi": float(self.phi),
            "xyz": [float(cart[0]), float(cart[1]), float(cart[2])]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SphericalCoordinate':
        """Create from dictionary"""
        return cls(
            r=data["r"],
            theta=data["theta"],
            phi=data["phi"]
        )
    
    @classmethod
    def from_cartesian(cls, cart: np.ndarray) -> 'SphericalCoordinate':
        """Create from Cartesian coordinates"""
        x, y, z = cart
        
        # Calculate radius
        r = np.linalg.norm(cart)
        
        # Handle zero radius
        if r < 1e-6:
            return cls(r=0.0, theta=0.0, phi=0.0)
        
        # Calculate angles
        phi = np.arccos(z / r)
        theta = np.arctan2(y, x)
        
        # Ensure theta is in [0, 2π)
        if theta < 0:
            theta += 2 * np.pi
        
        return cls(r=r, theta=theta, phi=phi)

class BlochSphereUniverse:
    """
    Universe based on the Bloch sphere.
    
    The Bloch sphere is a representation of quantum states,
    but here it's used to represent concepts and their relationships.
    
    Properties:
    - Center (r=0) represents null/void/unity
    - Surface (r=0.5) represents fully defined concepts
    - Antipodal points represent negations
    - Angular distance represents relationship type
    """
    
    def __init__(self):
        self.concepts = {}  # name -> SphericalCoordinate
        self.relationships = {}  # (concept1, concept2) -> relationship_type
    
    def add_concept(self, name: str, position: SphericalCoordinate, add_negation: bool = False) -> None:
        """
        Add a concept to the universe.
        
        Args:
            name: Concept name
            position: Spherical position
            add_negation: Whether to add negation automatically
        """
        # Add concept
        self.concepts[name] = position
        
        # Add negation if requested
        if add_negation:
            # Create negation name
            negation_name = f"!{name}"
            
            # Create antipodal position
            antipodal = self.get_antipodal_point(position)
            
            # Add negation
            self.concepts[negation_name] = antipodal
            
            # Add relationships
            self.relationships[(name, negation_name)] = "not"
            self.relationships[(negation_name, name)] = "not"
    
    def get_concept_position(self, name: str) -> Optional[SphericalCoordinate]:
        """Get the position of a concept"""
        return self.concepts.get(name)
    
    def get_antipodal_point(self, position: SphericalCoordinate) -> SphericalCoordinate:
        """
        Get the antipodal point of a position.
        
        The antipodal point is on the opposite side of the sphere,
        representing the negation of a concept.
        """
        # Calculate antipodal angles
        theta_antipodal = (position.theta + np.pi) % (2 * np.pi)
        phi_antipodal = np.pi - position.phi
        
        # Create antipodal position
        return SphericalCoordinate(
            r=position.r,
            theta=theta_antipodal,
            phi=phi_antipodal
        )
    
    def calculate_angular_distance(self, pos1: SphericalCoordinate, pos2: SphericalCoordinate) -> float:
        """
        Calculate angular distance between two positions.
        
        The angular distance is the angle between the two positions,
        ignoring the radius.
        """
        # Special case for positions on the z-axis (phi=0)
        if pos1.phi == 0.0 and pos2.phi == 0.0:
            # For positions on the z-axis, the angular distance is the difference in theta
            # If theta is the same, the distance is 0 if both are on the same side of the z-axis,
            # or pi if they are on opposite sides
            if abs(pos1.theta - pos2.theta) < 1e-6 or abs(abs(pos1.theta - pos2.theta) - 2*np.pi) < 1e-6:
                # Same theta, check if they're on the same side of the z-axis
                if np.sign(np.cos(pos1.phi)) == np.sign(np.cos(pos2.phi)):
                    return 0.0
                else:
                    return np.pi
            else:
                # Different theta, the distance is pi/2
                return np.pi/2
        
        # Convert to Cartesian
        cart1 = pos1.to_cartesian()
        cart2 = pos2.to_cartesian()
        
        # Normalize to unit vectors
        norm1 = np.linalg.norm(cart1)
        norm2 = np.linalg.norm(cart2)
        
        if norm1 > 0:
            cart1 = cart1 / norm1
        if norm2 > 0:
            cart2 = cart2 / norm2
        
        # Calculate dot product
        dot_product = np.clip(np.dot(cart1, cart2), -1.0, 1.0)
        
        # Calculate angle
        angle = np.arccos(dot_product)
        
        return angle
    
    def get_relationship_type(self, concept1: str, concept2: str) -> Optional[str]:
        """
        Get the relationship type between two concepts.
        
        Args:
            concept1: First concept name
            concept2: Second concept name
            
        Returns:
            Relationship type (and, or, not), or None if no relationship exists
        """
        # Check if relationship exists in the dictionary
        return self.relationships.get((concept1, concept2))
    
    def add_relationship(self, concept1: str, concept2: str, rel_type: str) -> bool:
        """
        Add a relationship between two concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            rel_type: Relationship type (and, or, not)
            
        Returns:
            bool: Whether the relationship was added
        """
        # Check if concepts exist
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return False
        
        # Add relationship
        self.relationships[(concept1, concept2)] = rel_type.lower()
        
        return True
        
    def remove_relationship(self, concept1: str, concept2: str) -> bool:
        """
        Remove a relationship between two concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            bool: Whether the relationship was removed
        """
        # Check if relationship exists
        if (concept1, concept2) in self.relationships:
            del self.relationships[(concept1, concept2)]
            return True
        elif (concept2, concept1) in self.relationships:
            del self.relationships[(concept2, concept1)]
            return True
        
        return False
        
    def remove_concept(self, concept: str) -> bool:
        """
        Remove a concept from the universe.
        
        Args:
            concept: Concept name
            
        Returns:
            bool: Whether the concept was removed
        """
        # Check if concept exists
        if concept not in self.concepts:
            return False
        
        # Remove concept
        del self.concepts[concept]
        
        # Remove relationships involving the concept
        to_remove = []
        for (c1, c2) in self.relationships:
            if c1 == concept or c2 == concept:
                to_remove.append((c1, c2))
        
        for rel in to_remove:
            del self.relationships[rel]
        
        return True
    
    def get_relationship(self, concept1: str, concept2: str) -> Optional[str]:
        """Get the relationship between two concepts"""
        return self.relationships.get((concept1, concept2))
    
    def get_related_concepts(self, concept: str, rel_type: Optional[str] = None) -> List[str]:
        """
        Get concepts related to a concept.
        
        Args:
            concept: Concept name
            rel_type: Optional relationship type filter
            
        Returns:
            List[str]: List of related concepts
        """
        # Check if concept exists
        if concept not in self.concepts:
            return []
        
        # Find related concepts
        related = []
        
        for (c1, c2), r_type in self.relationships.items():
            if c1 == concept and (rel_type is None or r_type == rel_type):
                related.append(c2)
        
        return related
    
    def get_concept_cluster(self, concept: str, max_distance: float = np.radians(45)) -> List[str]:
        """
        Get a cluster of concepts around a concept.
        
        Args:
            concept: Center concept
            max_distance: Maximum angular distance
            
        Returns:
            List[str]: List of concepts in the cluster
        """
        # Check if concept exists
        if concept not in self.concepts:
            return []
        
        # Get center position
        center_pos = self.concepts[concept]
        
        # Find concepts within max_distance
        cluster = [concept]  # Include the center concept itself
        
        for name, pos in self.concepts.items():
            if name == concept:
                continue
            
            # Calculate angular distance
            distance = self.calculate_angular_distance(center_pos, pos)
            
            # Check if within max_distance
            if distance <= max_distance:
                cluster.append(name)
        
        return cluster
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "concepts": {
                name: pos.to_dict()
                for name, pos in self.concepts.items()
            },
            "relationships": {
                f"{c1}:{c2}": rel_type
                for (c1, c2), rel_type in self.relationships.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BlochSphereUniverse':
        """Create from dictionary"""
        universe = cls()
        
        # Load concepts
        for name, pos_data in data.get("concepts", {}).items():
            universe.concepts[name] = SphericalCoordinate.from_dict(pos_data)
        
        # Load relationships
        for key, rel_type in data.get("relationships", {}).items():
            c1, c2 = key.split(":")
            universe.relationships[(c1, c2)] = rel_type
        
        return universe
