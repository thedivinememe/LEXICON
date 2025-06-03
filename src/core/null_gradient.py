"""
Null Gradient Manager for the Spherical Universe.
Manages the null field and epistemic humility constraints.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
import math

from src.core.spherical_universe import SphericalCoordinate

class NullGradientManager:
    """
    Manager for the null gradient in the spherical universe.
    
    The null gradient defines:
    - Null field intensity (decreases with radius)
    - Epistemic humility constraints (max radius = 0.5)
    - Null field properties (uncertainty, potentiality, etc.)
    """
    
    def __init__(self, max_radius: float = 0.5, null_field_properties: Optional[Dict[str, Any]] = None):
        self.max_radius = max_radius
        self.null_field_properties = null_field_properties if null_field_properties else {
            "uncertainty": 1.0,
            "potentiality": 1.0,
            "virtuality": 1.0,
            "indeterminacy": 1.0
        }
    
    def calculate_null_field(self, position: SphericalCoordinate) -> Dict[str, Any]:
        """
        Calculate null field properties at a position.
        
        Args:
            position: Spherical position
            
        Returns:
            Dict[str, Any]: Null field properties
        """
        # Calculate null intensity (1 at center, 0 at max_radius)
        r = position.r
        null_intensity = max(0.0, 1.0 - (r / self.max_radius))
        
        # Calculate field properties
        field = {}
        
        # Set null intensity
        field["null_intensity"] = null_intensity
        
        # Scale other properties by null intensity
        for prop, value in self.null_field_properties.items():
            field[prop] = value * null_intensity
        
        # Calculate additional properties
        field["certainty"] = 1.0 - field.get("uncertainty", 0.0)
        field["actuality"] = 1.0 - field.get("potentiality", 0.0)
        field["reality"] = 1.0 - field.get("virtuality", 0.0)
        field["determinacy"] = 1.0 - field.get("indeterminacy", 0.0)
        
        # Calculate null gradient
        gradient = self.calculate_null_gradient(position)
        field["null_gradient"] = gradient.tolist()
        
        return field
    
    def enforce_epistemic_humility(self, position: SphericalCoordinate) -> SphericalCoordinate:
        """
        Enforce epistemic humility by constraining radius.
        
        Args:
            position: Spherical position
            
        Returns:
            SphericalCoordinate: Constrained position
        """
        # Check if radius exceeds max_radius
        if position.r <= self.max_radius:
            return position
        
        # Constrain radius
        return SphericalCoordinate(
            r=self.max_radius,
            theta=position.theta,
            phi=position.phi
        )
    
    def calculate_null_distance(self, position: SphericalCoordinate) -> float:
        """
        Calculate distance from null (center).
        
        Args:
            position: Spherical position
            
        Returns:
            float: Distance from null (0 to 1)
        """
        # Normalize radius to [0, 1]
        return position.r / self.max_radius
    
    def calculate_null_gradient(self, position: SphericalCoordinate) -> np.ndarray:
        """
        Calculate gradient of null field at position.
        
        Args:
            position: Spherical position
            
        Returns:
            np.ndarray: Gradient vector (points toward center)
        """
        # Convert to Cartesian
        cart = position.to_cartesian()
        
        # Calculate distance from center
        distance = np.linalg.norm(cart)
        
        # Handle center case
        if distance < 1e-6:
            return np.zeros(3)
        
        # Calculate gradient (points toward center)
        gradient = -cart / distance
        
        # Scale by null intensity gradient
        scale = 1.0 / self.max_radius
        
        return gradient * scale
    
    def calculate_epistemic_barrier(self, position: SphericalCoordinate) -> float:
        """
        Calculate epistemic barrier strength at position.
        
        Args:
            position: Spherical position
            
        Returns:
            float: Barrier strength (0 to 1)
        """
        # Calculate normalized distance to max_radius
        r = position.r
        distance_to_barrier = max(0.0, self.max_radius - r)
        normalized_distance = distance_to_barrier / self.max_radius
        
        # Calculate barrier strength
        # Increases as position approaches max_radius
        barrier_strength = 1.0 - normalized_distance
        
        return barrier_strength
    
    def calculate_null_field_interaction(self, 
                                       position1: SphericalCoordinate, 
                                       position2: SphericalCoordinate) -> Dict[str, Any]:
        """
        Calculate interaction between null fields at two positions.
        
        Args:
            position1: First position
            position2: Second position
            
        Returns:
            Dict[str, Any]: Interaction properties
        """
        # Calculate null fields
        field1 = self.calculate_null_field(position1)
        field2 = self.calculate_null_field(position2)
        
        # Calculate interaction
        interaction = {}
        
        # Calculate average null intensity
        interaction["null_intensity"] = (field1["null_intensity"] + field2["null_intensity"]) / 2
        
        # Calculate null intensity difference
        interaction["null_intensity_diff"] = abs(field1["null_intensity"] - field2["null_intensity"])
        
        # Calculate null gradient alignment
        gradient1 = self.calculate_null_gradient(position1)
        gradient2 = self.calculate_null_gradient(position2)
        
        # Handle zero gradients
        if np.linalg.norm(gradient1) < 1e-6 or np.linalg.norm(gradient2) < 1e-6:
            interaction["gradient_alignment"] = 0.0
        else:
            # Normalize gradients
            gradient1 = gradient1 / np.linalg.norm(gradient1)
            gradient2 = gradient2 / np.linalg.norm(gradient2)
            
            # Calculate dot product
            dot_product = np.dot(gradient1, gradient2)
            
            # Map to [0, 1]
            interaction["gradient_alignment"] = (dot_product + 1.0) / 2.0
        
        return interaction
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "max_radius": self.max_radius,
            "null_field_properties": self.null_field_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NullGradientManager':
        """Create from dictionary"""
        return cls(
            max_radius=data.get("max_radius", 0.5),
            null_field_properties=data.get("null_field_properties", None)
        )
