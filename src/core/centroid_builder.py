"""
Centroid Builder for the Spherical Universe.
Builds concepts from the center (null) outward.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
import numpy as np
import asyncio
import random
import math

from src.core.spherical_universe import SphericalCoordinate, BlochSphereUniverse
from src.core.null_gradient import NullGradientManager

@dataclass
class ConceptBuildResult:
    """Result of building a concept from center"""
    concept_name: str
    negations: List[str]
    growth_pattern: str
    final_position: SphericalCoordinate
    antipodal_position: SphericalCoordinate
    growth_history: List[SphericalCoordinate]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "concept_name": self.concept_name,
            "negations": self.negations,
            "growth_pattern": self.growth_pattern,
            "final_position": self.final_position.to_dict(),
            "antipodal_position": self.antipodal_position.to_dict(),
            "growth_history": [pos.to_dict() for pos in self.growth_history]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptBuildResult':
        """Create from dictionary"""
        return cls(
            concept_name=data["concept_name"],
            negations=data["negations"],
            growth_pattern=data["growth_pattern"],
            final_position=SphericalCoordinate.from_dict(data["final_position"]),
            antipodal_position=SphericalCoordinate.from_dict(data["antipodal_position"]),
            growth_history=[SphericalCoordinate.from_dict(pos) for pos in data["growth_history"]]
        )

class CentroidConceptBuilder:
    """
    Builder for concepts in the spherical universe.
    
    Builds concepts from the center (null) outward using different growth patterns:
    - Radial: Straight line from center
    - Spiral: Spiral path from center
    - Branching: Branching path from center
    - Crystalline: Structured path with specific angles
    """
    
    def __init__(self, universe: Optional[BlochSphereUniverse] = None, null_gradient: Optional[NullGradientManager] = None):
        self.universe = universe if universe else BlochSphereUniverse()
        self.null_gradient = null_gradient if null_gradient else NullGradientManager()
    
    async def build_concept_from_center(self, 
                                      concept_name: str, 
                                      negations: List[str] = None, 
                                      growth_pattern: str = "radial", 
                                      target_radius: float = 0.4, 
                                      steps: int = 10) -> ConceptBuildResult:
        """
        Build a concept from the center (null) outward.
        
        Args:
            concept_name: Name of the concept
            negations: List of negation concepts
            growth_pattern: Growth pattern (radial, spiral, branching, crystalline)
            target_radius: Target radius (0 to 0.5)
            steps: Number of growth steps
            
        Returns:
            ConceptBuildResult: Result of building the concept
        """
        # Set defaults
        if negations is None:
            negations = []
        
        # Ensure target radius is within epistemic humility range
        target_radius = min(0.5, max(0, target_radius))
        
        # Start at center
        center = SphericalCoordinate(r=0.0, theta=0.0, phi=0.0)
        
        # Generate random direction
        theta = np.random.random() * 2 * np.pi
        phi = np.arccos(2 * np.random.random() - 1)  # Uniform on sphere
        
        # Create growth history
        growth_history = [center]
        
        # Generate growth path
        if growth_pattern == "radial":
            # Radial growth: straight line from center
            for i in range(1, steps + 1):
                # Calculate radius at this step
                r = target_radius * (i / steps)
                
                # Create position
                position = SphericalCoordinate(r=r, theta=theta, phi=phi)
                
                # Add to history
                growth_history.append(position)
        
        elif growth_pattern == "spiral":
            # Spiral growth: spiral path from center
            for i in range(1, steps + 1):
                # Calculate radius at this step
                r = target_radius * (i / steps)
                
                # Calculate theta offset (spiral)
                theta_offset = 2 * np.pi * (i / steps)
                
                # Create position
                position = SphericalCoordinate(
                    r=r,
                    theta=(theta + theta_offset) % (2 * np.pi),
                    phi=phi
                )
                
                # Add to history
                growth_history.append(position)
        
        elif growth_pattern == "branching":
            # Branching growth: branching path from center
            for i in range(1, steps + 1):
                # Calculate radius at this step
                r = target_radius * (i / steps)
                
                # Calculate theta and phi offsets (branching)
                if i % 3 == 0:
                    # Branch in theta direction
                    theta_offset = np.pi / 8 * np.sin(i)
                    phi_offset = 0
                elif i % 3 == 1:
                    # Branch in phi direction
                    theta_offset = 0
                    phi_offset = np.pi / 16 * np.sin(i)
                else:
                    # Branch in both directions
                    theta_offset = np.pi / 16 * np.sin(i)
                    phi_offset = np.pi / 32 * np.sin(i)
                
                # Create position
                position = SphericalCoordinate(
                    r=r,
                    theta=(theta + theta_offset) % (2 * np.pi),
                    phi=max(0, min(np.pi, phi + phi_offset))
                )
                
                # Add to history
                growth_history.append(position)
        
        elif growth_pattern == "crystalline":
            # Crystalline growth: structured path with specific angles
            for i in range(1, steps + 1):
                # Calculate radius at this step
                r = target_radius * (i / steps)
                
                # Calculate theta and phi (crystalline)
                if i % 4 == 0:
                    # Snap to 45-degree angles
                    theta_snap = np.round(theta / (np.pi / 4)) * (np.pi / 4)
                    phi_snap = np.round(phi / (np.pi / 6)) * (np.pi / 6)
                elif i % 4 == 1:
                    # Snap to 30-degree angles
                    theta_snap = np.round(theta / (np.pi / 6)) * (np.pi / 6)
                    phi_snap = np.round(phi / (np.pi / 4)) * (np.pi / 4)
                elif i % 4 == 2:
                    # Snap to 60-degree angles
                    theta_snap = np.round(theta / (np.pi / 3)) * (np.pi / 3)
                    phi_snap = np.round(phi / (np.pi / 3)) * (np.pi / 3)
                else:
                    # Snap to 90-degree angles
                    theta_snap = np.round(theta / (np.pi / 2)) * (np.pi / 2)
                    phi_snap = np.round(phi / (np.pi / 2)) * (np.pi / 2)
                
                # Create position
                position = SphericalCoordinate(
                    r=r,
                    theta=theta_snap % (2 * np.pi),
                    phi=max(0, min(np.pi, phi_snap))
                )
                
                # Add to history
                growth_history.append(position)
        
        else:
            # Default to radial growth
            for i in range(1, steps + 1):
                # Calculate radius at this step
                r = target_radius * (i / steps)
                
                # Create position
                position = SphericalCoordinate(r=r, theta=theta, phi=phi)
                
                # Add to history
                growth_history.append(position)
        
        # Get final position
        final_position = growth_history[-1]
        
        # Get antipodal position
        antipodal_position = self.universe.get_antipodal_point(final_position)
        
        # Add concept to universe
        self.universe.add_concept(concept_name, final_position, add_negation=True)
        
        # Add negations
        for negation in negations:
            if negation != f"!{concept_name}":  # Skip default negation
                # Create negation position (near antipodal)
                neg_pos = self.universe.get_antipodal_point(final_position)
                
                # Add small random variation
                neg_theta = (neg_pos.theta + (np.random.random() - 0.5) * 0.2) % (2 * np.pi)
                neg_phi = max(0, min(np.pi, neg_pos.phi + (np.random.random() - 0.5) * 0.2))
                
                # Create negation position
                neg_pos = SphericalCoordinate(r=final_position.r, theta=neg_theta, phi=neg_phi)
                
                # Add to universe
                self.universe.add_concept(negation, neg_pos)
                
                # Add relationship
                self.universe.add_relationship(concept_name, negation, "not")
                self.universe.add_relationship(negation, concept_name, "not")
        
        # Create result
        result = ConceptBuildResult(
            concept_name=concept_name,
            negations=negations,
            growth_pattern=growth_pattern,
            final_position=final_position,
            antipodal_position=antipodal_position,
            growth_history=growth_history
        )
        
        return result
    
    async def build_concept_cluster(self, 
                                  concepts: List[str], 
                                  relationships: Dict[Tuple[str, str], str], 
                                  target_radius: float = 0.4) -> Dict[str, ConceptBuildResult]:
        """
        Build a cluster of related concepts.
        
        Args:
            concepts: List of concept names
            relationships: Dictionary mapping (concept1, concept2) to relationship type
            target_radius: Target radius (0 to 0.5)
            
        Returns:
            Dict[str, ConceptBuildResult]: Mapping of concepts to build results
        """
        # Ensure target radius is within epistemic humility range
        target_radius = min(0.5, max(0, target_radius))
        
        # Build results
        results = {}
        
        # First pass: build concepts with random directions
        for concept in concepts:
            # Build concept
            result = await self.build_concept_from_center(
                concept_name=concept,
                target_radius=target_radius
            )
            
            # Store result
            results[concept] = result
        
        # Second pass: adjust positions based on relationships
        for (concept1, concept2), rel_type in relationships.items():
            # Skip if either concept not in results
            if concept1 not in results or concept2 not in results:
                continue
            
            # Get positions
            pos1 = results[concept1].final_position
            pos2 = results[concept2].final_position
            
            # Calculate current angular distance
            current_distance = self.universe.calculate_angular_distance(pos1, pos2)
            
            # Determine target angular distance based on relationship type
            target_distance = 0
            if rel_type == "and":
                # AND: close together (0-45 degrees)
                target_distance = np.radians(30)
            elif rel_type == "or":
                # OR: orthogonal (90 degrees)
                target_distance = np.radians(90)
            elif rel_type == "not":
                # NOT: opposite (180 degrees)
                target_distance = np.radians(180)
            
            # Skip if target distance is 0
            if target_distance == 0:
                continue
            
            # Calculate adjustment factor
            adjustment_factor = 0.5  # How much to adjust (0 to 1)
            
            # Calculate midpoint
            mid_cart = (pos1.to_cartesian() + pos2.to_cartesian()) / 2
            mid_pos = SphericalCoordinate.from_cartesian(mid_cart)
            
            # Adjust positions
            if rel_type == "not":
                # For NOT, make them antipodal
                new_pos1 = pos1
                new_pos2 = self.universe.get_antipodal_point(pos1)
            else:
                # For AND and OR, adjust both positions
                # Calculate rotation axis
                axis = np.cross(pos1.to_cartesian(), pos2.to_cartesian())
                if np.linalg.norm(axis) < 1e-6:
                    # Positions are collinear, choose arbitrary axis
                    axis = np.array([0, 0, 1])
                else:
                    axis = axis / np.linalg.norm(axis)
                
                # Calculate rotation angle
                angle = (target_distance - current_distance) * adjustment_factor
                
                # Rotate positions
                rot_matrix1 = self._rotation_matrix(axis, angle / 2)
                rot_matrix2 = self._rotation_matrix(axis, -angle / 2)
                
                new_cart1 = np.dot(rot_matrix1, pos1.to_cartesian())
                new_cart2 = np.dot(rot_matrix2, pos2.to_cartesian())
                
                new_pos1 = SphericalCoordinate.from_cartesian(new_cart1)
                new_pos2 = SphericalCoordinate.from_cartesian(new_cart2)
                
                # Maintain radius
                new_pos1 = SphericalCoordinate(
                    r=pos1.r,
                    theta=new_pos1.theta,
                    phi=new_pos1.phi
                )
                
                new_pos2 = SphericalCoordinate(
                    r=pos2.r,
                    theta=new_pos2.theta,
                    phi=new_pos2.phi
                )
            
            # Update universe
            self.universe.concepts[concept1] = new_pos1
            self.universe.concepts[concept2] = new_pos2
            
            # Update results
            results[concept1].final_position = new_pos1
            results[concept2].final_position = new_pos2
            
            # Update antipodal positions
            results[concept1].antipodal_position = self.universe.get_antipodal_point(new_pos1)
            results[concept2].antipodal_position = self.universe.get_antipodal_point(new_pos2)
            
            # Add relationship to universe
            self.universe.add_relationship(concept1, concept2, rel_type)
            self.universe.add_relationship(concept2, concept1, rel_type)
        
        return results
    
    def _rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Calculate rotation matrix for rotating around an axis.
        
        Args:
            axis: Rotation axis (unit vector)
            angle: Rotation angle (radians)
            
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        # Calculate rotation matrix
        a = np.cos(angle / 2)
        b, c, d = -axis * np.sin(angle / 2)
        
        return np.array([
            [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
            [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
            [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
        ])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "universe": self.universe.to_dict(),
            "null_gradient": self.null_gradient.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CentroidConceptBuilder':
        """Create from dictionary"""
        # Create universe
        universe = BlochSphereUniverse.from_dict(data.get("universe", {}))
        
        # Create null gradient
        null_gradient = NullGradientManager.from_dict(data.get("null_gradient", {}))
        
        # Create builder
        return cls(universe, null_gradient)
