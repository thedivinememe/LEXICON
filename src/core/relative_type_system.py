"""
Relative Type System.

This module provides a type system for the spherical universe,
where types are defined relative to concepts and their positions.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Union

from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager

logger = logging.getLogger(__name__)

@dataclass
class RelativeTypeHierarchy:
    """
    Relative type hierarchy for a concept.
    
    A type hierarchy defines a set of types related to a concept,
    including a bottom type (the concept itself), a top type (any concept),
    middle types (based on relationships), and a unified type (the origin).
    """
    
    concept: str
    position: SphericalCoordinate
    bottom_type: str
    top_type: str
    unified_type: str
    middle_types: List[str] = field(default_factory=list)
    type_positions: Dict[str, SphericalCoordinate] = field(default_factory=dict)
    subtype_relationships: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize type positions."""
        # Add bottom type position (same as concept)
        self.type_positions[self.bottom_type] = self.position
        
        # Add top type position (opposite of concept)
        top_cart = -self.position.to_cartesian()
        self.type_positions[self.top_type] = SphericalCoordinate.from_cartesian(top_cart)
        
        # Add unified type position (origin)
        self.type_positions[self.unified_type] = SphericalCoordinate(r=0.0, theta=0.0, phi=0.0)
        
        # Initialize subtype relationships
        self._initialize_subtype_relationships()
    
    def _initialize_subtype_relationships(self):
        """Initialize subtype relationships."""
        # Every type is a subtype of itself
        for type_name in self.type_positions.keys():
            self.subtype_relationships[(type_name, type_name)] = True
        
        # Bottom type is a subtype of all types
        for type_name in self.type_positions.keys():
            if type_name != self.bottom_type:
                self.subtype_relationships[(self.bottom_type, type_name)] = True
                self.subtype_relationships[(type_name, self.bottom_type)] = False
        
        # All types are subtypes of the unified type
        for type_name in self.type_positions.keys():
            if type_name != self.unified_type:
                self.subtype_relationships[(type_name, self.unified_type)] = True
                self.subtype_relationships[(self.unified_type, type_name)] = False
        
        # Top type is a supertype of all types except unified
        for type_name in self.type_positions.keys():
            if type_name != self.top_type and type_name != self.unified_type:
                self.subtype_relationships[(type_name, self.top_type)] = True
                self.subtype_relationships[(self.top_type, type_name)] = False
    
    def add_middle_type(self, type_name: str, position: SphericalCoordinate):
        """
        Add a middle type to the hierarchy.
        
        Args:
            type_name: Type name
            position: Type position
        """
        # Add to middle types
        self.middle_types.append(type_name)
        
        # Add position
        self.type_positions[type_name] = position
        
        # Add subtype relationships
        self.subtype_relationships[(type_name, type_name)] = True
        self.subtype_relationships[(self.bottom_type, type_name)] = True
        self.subtype_relationships[(type_name, self.bottom_type)] = False
        self.subtype_relationships[(type_name, self.unified_type)] = True
        self.subtype_relationships[(self.unified_type, type_name)] = False
        self.subtype_relationships[(type_name, self.top_type)] = True
        self.subtype_relationships[(self.top_type, type_name)] = False
        
        # Update relationships with other middle types
        for other_type in self.middle_types:
            if other_type != type_name:
                # Calculate angular distance
                pos1 = self.type_positions[type_name]
                pos2 = self.type_positions[other_type]
                
                # Convert to Cartesian
                cart1 = pos1.to_cartesian()
                cart2 = pos2.to_cartesian()
                
                # Calculate dot product
                dot_product = np.dot(cart1, cart2)
                
                # If dot product is positive, type_name is a subtype of other_type
                if dot_product > 0:
                    self.subtype_relationships[(type_name, other_type)] = True
                    self.subtype_relationships[(other_type, type_name)] = False
                else:
                    self.subtype_relationships[(type_name, other_type)] = False
                    self.subtype_relationships[(other_type, type_name)] = True

class RelativeTypeSystem:
    """
    Relative type system for the spherical universe.
    
    The type system defines types relative to concepts and their positions
    in the spherical universe. Types form a hierarchy with subtype relationships.
    """
    
    def __init__(self, 
                universe: Optional[BlochSphereUniverse] = None,
                null_gradient: Optional[NullGradientManager] = None):
        """
        Initialize the type system.
        
        Args:
            universe: Spherical universe
            null_gradient: Null gradient manager
        """
        self.universe = universe if universe else BlochSphereUniverse()
        self.null_gradient = null_gradient if null_gradient else NullGradientManager(self.universe)
        
        # Cache for type hierarchies
        self.hierarchies: Dict[str, RelativeTypeHierarchy] = {}
    
    async def create_relative_hierarchy(self, 
                                      concept: str, 
                                      position: Optional[SphericalCoordinate] = None) -> RelativeTypeHierarchy:
        """
        Create a relative type hierarchy for a concept.
        
        Args:
            concept: Concept name
            position: Concept position (if None, get from universe)
            
        Returns:
            Relative type hierarchy
            
        Raises:
            ValueError: If concept not found
        """
        logger.info(f"Creating relative type hierarchy for concept '{concept}'")
        
        # Check if concept exists
        if concept not in self.universe.concepts and position is None:
            raise ValueError(f"Concept '{concept}' not found")
        
        # Get position if not provided
        if position is None:
            position = self.universe.get_concept_position(concept)
        
        # Check if hierarchy already exists
        if concept in self.hierarchies:
            return self.hierarchies[concept]
        
        # Create type names
        bottom_type = f"{concept}_type"
        top_type = f"any_{concept}"
        unified_type = "unified_type"
        
        # Create hierarchy
        hierarchy = RelativeTypeHierarchy(
            concept=concept,
            position=position,
            bottom_type=bottom_type,
            top_type=top_type,
            unified_type=unified_type
        )
        
        # Add middle types based on relationships
        related_concepts = self.universe.get_related_concepts(concept)
        
        for related_concept in related_concepts:
            # Get relationship type
            rel_type = self.universe.get_relationship_type(concept, related_concept)
            
            # Create middle type name
            middle_type = f"{concept}_{rel_type}_{related_concept}"
            
            # Get related concept position
            related_position = self.universe.get_concept_position(related_concept)
            
            # Create middle type position (between concept and related concept)
            middle_cart = (position.to_cartesian() + related_position.to_cartesian()) / 2
            middle_position = SphericalCoordinate.from_cartesian(middle_cart)
            
            # Add middle type
            hierarchy.add_middle_type(middle_type, middle_position)
        
        # Cache hierarchy
        self.hierarchies[concept] = hierarchy
        
        logger.info(f"Created relative type hierarchy for concept '{concept}' with {len(hierarchy.middle_types)} middle types")
        
        return hierarchy
    
    async def calculate_type_boundaries(self, hierarchy: RelativeTypeHierarchy) -> Dict[str, SphericalCoordinate]:
        """
        Calculate type boundaries for a hierarchy.
        
        Args:
            hierarchy: Type hierarchy
            
        Returns:
            Dictionary mapping type names to boundary positions
        """
        logger.info(f"Calculating type boundaries for concept '{hierarchy.concept}'")
        
        # Return type positions
        return hierarchy.type_positions
    
    async def get_type_at_position(self, 
                                 hierarchy: RelativeTypeHierarchy, 
                                 position: SphericalCoordinate) -> str:
        """
        Get the most specific type at a position.
        
        Args:
            hierarchy: Type hierarchy
            position: Position
            
        Returns:
            Type name
        """
        logger.info(f"Getting type at position {position}")
        
        # Special case: if position is at the origin, return unified type
        if position.r < 1e-6:
            logger.info(f"Position is at origin, returning unified type '{hierarchy.unified_type}'")
            return hierarchy.unified_type
        
        # Calculate distances to type positions
        distances = {}
        
        for type_name, type_position in hierarchy.type_positions.items():
            # Calculate angular distance
            distance = self.universe.calculate_angular_distance(position, type_position)
            distances[type_name] = distance
        
        # Find closest type
        closest_type = min(distances.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Type at position {position} is '{closest_type}'")
        
        return closest_type
    
    async def get_subtype_relationship(self, 
                                     hierarchy: RelativeTypeHierarchy, 
                                     type1: str, 
                                     type2: str) -> bool:
        """
        Check if type1 is a subtype of type2.
        
        Args:
            hierarchy: Type hierarchy
            type1: First type
            type2: Second type
            
        Returns:
            True if type1 is a subtype of type2, False otherwise
            
        Raises:
            ValueError: If types not found in hierarchy
        """
        logger.info(f"Checking if '{type1}' is a subtype of '{type2}'")
        
        # Check if types exist
        if type1 not in hierarchy.type_positions:
            raise ValueError(f"Type '{type1}' not found in hierarchy")
        
        if type2 not in hierarchy.type_positions:
            raise ValueError(f"Type '{type2}' not found in hierarchy")
        
        # Check subtype relationship
        is_subtype = hierarchy.subtype_relationships.get((type1, type2), False)
        
        logger.info(f"'{type1}' is{'' if is_subtype else ' not'} a subtype of '{type2}'")
        
        return is_subtype
    
    async def get_common_supertype(self, 
                                 hierarchy: RelativeTypeHierarchy, 
                                 type1: str, 
                                 type2: str) -> str:
        """
        Get the most specific common supertype of two types.
        
        Args:
            hierarchy: Type hierarchy
            type1: First type
            type2: Second type
            
        Returns:
            Common supertype
            
        Raises:
            ValueError: If types not found in hierarchy
        """
        logger.info(f"Getting common supertype of '{type1}' and '{type2}'")
        
        # Check if types exist
        if type1 not in hierarchy.type_positions:
            raise ValueError(f"Type '{type1}' not found in hierarchy")
        
        if type2 not in hierarchy.type_positions:
            raise ValueError(f"Type '{type2}' not found in hierarchy")
        
        # If one is a subtype of the other, return the supertype
        if await self.get_subtype_relationship(hierarchy, type1, type2):
            return type2
        
        if await self.get_subtype_relationship(hierarchy, type2, type1):
            return type1
        
        # Otherwise, find common supertypes
        common_supertypes = []
        
        for type_name in hierarchy.type_positions.keys():
            if (await self.get_subtype_relationship(hierarchy, type1, type_name) and
                await self.get_subtype_relationship(hierarchy, type2, type_name)):
                common_supertypes.append(type_name)
        
        # If no common supertypes, return unified type
        if not common_supertypes:
            return hierarchy.unified_type
        
        # Find most specific common supertype
        most_specific = common_supertypes[0]
        
        for type_name in common_supertypes[1:]:
            if await self.get_subtype_relationship(hierarchy, type_name, most_specific):
                most_specific = type_name
        
        logger.info(f"Common supertype of '{type1}' and '{type2}' is '{most_specific}'")
        
        return most_specific
    
    async def get_common_subtype(self, 
                               hierarchy: RelativeTypeHierarchy, 
                               type1: str, 
                               type2: str) -> Optional[str]:
        """
        Get the most general common subtype of two types.
        
        Args:
            hierarchy: Type hierarchy
            type1: First type
            type2: Second type
            
        Returns:
            Common subtype, or None if no common subtype exists
            
        Raises:
            ValueError: If types not found in hierarchy
        """
        logger.info(f"Getting common subtype of '{type1}' and '{type2}'")
        
        # Check if types exist
        if type1 not in hierarchy.type_positions:
            raise ValueError(f"Type '{type1}' not found in hierarchy")
        
        if type2 not in hierarchy.type_positions:
            raise ValueError(f"Type '{type2}' not found in hierarchy")
        
        # If one is a subtype of the other, return the subtype
        if await self.get_subtype_relationship(hierarchy, type1, type2):
            return type1
        
        if await self.get_subtype_relationship(hierarchy, type2, type1):
            return type2
        
        # Otherwise, find common subtypes
        common_subtypes = []
        
        for type_name in hierarchy.type_positions.keys():
            if (await self.get_subtype_relationship(hierarchy, type_name, type1) and
                await self.get_subtype_relationship(hierarchy, type_name, type2)):
                common_subtypes.append(type_name)
        
        # If no common subtypes, return None
        if not common_subtypes:
            return None
        
        # Find most general common subtype
        most_general = common_subtypes[0]
        
        for type_name in common_subtypes[1:]:
            if await self.get_subtype_relationship(hierarchy, most_general, type_name):
                most_general = type_name
        
        logger.info(f"Common subtype of '{type1}' and '{type2}' is '{most_general}'")
        
        return most_general
    
    async def get_all_subtypes(self, 
                             hierarchy: RelativeTypeHierarchy, 
                             type_name: str) -> List[str]:
        """
        Get all subtypes of a type.
        
        Args:
            hierarchy: Type hierarchy
            type_name: Type name
            
        Returns:
            List of subtype names
            
        Raises:
            ValueError: If type not found in hierarchy
        """
        logger.info(f"Getting all subtypes of '{type_name}'")
        
        # Check if type exists
        if type_name not in hierarchy.type_positions:
            raise ValueError(f"Type '{type_name}' not found in hierarchy")
        
        # Find subtypes
        subtypes = []
        
        for other_type in hierarchy.type_positions.keys():
            if await self.get_subtype_relationship(hierarchy, other_type, type_name):
                subtypes.append(other_type)
        
        logger.info(f"Found {len(subtypes)} subtypes of '{type_name}'")
        
        return subtypes
    
    async def get_all_supertypes(self, 
                               hierarchy: RelativeTypeHierarchy, 
                               type_name: str) -> List[str]:
        """
        Get all supertypes of a type.
        
        Args:
            hierarchy: Type hierarchy
            type_name: Type name
            
        Returns:
            List of supertype names
            
        Raises:
            ValueError: If type not found in hierarchy
        """
        logger.info(f"Getting all supertypes of '{type_name}'")
        
        # Check if type exists
        if type_name not in hierarchy.type_positions:
            raise ValueError(f"Type '{type_name}' not found in hierarchy")
        
        # Find supertypes
        supertypes = []
        
        for other_type in hierarchy.type_positions.keys():
            if await self.get_subtype_relationship(hierarchy, type_name, other_type):
                supertypes.append(other_type)
        
        logger.info(f"Found {len(supertypes)} supertypes of '{type_name}'")
        
        return supertypes
