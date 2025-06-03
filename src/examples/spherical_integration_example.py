"""
Spherical Integration Example.

This module provides an example of how to use the spherical universe system,
including creating concepts, relationships, type hierarchies, and visualizations.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager
from src.core.relative_type_system import RelativeTypeSystem, RelativeTypeHierarchy
from src.neural.spherical_vectorizer import SphericalRelationshipVectorizer
from src.services.sphere_visualization import SphericalUniverseVisualizer
from src.data.core_definitions import CORE_DEFINITIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def initialize_universe() -> Tuple[BlochSphereUniverse, NullGradientManager, SphericalRelationshipVectorizer, RelativeTypeSystem, SphericalUniverseVisualizer]:
    """
    Initialize the spherical universe with core concepts.
    
    Returns:
        Tuple of (universe, null_gradient, vectorizer, type_system, visualizer)
    """
    logger.info("Initializing spherical universe")
    
    # Create components
    universe = BlochSphereUniverse()
    null_gradient = NullGradientManager(max_radius=0.5)
    vectorizer = SphericalRelationshipVectorizer(universe, null_gradient)
    type_system = RelativeTypeSystem(universe, null_gradient)
    visualizer = SphericalUniverseVisualizer(universe, null_gradient, vectorizer, type_system)
    
    # Add core concepts
    for concept, data in CORE_DEFINITIONS.items():
        # Get spherical properties
        spherical_props = data.get("spherical_properties", {})
        preferred_r = spherical_props.get("preferred_r", 0.3)
        preferred_theta = spherical_props.get("preferred_theta", 0.0)
        preferred_phi = spherical_props.get("preferred_phi", 0.0)
        
        # Create position
        position = SphericalCoordinate(
            r=preferred_r,
            theta=preferred_theta,
            phi=preferred_phi
        )
        
        # Add concept
        universe.add_concept(concept, position)
        
        logger.info(f"Added concept '{concept}' at position {position}")
    
    # Add relationships
    for concept, data in CORE_DEFINITIONS.items():
        # Add AND relationships
        for related_concept, strength in data.get("and_relationships", []):
            if related_concept in universe.concepts:
                universe.add_relationship(concept, related_concept, "and")
                logger.info(f"Added AND relationship: {concept} -> {related_concept}")
        
        # Add OR relationships
        for related_concept, strength in data.get("or_relationships", []):
            if related_concept in universe.concepts:
                universe.add_relationship(concept, related_concept, "or")
                logger.info(f"Added OR relationship: {concept} -> {related_concept}")
        
        # Add NOT relationships
        for related_concept, strength in data.get("not_relationships", []):
            if related_concept in universe.concepts:
                universe.add_relationship(concept, related_concept, "not")
                logger.info(f"Added NOT relationship: {concept} -> {related_concept}")
    
    logger.info(f"Initialized spherical universe with {len(universe.concepts)} concepts")
    
    return universe, null_gradient, vectorizer, type_system, visualizer

async def create_concept_with_constraints(vectorizer: SphericalRelationshipVectorizer, concept: str, constraints: Dict[str, Any]) -> SphericalCoordinate:
    """
    Create a concept with constraints.
    
    Args:
        vectorizer: Vectorizer
        concept: Concept name
        constraints: Constraints
        
    Returns:
        Concept position
    """
    logger.info(f"Creating concept '{concept}' with constraints")
    
    # Vectorize concept
    position = await vectorizer.vectorize_with_constraints(concept, constraints)
    
    logger.info(f"Created concept '{concept}' at position {position}")
    
    return position

async def create_opposite_concept(vectorizer: SphericalRelationshipVectorizer, concept: str, opposite_name: Optional[str] = None) -> Tuple[str, SphericalCoordinate]:
    """
    Create an opposite concept.
    
    Args:
        vectorizer: Vectorizer
        concept: Concept name
        opposite_name: Optional name for the opposite concept
        
    Returns:
        Tuple of (concept_name, position)
    """
    logger.info(f"Creating opposite concept for '{concept}'")
    
    # Create opposite concept
    opposite_name, position = await vectorizer.create_opposite_concept(concept, opposite_name)
    
    logger.info(f"Created opposite concept '{opposite_name}' at position {position}")
    
    return opposite_name, position

async def create_interpolated_concept(vectorizer: SphericalRelationshipVectorizer, concept1: str, concept2: str, weight: float = 0.5, new_concept: Optional[str] = None) -> Tuple[str, SphericalCoordinate]:
    """
    Create an interpolated concept.
    
    Args:
        vectorizer: Vectorizer
        concept1: First concept name
        concept2: Second concept name
        weight: Interpolation weight
        new_concept: Optional name for the new concept
        
    Returns:
        Tuple of (concept_name, position)
    """
    logger.info(f"Creating interpolated concept between '{concept1}' and '{concept2}' with weight {weight}")
    
    # Create interpolated concept
    new_concept, position = await vectorizer.create_concept_from_interpolation(concept1, concept2, weight, new_concept)
    
    logger.info(f"Created interpolated concept '{new_concept}' at position {position}")
    
    return new_concept, position

async def find_nearest_concepts(vectorizer: SphericalRelationshipVectorizer, concept: str, count: int = 5) -> List[Tuple[str, float]]:
    """
    Find the nearest concepts to a concept.
    
    Args:
        vectorizer: Vectorizer
        concept: Concept name
        count: Number of nearest concepts
        
    Returns:
        List of (concept_name, distance) tuples
    """
    logger.info(f"Finding nearest concepts to '{concept}'")
    
    # Get concept position
    position = vectorizer.universe.get_concept_position(concept)
    
    # Get nearest concepts
    nearest = await vectorizer.get_nearest_concepts(position, count)
    
    logger.info(f"Found {len(nearest)} nearest concepts to '{concept}'")
    
    return nearest

async def create_type_hierarchy(type_system: RelativeTypeSystem, concept: str) -> RelativeTypeHierarchy:
    """
    Create a type hierarchy for a concept.
    
    Args:
        type_system: Type system
        concept: Concept name
        
    Returns:
        Type hierarchy
    """
    logger.info(f"Creating type hierarchy for concept '{concept}'")
    
    # Get concept position
    position = type_system.universe.get_concept_position(concept)
    
    # Create hierarchy
    hierarchy = await type_system.create_relative_hierarchy(concept, position)
    
    logger.info(f"Created type hierarchy for concept '{concept}'")
    
    return hierarchy

async def generate_visualizations(visualizer: SphericalUniverseVisualizer, concepts: List[str]) -> Dict[str, Path]:
    """
    Generate visualizations for concepts.
    
    Args:
        visualizer: Visualizer
        concepts: List of concepts
        
    Returns:
        Dictionary mapping visualization types to file paths
    """
    logger.info(f"Generating visualizations for {len(concepts)} concepts")
    
    visualizations = {}
    
    # Generate sphere visualization
    sphere_path = await visualizer.generate_sphere_visualization(concepts)
    visualizations["sphere"] = sphere_path
    
    # Generate null gradient visualization
    null_path = await visualizer.generate_null_gradient_visualization()
    visualizations["null_gradient"] = null_path
    
    # Generate visualizations for each concept
    for concept in concepts:
        # Generate relationship visualization
        rel_path = await visualizer.generate_relationship_visualization(concept)
        visualizations[f"{concept}_relationships"] = rel_path
        
        # Generate type hierarchy visualization
        type_path = await visualizer.generate_type_hierarchy_visualization(concept)
        visualizations[f"{concept}_type_hierarchy"] = type_path
        
        # Generate concept cluster visualization
        cluster_path = await visualizer.generate_concept_cluster_visualization(concept)
        visualizations[f"{concept}_cluster"] = cluster_path
        
        # Generate nearest concepts visualization
        nearest_path = await visualizer.generate_nearest_concepts_visualization(concept)
        visualizations[f"{concept}_nearest"] = nearest_path
    
    # Generate interpolation visualization
    if len(concepts) >= 2:
        interp_path = await visualizer.generate_concept_interpolation_visualization(concepts[0], concepts[1])
        visualizations["interpolation"] = interp_path
    
    logger.info(f"Generated {len(visualizations)} visualizations")
    
    return visualizations

async def run_example() -> None:
    """Run the spherical integration example."""
    logger.info("Running spherical integration example")
    
    # Initialize universe
    universe, null_gradient, vectorizer, type_system, visualizer = await initialize_universe()
    
    # Create concepts with constraints
    await create_concept_with_constraints(
        vectorizer,
        "abstract_concept",
        {
            "min_radius": 0.3,
            "max_radius": 0.4,
            "near_concept": "abstraction",
            "max_angle": 0.5,
            "relationship_type": "and"
        }
    )
    
    await create_concept_with_constraints(
        vectorizer,
        "concrete_example",
        {
            "min_radius": 0.3,
            "max_radius": 0.4,
            "near_concept": "concrete",
            "max_angle": 0.5,
            "relationship_type": "and"
        }
    )
    
    # Create opposite concepts
    await create_opposite_concept(vectorizer, "abstract_concept", "concrete_concept")
    await create_opposite_concept(vectorizer, "concrete_example", "abstract_example")
    
    # Create interpolated concepts
    await create_interpolated_concept(vectorizer, "abstract_concept", "concrete_concept", 0.5, "mixed_concept")
    await create_interpolated_concept(vectorizer, "abstract_example", "concrete_example", 0.5, "mixed_example")
    
    # Find nearest concepts
    nearest_to_abstract = await find_nearest_concepts(vectorizer, "abstract_concept")
    nearest_to_concrete = await find_nearest_concepts(vectorizer, "concrete_concept")
    
    # Create type hierarchies
    abstract_hierarchy = await create_type_hierarchy(type_system, "abstract_concept")
    concrete_hierarchy = await create_type_hierarchy(type_system, "concrete_concept")
    
    # Generate visualizations
    concepts = [
        "abstract_concept",
        "concrete_concept",
        "mixed_concept",
        "abstract_example",
        "concrete_example",
        "mixed_example"
    ]
    
    visualizations = await generate_visualizations(visualizer, concepts)
    
    logger.info("Spherical integration example completed")
    
    # Return the main visualization path
    return visualizations["sphere"]

if __name__ == "__main__":
    # Run the example
    asyncio.run(run_example())
