"""
Tests for the Spherical Universe System.

This module contains tests for the spherical universe system,
including tests for concepts, relationships, type hierarchies,
and visualizations.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager
from src.core.relative_type_system import RelativeTypeSystem, RelativeTypeHierarchy
from src.neural.spherical_vectorizer import SphericalRelationshipVectorizer
from src.services.sphere_visualization import SphericalUniverseVisualizer
from src.data.core_definitions import CORE_DEFINITIONS

# Test fixtures
@pytest.fixture
def universe():
    """Create a BlochSphereUniverse instance."""
    return BlochSphereUniverse()

@pytest.fixture
def null_gradient():
    """Create a NullGradientManager instance."""
    return NullGradientManager(max_radius=0.5)

@pytest.fixture
def vectorizer(universe, null_gradient):
    """Create a SphericalRelationshipVectorizer instance."""
    return SphericalRelationshipVectorizer(universe, null_gradient)

@pytest.fixture
def type_system(universe, null_gradient):
    """Create a RelativeTypeSystem instance."""
    return RelativeTypeSystem(universe, null_gradient)

@pytest.fixture
def visualizer(universe, null_gradient, vectorizer, type_system):
    """Create a SphericalUniverseVisualizer instance."""
    return SphericalUniverseVisualizer(universe, null_gradient, vectorizer, type_system)

@pytest.fixture
def sample_concepts():
    """Create sample concepts."""
    return {
        "concept1": SphericalCoordinate(r=0.3, theta=0.0, phi=0.0),
        "concept2": SphericalCoordinate(r=0.3, theta=np.pi/2, phi=np.pi/4),
        "concept3": SphericalCoordinate(r=0.3, theta=np.pi, phi=np.pi/2)
    }

# Tests for SphericalCoordinate
def test_spherical_coordinate_creation():
    """Test creation of spherical coordinates."""
    # Create coordinate
    coord = SphericalCoordinate(r=0.5, theta=np.pi/4, phi=np.pi/2)
    
    # Check values
    assert coord.r == 0.5
    assert coord.theta == np.pi/4
    assert coord.phi == np.pi/2

def test_spherical_coordinate_to_cartesian():
    """Test conversion from spherical to Cartesian coordinates."""
    # Create coordinate
    coord = SphericalCoordinate(r=1.0, theta=0.0, phi=0.0)
    
    # Convert to Cartesian
    cart = coord.to_cartesian()
    
    # Check values (should be [0, 0, 1])
    assert np.isclose(cart[0], 0.0)
    assert np.isclose(cart[1], 0.0)
    assert np.isclose(cart[2], 1.0)

def test_spherical_coordinate_from_cartesian():
    """Test conversion from Cartesian to spherical coordinates."""
    # Create Cartesian coordinates
    cart = np.array([0.0, 0.0, 1.0])
    
    # Convert to spherical
    coord = SphericalCoordinate.from_cartesian(cart)
    
    # Check values (should be r=1, theta=0, phi=0)
    assert np.isclose(coord.r, 1.0)
    assert np.isclose(coord.theta, 0.0)
    assert np.isclose(coord.phi, 0.0)

def test_spherical_coordinate_str():
    """Test string representation of spherical coordinates."""
    # Create coordinate
    coord = SphericalCoordinate(r=0.5, theta=np.pi/4, phi=np.pi/2)
    
    # Check string representation
    assert str(coord) == f"SphericalCoordinate(r=0.5, theta={np.pi/4}, phi={np.pi/2})"

# Tests for BlochSphereUniverse
def test_universe_add_concept(universe, sample_concepts):
    """Test adding concepts to the universe."""
    # Add concepts
    for name, position in sample_concepts.items():
        universe.add_concept(name, position)
    
    # Check if concepts were added
    for name, position in sample_concepts.items():
        assert name in universe.concepts
        assert universe.get_concept_position(name) == position

def test_universe_remove_concept(universe, sample_concepts):
    """Test removing concepts from the universe."""
    # Add concepts
    for name, position in sample_concepts.items():
        universe.add_concept(name, position)
    
    # Remove a concept
    universe.remove_concept("concept1")
    
    # Check if concept was removed
    assert "concept1" not in universe.concepts
    assert "concept2" in universe.concepts
    assert "concept3" in universe.concepts

def test_universe_add_relationship(universe, sample_concepts):
    """Test adding relationships between concepts."""
    # Add concepts
    for name, position in sample_concepts.items():
        universe.add_concept(name, position)
    
    # Add relationships
    universe.add_relationship("concept1", "concept2", "and")
    universe.add_relationship("concept1", "concept3", "or")
    universe.add_relationship("concept2", "concept3", "not")
    
    # Check if relationships were added
    assert universe.get_relationship_type("concept1", "concept2") == "and"
    assert universe.get_relationship_type("concept1", "concept3") == "or"
    assert universe.get_relationship_type("concept2", "concept3") == "not"

def test_universe_remove_relationship(universe, sample_concepts):
    """Test removing relationships between concepts."""
    # Add concepts
    for name, position in sample_concepts.items():
        universe.add_concept(name, position)
    
    # Add relationships
    universe.add_relationship("concept1", "concept2", "and")
    universe.add_relationship("concept1", "concept3", "or")
    
    # Remove a relationship
    universe.remove_relationship("concept1", "concept2")
    
    # Check if relationship was removed
    assert universe.get_relationship_type("concept1", "concept2") is None
    assert universe.get_relationship_type("concept1", "concept3") == "or"

def test_universe_get_related_concepts(universe, sample_concepts):
    """Test getting related concepts."""
    # Add concepts
    for name, position in sample_concepts.items():
        universe.add_concept(name, position)
    
    # Add relationships
    universe.add_relationship("concept1", "concept2", "and")
    universe.add_relationship("concept1", "concept3", "or")
    
    # Get related concepts
    related = universe.get_related_concepts("concept1")
    
    # Check related concepts
    assert set(related) == {"concept2", "concept3"}
    
    # Get related concepts by relationship type
    and_related = universe.get_related_concepts("concept1", "and")
    or_related = universe.get_related_concepts("concept1", "or")
    
    # Check related concepts by type
    assert set(and_related) == {"concept2"}
    assert set(or_related) == {"concept3"}

def test_universe_calculate_angular_distance(universe):
    """Test calculating angular distance between positions."""
    # Create positions
    pos1 = SphericalCoordinate(r=1.0, theta=0.0, phi=0.0)
    pos2 = SphericalCoordinate(r=1.0, theta=np.pi/2, phi=0.0)
    
    # Calculate distance
    distance = universe.calculate_angular_distance(pos1, pos2)
    
    # Check distance (should be pi/2)
    assert np.isclose(distance, np.pi/2)

    def test_universe_get_concept_cluster(universe, sample_concepts):
        """Test getting concept clusters."""
        # Add concepts
        for name, position in sample_concepts.items():
            universe.add_concept(name, position)
        
        # Get cluster with small distance
        small_cluster = universe.get_concept_cluster("concept1", np.pi/8)  # Use a smaller distance
        
        # Check small cluster (should only include concept1)
        assert set(small_cluster) == {"concept1"}
        
        # Get cluster with larger distance
        large_cluster = universe.get_concept_cluster("concept1", np.pi)
        
        # Check large cluster (should include all concepts)
        assert set(large_cluster) == {"concept1", "concept2", "concept3"}

# Tests for NullGradientManager
def test_null_gradient_calculate_null_field(null_gradient):
    """Test calculating null field."""
    # Create position
    position = SphericalCoordinate(r=0.5, theta=0.0, phi=0.0)
    
    # Calculate null field
    field = null_gradient.calculate_null_field(position)
    
    # Check field
    assert "null_intensity" in field
    assert "null_gradient" in field
    assert isinstance(field["null_intensity"], float)
    assert isinstance(field["null_gradient"], list)
    assert len(field["null_gradient"]) == 3

# Tests for SphericalRelationshipVectorizer
@pytest.mark.asyncio
async def test_vectorizer_vectorize_concept(vectorizer):
    """Test vectorizing a concept."""
    # Vectorize concept
    position = await vectorizer.vectorize_concept("test_concept")
    
    # Check position
    assert isinstance(position, SphericalCoordinate)
    assert 0.2 <= position.r <= 0.4
    assert 0 <= position.theta <= 2 * np.pi
    assert 0 <= position.phi <= np.pi
    
    # Check if concept was added to universe
    assert "test_concept" in vectorizer.universe.concepts

@pytest.mark.asyncio
async def test_vectorizer_vectorize_with_constraints(vectorizer):
    """Test vectorizing a concept with constraints."""
    # Add a concept to be near
    await vectorizer.vectorize_concept("near_concept")
    
    # Define constraints
    constraints = {
        "min_radius": 0.3,
        "max_radius": 0.4,
        "near_concept": "near_concept",
        "max_angle": np.pi/4,
        "relationship_type": "and"
    }
    
    # Vectorize with constraints
    position = await vectorizer.vectorize_with_constraints("test_concept", constraints)
    
    # Check position
    assert isinstance(position, SphericalCoordinate)
    assert 0.3 <= position.r <= 0.4
    
    # Check if concept was added to universe
    assert "test_concept" in vectorizer.universe.concepts
    
    # Check if relationship was added
    assert vectorizer.universe.get_relationship_type("test_concept", "near_concept") == "and"

@pytest.mark.asyncio
async def test_vectorizer_get_nearest_concepts(vectorizer, sample_concepts):
    """Test getting nearest concepts."""
    # Add concepts
    for name, position in sample_concepts.items():
        vectorizer.universe.add_concept(name, position)
    
    # Get nearest concepts
    nearest = await vectorizer.get_nearest_concepts(sample_concepts["concept1"], 2)
    
    # Check nearest concepts
    assert len(nearest) == 2
    assert nearest[0][0] == "concept1"  # First should be the concept itself
    assert nearest[1][0] in ["concept2", "concept3"]

@pytest.mark.asyncio
async def test_vectorizer_interpolate_concepts(vectorizer, sample_concepts):
    """Test interpolating between concepts."""
    # Add concepts
    for name, position in sample_concepts.items():
        vectorizer.universe.add_concept(name, position)
    
    # Interpolate
    interp_pos = await vectorizer.interpolate_concepts("concept1", "concept2", 0.5)
    
    # Check interpolated position
    assert isinstance(interp_pos, SphericalCoordinate)
    
    # Should be between concept1 and concept2
    cart1 = sample_concepts["concept1"].to_cartesian()
    cart2 = sample_concepts["concept2"].to_cartesian()
    interp_cart = interp_pos.to_cartesian()
    
    # Check if interpolated position is roughly in the middle
    assert np.allclose(interp_cart, (cart1 + cart2) / 2, atol=0.1)

# Tests for RelativeTypeSystem
@pytest.mark.asyncio
async def test_type_system_create_relative_hierarchy(type_system, sample_concepts):
    """Test creating a relative type hierarchy."""
    # Add concepts
    for name, position in sample_concepts.items():
        type_system.universe.add_concept(name, position)
    
    # Add relationships
    type_system.universe.add_relationship("concept1", "concept2", "and")
    type_system.universe.add_relationship("concept1", "concept3", "or")
    
    # Create hierarchy
    hierarchy = await type_system.create_relative_hierarchy("concept1", sample_concepts["concept1"])
    
    # Check hierarchy
    assert hierarchy.concept == "concept1"
    assert hierarchy.position == sample_concepts["concept1"]
    assert hierarchy.bottom_type == "concept1_type"
    assert hierarchy.top_type == "any_concept1"
    assert hierarchy.unified_type == "unified_type"
    assert len(hierarchy.middle_types) == 2
    assert "concept1_and_concept2" in hierarchy.middle_types
    assert "concept1_or_concept3" in hierarchy.middle_types

@pytest.mark.asyncio
async def test_type_system_calculate_type_boundaries(type_system, sample_concepts):
    """Test calculating type boundaries."""
    # Add concepts
    for name, position in sample_concepts.items():
        type_system.universe.add_concept(name, position)
    
    # Add relationships
    type_system.universe.add_relationship("concept1", "concept2", "and")
    type_system.universe.add_relationship("concept1", "concept3", "or")
    
    # Create hierarchy
    hierarchy = await type_system.create_relative_hierarchy("concept1", sample_concepts["concept1"])
    
    # Calculate boundaries
    boundaries = await type_system.calculate_type_boundaries(hierarchy)
    
    # Check boundaries
    assert len(boundaries) == 5  # bottom, top, unified, and 2 middle types
    assert hierarchy.bottom_type in boundaries
    assert hierarchy.top_type in boundaries
    assert hierarchy.unified_type in boundaries
    assert all(t in boundaries for t in hierarchy.middle_types)

@pytest.mark.asyncio
async def test_type_system_get_type_at_position(type_system, sample_concepts):
    """Test getting type at a position."""
    # Add concepts
    for name, position in sample_concepts.items():
        type_system.universe.add_concept(name, position)
    
    # Create hierarchy
    hierarchy = await type_system.create_relative_hierarchy("concept1", sample_concepts["concept1"])
    
    # Get type at bottom type position
    bottom_type = await type_system.get_type_at_position(hierarchy, sample_concepts["concept1"])
    
    # Check type
    assert bottom_type == hierarchy.bottom_type
    
    # Get type at origin (unified type)
    unified_type = await type_system.get_type_at_position(hierarchy, SphericalCoordinate(r=0.0, theta=0.0, phi=0.0))
    
    # Check type
    assert unified_type == hierarchy.unified_type

@pytest.mark.asyncio
async def test_type_system_get_subtype_relationship(type_system, sample_concepts):
    """Test getting subtype relationships."""
    # Add concepts
    for name, position in sample_concepts.items():
        type_system.universe.add_concept(name, position)
    
    # Create hierarchy
    hierarchy = await type_system.create_relative_hierarchy("concept1", sample_concepts["concept1"])
    
    # Check subtype relationships
    assert await type_system.get_subtype_relationship(hierarchy, hierarchy.bottom_type, hierarchy.top_type)
    assert not await type_system.get_subtype_relationship(hierarchy, hierarchy.top_type, hierarchy.bottom_type)
    assert await type_system.get_subtype_relationship(hierarchy, hierarchy.bottom_type, hierarchy.unified_type)
    assert await type_system.get_subtype_relationship(hierarchy, hierarchy.top_type, hierarchy.unified_type)

# Tests for SphericalUniverseVisualizer
@pytest.mark.asyncio
async def test_visualizer_generate_sphere_visualization(visualizer, sample_concepts):
    """Test generating sphere visualization."""
    # Add concepts
    for name, position in sample_concepts.items():
        visualizer.universe.add_concept(name, position)
    
    # Generate visualization
    vis_path = await visualizer.generate_sphere_visualization()
    
    # Check if file was created
    assert vis_path.exists()
    assert vis_path.suffix == ".html"

@pytest.mark.asyncio
async def test_visualizer_generate_relationship_visualization(visualizer, sample_concepts):
    """Test generating relationship visualization."""
    # Add concepts
    for name, position in sample_concepts.items():
        visualizer.universe.add_concept(name, position)
    
    # Add relationships
    visualizer.universe.add_relationship("concept1", "concept2", "and")
    visualizer.universe.add_relationship("concept1", "concept3", "or")
    
    # Generate visualization
    vis_path = await visualizer.generate_relationship_visualization("concept1")
    
    # Check if file was created
    assert vis_path.exists()
    assert vis_path.suffix == ".html"
    assert "relationship_visualization" in vis_path.name

@pytest.mark.asyncio
async def test_visualizer_generate_type_hierarchy_visualization(visualizer, sample_concepts):
    """Test generating type hierarchy visualization."""
    # Add concepts
    for name, position in sample_concepts.items():
        visualizer.universe.add_concept(name, position)
    
    # Generate visualization
    vis_path = await visualizer.generate_type_hierarchy_visualization("concept1")
    
    # Check if file was created
    assert vis_path.exists()
    assert vis_path.suffix == ".html"
    assert "type_hierarchy_visualization" in vis_path.name

@pytest.mark.asyncio
async def test_visualizer_generate_null_gradient_visualization(visualizer):
    """Test generating null gradient visualization."""
    # Generate visualization
    vis_path = await visualizer.generate_null_gradient_visualization()
    
    # Check if file was created
    assert vis_path.exists()
    assert vis_path.suffix == ".html"
    assert "null_gradient_visualization" in vis_path.name

@pytest.mark.asyncio
async def test_visualizer_generate_concept_cluster_visualization(visualizer, sample_concepts):
    """Test generating concept cluster visualization."""
    # Add concepts
    for name, position in sample_concepts.items():
        visualizer.universe.add_concept(name, position)
    
    # Generate visualization
    vis_path = await visualizer.generate_concept_cluster_visualization("concept1")
    
    # Check if file was created
    assert vis_path.exists()
    assert vis_path.suffix == ".html"
    assert "concept_cluster_visualization" in vis_path.name

@pytest.mark.asyncio
async def test_visualizer_generate_nearest_concepts_visualization(visualizer, sample_concepts):
    """Test generating nearest concepts visualization."""
    # Add concepts
    for name, position in sample_concepts.items():
        visualizer.universe.add_concept(name, position)
    
    # Generate visualization
    vis_path = await visualizer.generate_nearest_concepts_visualization("concept1")
    
    # Check if file was created
    assert vis_path.exists()
    assert vis_path.suffix == ".html"
    assert "nearest_concepts_visualization" in vis_path.name

@pytest.mark.asyncio
async def test_visualizer_generate_concept_interpolation_visualization(visualizer, sample_concepts):
    """Test generating concept interpolation visualization."""
    # Add concepts
    for name, position in sample_concepts.items():
        visualizer.universe.add_concept(name, position)
    
    # Generate visualization
    vis_path = await visualizer.generate_concept_interpolation_visualization("concept1", "concept2")
    
    # Check if file was created
    assert vis_path.exists()
    assert vis_path.suffix == ".html"
    assert "concept_interpolation_visualization" in vis_path.name

# Integration tests
@pytest.mark.asyncio
async def test_integration_core_definitions():
    """Test integration with core definitions."""
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
    
    # Add relationships
    for concept, data in CORE_DEFINITIONS.items():
        # Add AND relationships
        for related_concept, strength in data.get("and_relationships", []):
            if related_concept in universe.concepts:
                universe.add_relationship(concept, related_concept, "and")
        
        # Add OR relationships
        for related_concept, strength in data.get("or_relationships", []):
            if related_concept in universe.concepts:
                universe.add_relationship(concept, related_concept, "or")
        
        # Add NOT relationships
        for related_concept, strength in data.get("not_relationships", []):
            if related_concept in universe.concepts:
                universe.add_relationship(concept, related_concept, "not")
    
    # Check if concepts were added
    assert len(universe.concepts) > 0
    assert "concept" in universe.concepts
    assert "meaning" in universe.concepts
    
    # Check if relationships were added
    assert universe.get_relationship_type("concept", "meaning") == "and"
    
    # Create type hierarchy
    concept_position = universe.get_concept_position("concept")
    hierarchy = await type_system.create_relative_hierarchy("concept", concept_position)
    
    # Check hierarchy
    assert hierarchy.concept == "concept"
    assert hierarchy.bottom_type == "concept_type"
    assert hierarchy.top_type == "any_concept"
    assert hierarchy.unified_type == "unified_type"
    assert len(hierarchy.middle_types) > 0
    
    # Generate visualization
    vis_path = await visualizer.generate_sphere_visualization(["concept", "meaning", "abstraction"])
    
    # Check if file was created
    assert vis_path.exists()
    assert vis_path.suffix == ".html"
