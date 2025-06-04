"""
Spherical Universe REST API.

This module provides REST API endpoints for interacting with the spherical universe,
including endpoints for concepts, relationships, type hierarchies, and visualizations.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager
from src.core.relative_type_system import RelativeTypeSystem, RelativeTypeHierarchy
from src.neural.spherical_vectorizer import SphericalRelationshipVectorizer
from src.services.sphere_visualization import SphericalUniverseVisualizer
from src.api.dependencies import get_universe, get_null_gradient, get_spherical_vectorizer, get_type_system, get_sphere_visualizer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/spherical",
    tags=["spherical"],
    responses={404: {"description": "Not found"}},
)

# Models for request and response
class SphericalCoordinateModel(BaseModel):
    """Model for spherical coordinates."""
    r: float = Field(..., description="Radius")
    theta: float = Field(..., description="Azimuthal angle (longitude)")
    phi: float = Field(..., description="Polar angle (latitude)")

class ConceptModel(BaseModel):
    """Model for a concept."""
    name: str = Field(..., description="Concept name")
    position: SphericalCoordinateModel = Field(..., description="Concept position")

class RelationshipModel(BaseModel):
    """Model for a relationship between concepts."""
    concept1: str = Field(..., description="First concept name")
    concept2: str = Field(..., description="Second concept name")
    relationship_type: str = Field(..., description="Relationship type (and, or, not)")

class TypeHierarchyModel(BaseModel):
    """Model for a type hierarchy."""
    concept: str = Field(..., description="Concept name")
    bottom_type: str = Field(..., description="Bottom type")
    top_type: str = Field(..., description="Top type")
    unified_type: str = Field(..., description="Unified type")
    middle_types: List[str] = Field(default_factory=list, description="Middle types")
    subtype_relationships: Dict[str, bool] = Field(default_factory=dict, description="Subtype relationships")

class VisualizationModel(BaseModel):
    """Model for a visualization."""
    visualization_type: str = Field(..., description="Visualization type")
    file_path: str = Field(..., description="File path")
    url: str = Field(..., description="URL")

class ConceptConstraintsModel(BaseModel):
    """Model for concept vectorization constraints."""
    min_radius: Optional[float] = Field(None, description="Minimum radius")
    max_radius: Optional[float] = Field(None, description="Maximum radius")
    near_concept: Optional[str] = Field(None, description="Concept to be near")
    max_angle: Optional[float] = Field(None, description="Maximum angle from near_concept")
    opposite_concept: Optional[str] = Field(None, description="Concept to be opposite to")
    relationship_type: Optional[str] = Field(None, description="Relationship type with near_concept")

class NullFieldModel(BaseModel):
    """Model for null field."""
    null_intensity: float = Field(..., description="Null intensity")
    null_gradient: List[float] = Field(..., description="Null gradient")

# Endpoints for concepts
@router.get("/concepts", response_model=List[ConceptModel])
async def get_concepts(
    universe: BlochSphereUniverse = Depends(get_universe)
) -> List[ConceptModel]:
    """Get all concepts."""
    logger.info("Getting all concepts")
    
    concepts = []
    
    for name, position in universe.concepts.items():
        concepts.append(ConceptModel(
            name=name,
            position=SphericalCoordinateModel(
                r=position.r,
                theta=position.theta,
                phi=position.phi
            )
        ))
    
    return concepts

@router.get("/concepts/{concept_name}", response_model=ConceptModel)
async def get_concept(
    concept_name: str = Path(..., description="Concept name"),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> ConceptModel:
    """Get a concept by name."""
    logger.info(f"Getting concept '{concept_name}'")
    
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    position = universe.get_concept_position(concept_name)
    
    return ConceptModel(
        name=concept_name,
        position=SphericalCoordinateModel(
            r=position.r,
            theta=position.theta,
            phi=position.phi
        )
    )

@router.post("/concepts", response_model=ConceptModel)
async def create_concept(
    concept: ConceptModel = Body(..., description="Concept to create"),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> ConceptModel:
    """Create a new concept."""
    logger.info(f"Creating concept '{concept.name}'")
    
    if concept.name in universe.concepts:
        raise HTTPException(status_code=400, detail=f"Concept '{concept.name}' already exists")
    
    position = SphericalCoordinate(
        r=concept.position.r,
        theta=concept.position.theta,
        phi=concept.position.phi
    )
    
    universe.add_concept(concept.name, position)
    
    return concept

@router.post("/concepts/vectorize", response_model=ConceptModel)
async def vectorize_concept(
    concept_name: str = Body(..., description="Concept name"),
    constraints: Optional[ConceptConstraintsModel] = Body(None, description="Vectorization constraints"),
    vectorizer: SphericalRelationshipVectorizer = Depends(get_spherical_vectorizer)
) -> ConceptModel:
    """Vectorize a concept."""
    logger.info(f"Vectorizing concept '{concept_name}'")
    
    if constraints:
        # Convert constraints to dict
        constraints_dict = constraints.dict(exclude_none=True)
        
        # Vectorize with constraints
        position = await vectorizer.vectorize_with_constraints(concept_name, constraints_dict)
    else:
        # Vectorize without constraints
        position = await vectorizer.vectorize_concept(concept_name)
    
    return ConceptModel(
        name=concept_name,
        position=SphericalCoordinateModel(
            r=position.r,
            theta=position.theta,
            phi=position.phi
        )
    )

@router.delete("/concepts/{concept_name}")
async def delete_concept(
    concept_name: str = Path(..., description="Concept name"),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> Dict[str, str]:
    """Delete a concept."""
    logger.info(f"Deleting concept '{concept_name}'")
    
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    universe.remove_concept(concept_name)
    
    return {"message": f"Concept '{concept_name}' deleted"}

# Endpoints for relationships
@router.get("/relationships", response_model=List[RelationshipModel])
async def get_relationships(
    universe: BlochSphereUniverse = Depends(get_universe)
) -> List[RelationshipModel]:
    """Get all relationships."""
    logger.info("Getting all relationships")
    
    relationships = []
    
    for (concept1, concept2), rel_type in universe.relationships.items():
        relationships.append(RelationshipModel(
            concept1=concept1,
            concept2=concept2,
            relationship_type=rel_type
        ))
    
    return relationships

@router.get("/relationships/{concept_name}", response_model=List[RelationshipModel])
async def get_concept_relationships(
    concept_name: str = Path(..., description="Concept name"),
    relationship_type: Optional[str] = Query(None, description="Relationship type (and, or, not)"),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> List[RelationshipModel]:
    """Get relationships for a concept."""
    logger.info(f"Getting relationships for concept '{concept_name}'")
    
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    relationships = []
    
    # Get related concepts
    related_concepts = universe.get_related_concepts(concept_name, relationship_type)
    
    for related in related_concepts:
        rel_type = universe.get_relationship_type(concept_name, related)
        
        relationships.append(RelationshipModel(
            concept1=concept_name,
            concept2=related,
            relationship_type=rel_type
        ))
    
    return relationships

@router.post("/relationships", response_model=RelationshipModel)
async def create_relationship(
    relationship: RelationshipModel = Body(..., description="Relationship to create"),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> RelationshipModel:
    """Create a new relationship."""
    logger.info(f"Creating relationship: {relationship.concept1} {relationship.relationship_type} {relationship.concept2}")
    
    # Check if concepts exist
    if relationship.concept1 not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{relationship.concept1}' not found")
    
    if relationship.concept2 not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{relationship.concept2}' not found")
    
    # Check if relationship type is valid
    if relationship.relationship_type not in ["and", "or", "not"]:
        raise HTTPException(status_code=400, detail=f"Invalid relationship type '{relationship.relationship_type}'")
    
    # Add relationship
    universe.add_relationship(relationship.concept1, relationship.concept2, relationship.relationship_type)
    
    return relationship

@router.delete("/relationships")
async def delete_relationship(
    concept1: str = Query(..., description="First concept name"),
    concept2: str = Query(..., description="Second concept name"),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> Dict[str, str]:
    """Delete a relationship."""
    logger.info(f"Deleting relationship between '{concept1}' and '{concept2}'")
    
    # Check if concepts exist
    if concept1 not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept1}' not found")
    
    if concept2 not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept2}' not found")
    
    # Check if relationship exists
    if (concept1, concept2) not in universe.relationships and (concept2, concept1) not in universe.relationships:
        raise HTTPException(status_code=404, detail=f"Relationship between '{concept1}' and '{concept2}' not found")
    
    # Remove relationship
    universe.remove_relationship(concept1, concept2)
    
    return {"message": f"Relationship between '{concept1}' and '{concept2}' deleted"}

# Endpoints for type hierarchies
@router.get("/type-hierarchies/{concept_name}", response_model=TypeHierarchyModel)
async def get_type_hierarchy(
    concept_name: str = Path(..., description="Concept name"),
    type_system: RelativeTypeSystem = Depends(get_type_system),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> TypeHierarchyModel:
    """Get type hierarchy for a concept."""
    logger.info(f"Getting type hierarchy for concept '{concept_name}'")
    
    # Check if concept exists
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    # Get concept position
    position = universe.get_concept_position(concept_name)
    
    # Create type hierarchy
    hierarchy = await type_system.create_relative_hierarchy(concept_name, position)
    
    # Convert subtype relationships to dict
    subtype_relationships = {}
    
    for (t1, t2), is_subtype in hierarchy.subtype_relationships.items():
        subtype_relationships[f"{t1},{t2}"] = is_subtype
    
    return TypeHierarchyModel(
        concept=concept_name,
        bottom_type=hierarchy.bottom_type,
        top_type=hierarchy.top_type,
        unified_type=hierarchy.unified_type,
        middle_types=hierarchy.middle_types,
        subtype_relationships=subtype_relationships
    )

@router.get("/type-boundaries/{concept_name}")
async def get_type_boundaries(
    concept_name: str = Path(..., description="Concept name"),
    type_system: RelativeTypeSystem = Depends(get_type_system),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> Dict[str, SphericalCoordinateModel]:
    """Get type boundaries for a concept."""
    logger.info(f"Getting type boundaries for concept '{concept_name}'")
    
    # Check if concept exists
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    # Get concept position
    position = universe.get_concept_position(concept_name)
    
    # Create type hierarchy
    hierarchy = await type_system.create_relative_hierarchy(concept_name, position)
    
    # Calculate type boundaries
    boundaries = await type_system.calculate_type_boundaries(hierarchy)
    
    # Convert to response model
    response = {}
    
    for type_name, boundary in boundaries.items():
        response[type_name] = SphericalCoordinateModel(
            r=boundary.r,
            theta=boundary.theta,
            phi=boundary.phi
        )
    
    return response

@router.get("/type-at-position/{concept_name}")
async def get_type_at_position(
    concept_name: str = Path(..., description="Concept name"),
    r: float = Query(..., description="Radius"),
    theta: float = Query(..., description="Azimuthal angle (longitude)"),
    phi: float = Query(..., description="Polar angle (latitude)"),
    type_system: RelativeTypeSystem = Depends(get_type_system),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> Dict[str, str]:
    """Get type at a position relative to a concept's type hierarchy."""
    logger.info(f"Getting type at position ({r}, {theta}, {phi}) relative to concept '{concept_name}'")
    
    # Check if concept exists
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    # Get concept position
    concept_position = universe.get_concept_position(concept_name)
    
    # Create position
    position = SphericalCoordinate(r=r, theta=theta, phi=phi)
    
    # Create type hierarchy
    hierarchy = await type_system.create_relative_hierarchy(concept_name, concept_position)
    
    # Get type at position
    type_at_position = await type_system.get_type_at_position(hierarchy, position)
    
    return {"type": type_at_position}

# Endpoints for null gradient
@router.get("/null-field")
async def get_null_field(
    r: float = Query(..., description="Radius"),
    theta: float = Query(..., description="Azimuthal angle (longitude)"),
    phi: float = Query(..., description="Polar angle (latitude)"),
    null_gradient: NullGradientManager = Depends(get_null_gradient)
) -> NullFieldModel:
    """Get null field at a position."""
    logger.info(f"Getting null field at position ({r}, {theta}, {phi})")
    
    # Create position
    position = SphericalCoordinate(r=r, theta=theta, phi=phi)
    
    # Calculate null field
    field = null_gradient.calculate_null_field(position)
    
    return NullFieldModel(
        null_intensity=field["null_intensity"],
        null_gradient=field["null_gradient"]
    )

# Endpoints for visualizations
@router.get("/visualizations/sphere", response_model=VisualizationModel)
async def generate_sphere_visualization(
    concepts: Optional[List[str]] = Query(None, description="Concepts to visualize"),
    visualizer: SphericalUniverseVisualizer = Depends(get_sphere_visualizer)
) -> VisualizationModel:
    """Generate a visualization of the spherical universe."""
    logger.info("Generating sphere visualization")
    
    # Generate visualization
    vis_path = await visualizer.generate_sphere_visualization(concepts)
    
    # Convert to URL
    url = f"/visualizations/{vis_path.name}"
    
    return VisualizationModel(
        visualization_type="sphere",
        file_path=str(vis_path),
        url=url
    )

@router.get("/visualizations/relationship/{concept_name}", response_model=VisualizationModel)
async def generate_relationship_visualization(
    concept_name: str = Path(..., description="Concept name"),
    visualizer: SphericalUniverseVisualizer = Depends(get_sphere_visualizer),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> VisualizationModel:
    """Generate a visualization of relationships for a concept."""
    logger.info(f"Generating relationship visualization for concept '{concept_name}'")
    
    # Check if concept exists
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    # Generate visualization
    vis_path = await visualizer.generate_relationship_visualization(concept_name)
    
    # Convert to URL
    url = f"/visualizations/{vis_path.name}"
    
    return VisualizationModel(
        visualization_type="relationship",
        file_path=str(vis_path),
        url=url
    )

@router.get("/visualizations/type-hierarchy/{concept_name}", response_model=VisualizationModel)
async def generate_type_hierarchy_visualization(
    concept_name: str = Path(..., description="Concept name"),
    visualizer: SphericalUniverseVisualizer = Depends(get_sphere_visualizer),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> VisualizationModel:
    """Generate a visualization of the type hierarchy for a concept."""
    logger.info(f"Generating type hierarchy visualization for concept '{concept_name}'")
    
    # Check if concept exists
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    # Generate visualization
    vis_path = await visualizer.generate_type_hierarchy_visualization(concept_name)
    
    # Convert to URL
    url = f"/visualizations/{vis_path.name}"
    
    return VisualizationModel(
        visualization_type="type_hierarchy",
        file_path=str(vis_path),
        url=url
    )

@router.get("/visualizations/null-gradient", response_model=VisualizationModel)
async def generate_null_gradient_visualization(
    visualizer: SphericalUniverseVisualizer = Depends(get_sphere_visualizer)
) -> VisualizationModel:
    """Generate a visualization of the null gradient."""
    logger.info("Generating null gradient visualization")
    
    # Generate visualization
    vis_path = await visualizer.generate_null_gradient_visualization()
    
    # Convert to URL
    url = f"/visualizations/{vis_path.name}"
    
    return VisualizationModel(
        visualization_type="null_gradient",
        file_path=str(vis_path),
        url=url
    )

@router.get("/visualizations/concept-cluster/{concept_name}", response_model=VisualizationModel)
async def generate_concept_cluster_visualization(
    concept_name: str = Path(..., description="Concept name"),
    max_distance: float = Query(1.57, description="Maximum angular distance"),
    visualizer: SphericalUniverseVisualizer = Depends(get_sphere_visualizer),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> VisualizationModel:
    """Generate a visualization of a concept cluster."""
    logger.info(f"Generating concept cluster visualization for concept '{concept_name}'")
    
    # Check if concept exists
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    # Generate visualization
    vis_path = await visualizer.generate_concept_cluster_visualization(concept_name, max_distance)
    
    # Convert to URL
    url = f"/visualizations/{vis_path.name}"
    
    return VisualizationModel(
        visualization_type="concept_cluster",
        file_path=str(vis_path),
        url=url
    )

@router.get("/visualizations/nearest-concepts/{concept_name}", response_model=VisualizationModel)
async def generate_nearest_concepts_visualization(
    concept_name: str = Path(..., description="Concept name"),
    count: int = Query(5, description="Number of nearest concepts"),
    visualizer: SphericalUniverseVisualizer = Depends(get_sphere_visualizer),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> VisualizationModel:
    """Generate a visualization of the nearest concepts to a concept."""
    logger.info(f"Generating nearest concepts visualization for concept '{concept_name}'")
    
    # Check if concept exists
    if concept_name not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    # Generate visualization
    vis_path = await visualizer.generate_nearest_concepts_visualization(concept_name, count)
    
    # Convert to URL
    url = f"/visualizations/{vis_path.name}"
    
    return VisualizationModel(
        visualization_type="nearest_concepts",
        file_path=str(vis_path),
        url=url
    )

@router.get("/visualizations/concept-interpolation", response_model=VisualizationModel)
async def generate_concept_interpolation_visualization(
    concept1: str = Query(..., description="First concept name"),
    concept2: str = Query(..., description="Second concept name"),
    steps: int = Query(5, description="Number of interpolation steps"),
    visualizer: SphericalUniverseVisualizer = Depends(get_sphere_visualizer),
    universe: BlochSphereUniverse = Depends(get_universe)
) -> VisualizationModel:
    """Generate a visualization of the interpolation between two concepts."""
    logger.info(f"Generating concept interpolation visualization for concepts '{concept1}' and '{concept2}'")
    
    # Check if concepts exist
    if concept1 not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept1}' not found")
    
    if concept2 not in universe.concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept2}' not found")
    
    # Generate visualization
    vis_path = await visualizer.generate_concept_interpolation_visualization(concept1, concept2, steps)
    
    # Convert to URL
    url = f"/visualizations/{vis_path.name}"
    
    return VisualizationModel(
        visualization_type="concept_interpolation",
        file_path=str(vis_path),
        url=url
    )
