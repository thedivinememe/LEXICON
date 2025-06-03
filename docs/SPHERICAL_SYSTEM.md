# LEXICON Spherical Universal Set

This document explains how to use and visualize the LEXICON spherical universal set system.

## Overview

The LEXICON spherical universal set is a topological representation of concepts where:

- **Center (r=0)**: Pure Null (undefined)
- **Surface (r=1)**: Maximum definition (limited to r=0.5 for epistemic humility)
- **Antipodal Points**: Perfect negations (opposite sides of the sphere)
- **Angular Distance**: Determines relationship type (AND, OR, NOT)

## Visualizing the Bloch Sphere

To visualize the Bloch sphere with core concepts:

### Command Line Options

```bash
# Generate all visualizations and open in browser
python scripts/run_spherical_visualization.py --all --open

# Visualize a specific concept
python scripts/run_spherical_visualization.py --concept concept --open

# Visualize interpolation between two concepts
python scripts/run_spherical_visualization.py --interpolate abstraction,concrete --open

# Visualize null gradient
python scripts/run_spherical_visualization.py --null-gradient --open
```

### Using Batch Scripts

#### Windows
```
scripts\run_spherical_visualization.bat
```

#### Unix/Mac
```
chmod +x scripts/run_spherical_visualization.sh
./scripts/run_spherical_visualization.sh
```

This will:
1. Generate multiple 3D visualizations of the Bloch sphere
2. Open them in your default web browser
3. Also generate static PNG images for 2D projections

## Understanding the Visualizations

The system generates several interactive 3D visualizations:

### 1. Sphere Visualization
Shows all concepts positioned in the spherical universe.
- **Colored Points**: Core concepts positioned on the sphere
- **Black Center Point**: The null center (pure undefined)
- **Connecting Lines**: Relationships between concepts
- **Wireframe Sphere**: The universal set boundary

### 2. Null Gradient Visualization
Displays the null field intensity gradient from the center to the surface.
- **Color Gradient**: Represents null intensity (darker = higher null presence)
- **Sphere Surface**: Shows how the null field diminishes at the boundary

### 3. Type Hierarchy Visualization
Displays the relative type hierarchy for a specific concept.
- **Vertical Arrangement**: Shows type hierarchy levels
- **Connecting Lines**: Shows subtype/supertype relationships
- **Color Coding**: Differentiates between types at different levels

### 4. Relationship Visualization
Shows AND/OR/NOT relationships between a concept and related concepts.
- **Red Node**: The central concept being visualized
- **Blue Nodes**: OR relationships (45°-135° angular distance)
- **Green Nodes**: AND relationships (<45° angular distance)
- **Purple Nodes**: NOT relationships (>135° angular distance)

### 5. Concept Interpolation Visualization
Shows the interpolation path between two concepts.
- **Path Points**: Intermediate concepts at different interpolation weights
- **Labels**: Shows interpolation weights (0.0 to 1.0)

### 6. Concept Cluster Visualization
Shows clusters of related concepts based on semantic similarity.
- **Color Clusters**: Different colors represent different concept clusters
- **Spatial Proximity**: Similar concepts are positioned closer together

### 7. Nearest Concepts Visualization
Displays the nearest concepts to a given concept based on angular distance.
- **Central Node**: The concept being analyzed
- **Surrounding Nodes**: The k-nearest concepts
- **Distance Labels**: Shows angular distance in degrees

Concepts are positioned based on:
- **Radius (r)**: Definition level (0 = undefined, 1 = fully defined)
- **Angular Position (θ, φ)**: Relationship to other concepts
- **Growth Pattern**: How the concept evolved from the center

## Adding New Concepts

To add new concepts to the visualization:

1. Edit `src/examples/spherical_integration_example.py`
2. Add new concept definitions using the `centroid_builder.build_concept_from_center()` method
3. Add the new concepts to the `concepts` dictionary
4. Run the visualization script again

Example:
```python
# Define a new concept
truth_result = await centroid_builder.build_concept_from_center(
    concept_name="truth",
    negations=["falsehood", "lie", "deception", "error"],
    growth_pattern="crystalline",
    target_radius=0.45,
    steps=10
)

# Add to concepts dictionary
concepts["truth"] = truth_result.final_position
```

## Testing the System

To run tests for the spherical system:

```
pytest tests/test_spherical_system.py -v
```

This will test:
- Antipodal negation
- Null gradient
- Epistemic humility enforcement
- Spherical relationships
- Concept growth
- Relative type hierarchies
- Unity paths

## Core Components

The spherical system consists of these core components:

1. **BlochSphereUniverse**: The universal set as a Bloch sphere
2. **NullGradientManager**: Manages the gradient from null to not-null
3. **CentroidConceptBuilder**: Builds concepts from the center outward
4. **RelativeTypeSystem**: Creates type hierarchies relative to concepts
5. **ExistenceTypeRegistry**: Manages existence types and their properties
6. **SphericalRelationshipVectorizer**: Generates vectors with spherical embedding
7. **SphericalUniverseVisualizer**: Visualizes the spherical universe

## Relationship Types

Relationships between concepts are determined by angular distance:

- **AND (&&)**: < 45° - Concepts that co-exist
- **OR (||)**: 45° - 135° - Alternative concepts
- **NOT (!)**: > 135° - Negation relationships

## Growth Patterns

Concepts can grow from the center using different patterns:

- **Radial**: Grows outward equally in all directions
- **Spiral**: Grows in a spiral pattern with angular momentum
- **Branching**: Grows with bifurcations (tree-like)
- **Crystalline**: Grows with regular geometric patterns

## Existence Types

The system includes a hierarchy of existence types:

1. **VOID** (Level 0): Pure non-existence
2. **POTENTIAL** (Level 1): Quantum superposition
3. **PHYSICAL** (Level 2): Material existence
4. **ENERGETIC** (Level 3): Energy patterns
5. **INFORMATIONAL** (Level 4): Information patterns
6. **BIOLOGICAL** (Level 5): Life patterns
7. **MENTAL** (Level 6): Thought patterns
8. **LINGUISTIC** (Level 7): Language patterns
9. **SOCIAL** (Level 8): Social patterns
10. **CONSCIOUS** (Level 9): Consciousness
11. **TRANSCENDENT** (Level 10): Beyond individual consciousness

Each type has specific properties and preferred positions in the spherical space.
