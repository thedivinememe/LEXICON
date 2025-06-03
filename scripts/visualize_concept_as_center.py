"""
Visualize Concept as Center.

This script visualizes a concept as the subjective center of the Bloch sphere,
recalculating all other concepts' positions relative to this central concept.
"""

import asyncio
import logging
import os
import sys
import argparse
import webbrowser
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager
from src.core.relative_type_system import RelativeTypeSystem
from src.neural.spherical_vectorizer import SphericalRelationshipVectorizer
from src.services.sphere_visualization import SphericalUniverseVisualizer
from src.data.core_definitions import CORE_DEFINITIONS
from src.examples.spherical_integration_example import initialize_universe

import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ConceptCentricVisualizer:
    """
    Visualizer that places a specific concept at the center of the universe.
    
    This visualizer recalculates all other concepts' positions relative to
    the central concept, creating a subjective view of the universe from
    the perspective of that concept.
    """
    
    def _format_definition_for_hover(self, concept: str, definition: Dict[str, Any]) -> str:
        """
        Format a concept definition for hover display.
        
        Args:
            concept: Concept name
            definition: Concept definition dictionary
            
        Returns:
            Formatted hover text
        """
        # Start with concept name
        hover_text = f"<b>{concept}</b><br>"
        
        # Add key properties
        if "atomic_pattern" in definition:
            hover_text += f"<b>Pattern:</b> {definition['atomic_pattern']}<br>"
        
        if "description" in definition:
            hover_text += f"<b>Description:</b> {definition['description']}<br>"
        
        # Add relationship information
        if "and_relationships" in definition and definition["and_relationships"]:
            and_rels = ", ".join([f"{rel[0]} ({rel[1]})" for rel in definition["and_relationships"]])
            hover_text += f"<b>AND:</b> {and_rels}<br>"
            
        if "or_relationships" in definition and definition["or_relationships"]:
            or_rels = ", ".join([f"{rel[0]} ({rel[1]})" for rel in definition["or_relationships"]])
            hover_text += f"<b>OR:</b> {or_rels}<br>"
            
        if "not_relationships" in definition and definition["not_relationships"]:
            not_rels = ", ".join([f"{rel[0]} ({rel[1]})" for rel in definition["not_relationships"]])
            hover_text += f"<b>NOT:</b> {not_rels}<br>"
        
        # Add spherical properties if available
        if "spherical_properties" in definition:
            sp = definition["spherical_properties"]
            hover_text += "<b>Spherical Properties:</b><br>"
            for key, value in sp.items():
                hover_text += f"&nbsp;&nbsp;{key}: {value}<br>"
        
        # Add full JSON representation
        hover_text += "<br><details><summary><b>Full Definition</b></summary>"
        hover_text += f"<pre>{json.dumps(definition, indent=2)}</pre></details>"
        
        return hover_text
    
    def __init__(self, 
                universe: BlochSphereUniverse,
                central_concept: str,
                vis_dir: Path = Path("visualizations")):
        """
        Initialize the visualizer.
        
        Args:
            universe: Spherical universe
            central_concept: Name of the concept to place at the center
            vis_dir: Directory for visualizations
        """
        self.universe = universe
        self.central_concept = central_concept
        self.vis_dir = vis_dir
        
        # Create visualizations directory if it doesn't exist
        self.vis_dir.mkdir(exist_ok=True)
        
        # Check if central concept exists
        if central_concept not in universe.concepts:
            raise ValueError(f"Concept '{central_concept}' not found in universe")
        
        # Get central concept position
        self.central_position = universe.get_concept_position(central_concept)
        
        # Calculate relative positions
        self.relative_positions = self._calculate_relative_positions()
    
    def _calculate_relative_positions(self) -> Dict[str, SphericalCoordinate]:
        """
        Calculate positions of all concepts relative to the central concept.
        
        Returns:
            Dictionary mapping concept names to relative positions
        """
        relative_positions = {}
        
        # Get central concept cartesian coordinates
        central_cart = self.central_position.to_cartesian()
        
        for concept, position in self.universe.concepts.items():
            if concept == self.central_concept:
                # Place central concept at the origin
                relative_positions[concept] = SphericalCoordinate(r=0.0, theta=0.0, phi=0.0)
            else:
                # Calculate relative position
                concept_cart = position.to_cartesian()
                
                # Vector from central concept to this concept
                relative_cart = concept_cart - central_cart
                
                # Convert back to spherical
                relative_position = SphericalCoordinate.from_cartesian(relative_cart)
                
                relative_positions[concept] = relative_position
        
        return relative_positions
    
    def generate_visualization(self) -> Path:
        """
        Generate a visualization with the central concept at the center.
        
        Returns:
            Path to the visualization file
        """
        logger.info(f"Generating visualization with '{self.central_concept}' as center")
        
        # Create figure
        fig = go.Figure()
        
        # Add sphere surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 0.5 * np.outer(np.cos(u), np.sin(v))
        y = 0.5 * np.outer(np.sin(u), np.sin(v))
        z = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            colorscale="Blues",
            showscale=False
        ))
        
        # Format the central concept's definition for hover
        central_def = CORE_DEFINITIONS.get(self.central_concept, {})
        central_hover_text = self._format_definition_for_hover(self.central_concept, central_def)
        
        # Add central concept
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            text=[self.central_concept],
            mode="markers+text",
            marker=dict(
                size=20,
                color="red",
                opacity=0.8
            ),
            textposition="top center",
            hovertext=[central_hover_text],
            hoverinfo="text"
        ))
        
        # Add other concepts
        x_vals = []
        y_vals = []
        z_vals = []
        labels = []
        hover_texts = []
        colors = []
        sizes = []
        
        for concept, position in self.relative_positions.items():
            if concept != self.central_concept:
                cart = position.to_cartesian()
                
                x_vals.append(cart[0])
                y_vals.append(cart[1])
                z_vals.append(cart[2])
                labels.append(concept)
                
                # Format definition for hover
                concept_def = CORE_DEFINITIONS.get(concept, {})
                hover_text = self._format_definition_for_hover(concept, concept_def)
                hover_texts.append(hover_text)
                
                # Color based on radius
                colors.append(position.r)
                
                # Size based on relationship to central concept
                rel_type = self.universe.get_relationship_type(self.central_concept, concept)
                if rel_type == "and":
                    sizes.append(15)
                elif rel_type == "or":
                    sizes.append(12)
                elif rel_type == "not":
                    sizes.append(10)
                else:
                    sizes.append(8)
        
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            text=labels,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="Viridis",
                colorbar=dict(
                    title="Distance from Center"
                ),
                opacity=0.8
            ),
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text"
        ))
        
        # Add relationships
        for concept in self.universe.concepts:
            if concept == self.central_concept:
                # Add relationships from central concept
                related = self.universe.get_related_concepts(concept)
                
                for related_concept in related:
                    rel_type = self.universe.get_relationship_type(concept, related_concept)
                    
                    # Get positions
                    rel_pos = self.relative_positions[related_concept]
                    rel_cart = rel_pos.to_cartesian()
                    
                    # Create line
                    x_line = [0, rel_cart[0]]
                    y_line = [0, rel_cart[1]]
                    z_line = [0, rel_cart[2]]
                    
                    # Set color based on relationship type
                    if rel_type == "and":
                        color = "green"
                    elif rel_type == "or":
                        color = "blue"
                    elif rel_type == "not":
                        color = "red"
                    else:
                        color = "gray"
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_line,
                        y=y_line,
                        z=z_line,
                        mode="lines",
                        line=dict(
                            color=color,
                            width=3
                        ),
                        hoverinfo="none"
                    ))
        
        # Set layout
        fig.update_layout(
            title=f"Bloch Sphere with '{self.central_concept}' as Subjective Center",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )
        
        # Save visualization - sanitize concept name for file path
        # Replace any special characters that could cause issues in filenames
        safe_concept_name = self.central_concept
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '&']:
            safe_concept_name = safe_concept_name.replace(char, '_')
        vis_path = self.vis_dir / f"{safe_concept_name}_as_center_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated visualization with '{self.central_concept}' as center: {vis_path}")
        
        return vis_path

async def visualize_concept_as_center(concept: str, open_browser: bool = True) -> Path:
    """
    Visualize a concept as the center of the Bloch sphere.
    
    Args:
        concept: Concept name
        open_browser: Whether to open the visualization in a browser
        
    Returns:
        Path to the visualization file
    """
    logger.info(f"Visualizing concept '{concept}' as center")
    
    # Initialize universe
    universe, null_gradient, vectorizer, type_system, visualizer = await initialize_universe()
    
    # Check if concept exists
    if concept not in universe.concepts:
        raise ValueError(f"Concept '{concept}' not found in universe")
    
    # Create concept-centric visualizer
    concept_visualizer = ConceptCentricVisualizer(universe, concept)
    
    # Generate visualization
    vis_path = concept_visualizer.generate_visualization()
    
    # Open in browser if requested
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(vis_path)}")
    
    return vis_path

async def visualize_all_concepts_as_centers(open_browser: bool = False) -> Dict[str, Path]:
    """
    Visualize all concepts as centers of the Bloch sphere.
    
    Args:
        open_browser: Whether to open the visualizations in a browser
        
    Returns:
        Dictionary mapping concept names to visualization paths
    """
    logger.info("Visualizing all concepts as centers")
    
    # Initialize universe
    universe, null_gradient, vectorizer, type_system, visualizer = await initialize_universe()
    
    # Get all concepts
    concepts = list(universe.concepts.keys())
    
    # Generate visualizations
    vis_paths = {}
    
    for concept in concepts:
        # Create concept-centric visualizer
        concept_visualizer = ConceptCentricVisualizer(universe, concept)
        
        # Generate visualization
        vis_path = concept_visualizer.generate_visualization()
        
        vis_paths[concept] = vis_path
    
    # Create index page
    index_path = create_index_page(vis_paths)
    
    # Open index in browser if requested
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(index_path)}")
    
    return vis_paths

def create_index_page(vis_paths: Dict[str, Path]) -> Path:
    """
    Create an index page for the visualizations.
    
    Args:
        vis_paths: Dictionary mapping concept names to visualization paths
        
    Returns:
        Path to the index page
    """
    logger.info("Creating index page for visualizations")
    
    # Create index HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Concept-Centric Visualizations</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #333;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .concept-list {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .concept-item {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .concept-item:hover {
                background-color: #e0e0e0;
            }
            a {
                text-decoration: none;
                color: #0066cc;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h1>Concept-Centric Visualizations</h1>
        <p>
            Each visualization shows the Bloch sphere with a different concept as the subjective center.
            Click on a concept to view its visualization.
        </p>
        <div class="concept-list">
    """
    
    # Add links to visualizations
    for concept, path in sorted(vis_paths.items()):
        rel_path = os.path.basename(path)
        html += f'        <div class="concept-item"><a href="{rel_path}">{concept}</a></div>\n'
    
    # Close HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    # Save index page
    index_path = Path("visualizations") / "concept_centric_index.html"
    with open(index_path, "w") as f:
        f.write(html)
    
    logger.info(f"Created index page: {index_path}")
    
    return index_path

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize a concept as the center of the Bloch sphere")
    
    # Visualization options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--concept", type=str, help="Concept to visualize as center")
    group.add_argument("--all", action="store_true", help="Visualize all concepts as centers")
    
    # Other options
    parser.add_argument("--open", action="store_true", help="Open visualization in browser")
    
    args = parser.parse_args()
    
    # Run visualization
    if args.all:
        asyncio.run(visualize_all_concepts_as_centers(args.open))
    elif args.concept:
        asyncio.run(visualize_concept_as_center(args.concept, args.open))
    else:
        # Default to visualizing "existence" as center
        asyncio.run(visualize_concept_as_center("existence", args.open))

if __name__ == "__main__":
    main()
