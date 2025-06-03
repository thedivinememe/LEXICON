"""
Spherical Universe Visualization.

This module provides visualization capabilities for the spherical universe,
including 3D visualizations of concepts, relationships, and type hierarchies.
"""

import asyncio
import logging
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union

from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager
from src.core.relative_type_system import RelativeTypeSystem, RelativeTypeHierarchy
from src.neural.spherical_vectorizer import SphericalRelationshipVectorizer

logger = logging.getLogger(__name__)

class SphericalUniverseVisualizer:
    """
    Visualizer for the spherical universe.
    
    Provides methods for generating 3D visualizations of concepts,
    relationships, type hierarchies, and other aspects of the
    spherical universe.
    """
    
    def __init__(self, 
                universe: Optional[BlochSphereUniverse] = None,
                null_gradient: Optional[NullGradientManager] = None,
                vectorizer: Optional[SphericalRelationshipVectorizer] = None,
                type_system: Optional[RelativeTypeSystem] = None):
        """
        Initialize the visualizer.
        
        Args:
            universe: Spherical universe
            null_gradient: Null gradient manager
            vectorizer: Spherical relationship vectorizer
            type_system: Relative type system
        """
        self.universe = universe if universe else BlochSphereUniverse()
        self.null_gradient = null_gradient if null_gradient else NullGradientManager(self.universe)
        self.vectorizer = vectorizer if vectorizer else SphericalRelationshipVectorizer(self.universe, self.null_gradient)
        self.type_system = type_system if type_system else RelativeTypeSystem(self.universe, self.null_gradient)
        
        # Create visualizations directory if it doesn't exist
        self.vis_dir = Path("visualizations")
        self.vis_dir.mkdir(exist_ok=True)
    
    async def generate_sphere_visualization(self, 
                                          concepts: Optional[List[str]] = None) -> Path:
        """
        Generate a 3D visualization of the spherical universe.
        
        Args:
            concepts: List of concepts to visualize (if None, visualize all)
            
        Returns:
            Path to the visualization file
        """
        logger.info("Generating sphere visualization")
        
        # Get concepts to visualize
        if concepts is None:
            concepts = list(self.universe.concepts.keys())
        else:
            # Filter out concepts that don't exist
            concepts = [c for c in concepts if c in self.universe.concepts]
        
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
        
        # Add concepts
        x_vals = []
        y_vals = []
        z_vals = []
        labels = []
        colors = []
        sizes = []
        
        for concept in concepts:
            position = self.universe.get_concept_position(concept)
            cart = position.to_cartesian()
            
            x_vals.append(cart[0])
            y_vals.append(cart[1])
            z_vals.append(cart[2])
            labels.append(concept)
            
            # Color based on radius
            colors.append(position.r)
            
            # Size based on number of relationships
            related = self.universe.get_related_concepts(concept)
            sizes.append(10 + 5 * len(related))
        
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
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add relationships
        for concept in concepts:
            related = self.universe.get_related_concepts(concept)
            related = [r for r in related if r in concepts]
            
            for related_concept in related:
                rel_type = self.universe.get_relationship_type(concept, related_concept)
                
                # Get positions
                pos1 = self.universe.get_concept_position(concept)
                pos2 = self.universe.get_concept_position(related_concept)
                
                cart1 = pos1.to_cartesian()
                cart2 = pos2.to_cartesian()
                
                # Create line
                x_line = [cart1[0], cart2[0]]
                y_line = [cart1[1], cart2[1]]
                z_line = [cart1[2], cart2[2]]
                
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
            title="Spherical Universe Visualization",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )
        
        # Save visualization
        vis_path = self.vis_dir / "sphere_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated sphere visualization: {vis_path}")
        
        return vis_path
    
    async def generate_relationship_visualization(self, concept: str) -> Path:
        """
        Generate a visualization of relationships for a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            Path to the visualization file
            
        Raises:
            ValueError: If concept not found
        """
        logger.info(f"Generating relationship visualization for concept '{concept}'")
        
        # Check if concept exists
        if concept not in self.universe.concepts:
            raise ValueError(f"Concept '{concept}' not found")
        
        # Get related concepts
        related = self.universe.get_related_concepts(concept)
        
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
        
        # Add main concept
        pos = self.universe.get_concept_position(concept)
        cart = pos.to_cartesian()
        
        fig.add_trace(go.Scatter3d(
            x=[cart[0]],
            y=[cart[1]],
            z=[cart[2]],
            text=[concept],
            mode="markers+text",
            marker=dict(
                size=15,
                color="red",
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add related concepts
        x_vals = []
        y_vals = []
        z_vals = []
        labels = []
        colors = []
        
        for related_concept in related:
            rel_type = self.universe.get_relationship_type(concept, related_concept)
            
            # Get position
            rel_pos = self.universe.get_concept_position(related_concept)
            rel_cart = rel_pos.to_cartesian()
            
            x_vals.append(rel_cart[0])
            y_vals.append(rel_cart[1])
            z_vals.append(rel_cart[2])
            labels.append(f"{related_concept} ({rel_type})")
            
            # Color based on relationship type
            if rel_type == "and":
                colors.append("green")
            elif rel_type == "or":
                colors.append("blue")
            elif rel_type == "not":
                colors.append("red")
            else:
                colors.append("gray")
        
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            text=labels,
            mode="markers+text",
            marker=dict(
                size=10,
                color=colors,
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add relationships
        for related_concept in related:
            rel_type = self.universe.get_relationship_type(concept, related_concept)
            
            # Get positions
            rel_pos = self.universe.get_concept_position(related_concept)
            rel_cart = rel_pos.to_cartesian()
            
            # Create line
            x_line = [cart[0], rel_cart[0]]
            y_line = [cart[1], rel_cart[1]]
            z_line = [cart[2], rel_cart[2]]
            
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
            title=f"Relationships for Concept '{concept}'",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )
        
        # Save visualization
        vis_path = self.vis_dir / f"{concept}_relationship_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated relationship visualization for concept '{concept}': {vis_path}")
        
        return vis_path
    
    async def generate_type_hierarchy_visualization(self, concept: str) -> Path:
        """
        Generate a visualization of the type hierarchy for a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            Path to the visualization file
            
        Raises:
            ValueError: If concept not found
        """
        logger.info(f"Generating type hierarchy visualization for concept '{concept}'")
        
        # Check if concept exists
        if concept not in self.universe.concepts:
            raise ValueError(f"Concept '{concept}' not found")
        
        # Get concept position
        position = self.universe.get_concept_position(concept)
        
        # Create type hierarchy
        hierarchy = await self.type_system.create_relative_hierarchy(concept, position)
        
        # Calculate type boundaries
        boundaries = await self.type_system.calculate_type_boundaries(hierarchy)
        
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
        
        # Add type boundaries
        x_vals = []
        y_vals = []
        z_vals = []
        labels = []
        colors = []
        sizes = []
        
        for type_name, boundary in boundaries.items():
            cart = boundary.to_cartesian()
            
            x_vals.append(cart[0])
            y_vals.append(cart[1])
            z_vals.append(cart[2])
            labels.append(type_name)
            
            # Color based on type
            if type_name == hierarchy.bottom_type:
                colors.append("red")
                sizes.append(15)
            elif type_name == hierarchy.top_type:
                colors.append("blue")
                sizes.append(15)
            elif type_name == hierarchy.unified_type:
                colors.append("purple")
                sizes.append(15)
            else:
                colors.append("green")
                sizes.append(10)
        
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            text=labels,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add subtype relationships
        for (t1, t2), is_subtype in hierarchy.subtype_relationships.items():
            if is_subtype and t1 != t2:
                # Get positions
                pos1 = boundaries[t1]
                pos2 = boundaries[t2]
                
                cart1 = pos1.to_cartesian()
                cart2 = pos2.to_cartesian()
                
                # Create line
                x_line = [cart1[0], cart2[0]]
                y_line = [cart1[1], cart2[1]]
                z_line = [cart1[2], cart2[2]]
                
                fig.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode="lines",
                    line=dict(
                        color="gray",
                        width=2
                    ),
                    hoverinfo="none"
                ))
        
        # Set layout
        fig.update_layout(
            title=f"Type Hierarchy for Concept '{concept}'",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )
        
        # Save visualization
        vis_path = self.vis_dir / f"{concept}_type_hierarchy_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated type hierarchy visualization for concept '{concept}': {vis_path}")
        
        return vis_path
    
    async def generate_null_gradient_visualization(self) -> Path:
        """
        Generate a visualization of the null gradient.
        
        Returns:
            Path to the visualization file
        """
        logger.info("Generating null gradient visualization")
        
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
        
        # Create grid of points
        r_vals = np.linspace(0.1, 0.5, 5)
        theta_vals = np.linspace(0, 2 * np.pi, 8)
        phi_vals = np.linspace(0, np.pi, 4)
        
        x_vals = []
        y_vals = []
        z_vals = []
        u_vals = []
        v_vals = []
        w_vals = []
        intensities = []
        
        for r in r_vals:
            for theta in theta_vals:
                for phi in phi_vals:
                    # Create position
                    position = SphericalCoordinate(r=r, theta=theta, phi=phi)
                    cart = position.to_cartesian()
                    
                    # Calculate null field
                    field = self.null_gradient.calculate_null_field(position)
                    
                    x_vals.append(cart[0])
                    y_vals.append(cart[1])
                    z_vals.append(cart[2])
                    
                    u_vals.append(field["null_gradient"][0])
                    v_vals.append(field["null_gradient"][1])
                    w_vals.append(field["null_gradient"][2])
                    
                    intensities.append(field["null_intensity"])
        
        # Add null gradient vectors
        fig.add_trace(go.Cone(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            u=u_vals,
            v=v_vals,
            w=w_vals,
            colorscale="Viridis",
            colorbar=dict(
                title="Null Intensity"
            ),
            sizemode="absolute",
            sizeref=0.05
        ))
        
        # Set layout
        fig.update_layout(
            title="Null Gradient Visualization",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save visualization
        vis_path = self.vis_dir / "null_gradient_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated null gradient visualization: {vis_path}")
        
        return vis_path
    
    async def generate_concept_cluster_visualization(self, 
                                                   concept: str,
                                                   max_distance: float = 1.57) -> Path:
        """
        Generate a visualization of a concept cluster.
        
        Args:
            concept: Concept name
            max_distance: Maximum angular distance
            
        Returns:
            Path to the visualization file
            
        Raises:
            ValueError: If concept not found
        """
        logger.info(f"Generating concept cluster visualization for concept '{concept}'")
        
        # Check if concept exists
        if concept not in self.universe.concepts:
            raise ValueError(f"Concept '{concept}' not found")
        
        # Get concept cluster
        cluster = self.universe.get_concept_cluster(concept, max_distance)
        
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
        
        # Add main concept
        pos = self.universe.get_concept_position(concept)
        cart = pos.to_cartesian()
        
        fig.add_trace(go.Scatter3d(
            x=[cart[0]],
            y=[cart[1]],
            z=[cart[2]],
            text=[concept],
            mode="markers+text",
            marker=dict(
                size=15,
                color="red",
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add cluster concepts
        x_vals = []
        y_vals = []
        z_vals = []
        labels = []
        distances = []
        
        for cluster_concept in cluster:
            if cluster_concept != concept:
                # Get position
                cluster_pos = self.universe.get_concept_position(cluster_concept)
                cluster_cart = cluster_pos.to_cartesian()
                
                x_vals.append(cluster_cart[0])
                y_vals.append(cluster_cart[1])
                z_vals.append(cluster_cart[2])
                labels.append(cluster_concept)
                
                # Calculate distance
                distance = self.universe.calculate_angular_distance(pos, cluster_pos)
                distances.append(distance)
        
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            text=labels,
            mode="markers+text",
            marker=dict(
                size=10,
                color=distances,
                colorscale="Viridis",
                colorbar=dict(
                    title="Angular Distance"
                ),
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add distance sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        # Create sphere centered at concept position with radius max_distance
        sphere_x = cart[0] + max_distance * np.outer(np.cos(u), np.sin(v))
        sphere_y = cart[1] + max_distance * np.outer(np.sin(u), np.sin(v))
        sphere_z = cart[2] + max_distance * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            opacity=0.1,
            colorscale="Reds",
            showscale=False
        ))
        
        # Set layout
        fig.update_layout(
            title=f"Concept Cluster for '{concept}' (Max Distance: {max_distance:.2f})",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save visualization
        vis_path = self.vis_dir / f"{concept}_concept_cluster_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated concept cluster visualization for concept '{concept}': {vis_path}")
        
        return vis_path
    
    async def generate_nearest_concepts_visualization(self, 
                                                    concept: str,
                                                    count: int = 5) -> Path:
        """
        Generate a visualization of the nearest concepts to a concept.
        
        Args:
            concept: Concept name
            count: Number of nearest concepts
            
        Returns:
            Path to the visualization file
            
        Raises:
            ValueError: If concept not found
        """
        logger.info(f"Generating nearest concepts visualization for concept '{concept}'")
        
        # Check if concept exists
        if concept not in self.universe.concepts:
            raise ValueError(f"Concept '{concept}' not found")
        
        # Get concept position
        position = self.universe.get_concept_position(concept)
        
        # Get nearest concepts
        nearest = await self.vectorizer.get_nearest_concepts(position, count + 1)  # +1 because the concept itself will be included
        
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
        
        # Add main concept
        cart = position.to_cartesian()
        
        fig.add_trace(go.Scatter3d(
            x=[cart[0]],
            y=[cart[1]],
            z=[cart[2]],
            text=[concept],
            mode="markers+text",
            marker=dict(
                size=15,
                color="red",
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add nearest concepts
        x_vals = []
        y_vals = []
        z_vals = []
        labels = []
        distances = []
        
        for nearest_concept, distance in nearest:
            if nearest_concept != concept:
                # Get position
                nearest_pos = self.universe.get_concept_position(nearest_concept)
                nearest_cart = nearest_pos.to_cartesian()
                
                x_vals.append(nearest_cart[0])
                y_vals.append(nearest_cart[1])
                z_vals.append(nearest_cart[2])
                labels.append(f"{nearest_concept} ({distance:.2f})")
                distances.append(distance)
                
                # Add line to main concept
                x_line = [cart[0], nearest_cart[0]]
                y_line = [cart[1], nearest_cart[1]]
                z_line = [cart[2], nearest_cart[2]]
                
                fig.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode="lines",
                    line=dict(
                        color="gray",
                        width=2
                    ),
                    hoverinfo="none"
                ))
        
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            text=labels,
            mode="markers+text",
            marker=dict(
                size=10,
                color=distances,
                colorscale="Viridis",
                colorbar=dict(
                    title="Angular Distance"
                ),
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Set layout
        fig.update_layout(
            title=f"Nearest Concepts to '{concept}'",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save visualization
        vis_path = self.vis_dir / f"{concept}_nearest_concepts_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated nearest concepts visualization for concept '{concept}': {vis_path}")
        
        return vis_path
    
    async def generate_concept_interpolation_visualization(self, 
                                                         concept1: str,
                                                         concept2: str,
                                                         steps: int = 5) -> Path:
        """
        Generate a visualization of the interpolation between two concepts.
        
        Args:
            concept1: First concept name
            concept2: Second concept name
            steps: Number of interpolation steps
            
        Returns:
            Path to the visualization file
            
        Raises:
            ValueError: If concepts not found
        """
        logger.info(f"Generating concept interpolation visualization for concepts '{concept1}' and '{concept2}'")
        
        # Check if concepts exist
        if concept1 not in self.universe.concepts:
            raise ValueError(f"Concept '{concept1}' not found")
        
        if concept2 not in self.universe.concepts:
            raise ValueError(f"Concept '{concept2}' not found")
        
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
        
        # Get concept positions
        pos1 = self.universe.get_concept_position(concept1)
        pos2 = self.universe.get_concept_position(concept2)
        
        cart1 = pos1.to_cartesian()
        cart2 = pos2.to_cartesian()
        
        # Add concept1
        fig.add_trace(go.Scatter3d(
            x=[cart1[0]],
            y=[cart1[1]],
            z=[cart1[2]],
            text=[concept1],
            mode="markers+text",
            marker=dict(
                size=15,
                color="red",
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add concept2
        fig.add_trace(go.Scatter3d(
            x=[cart2[0]],
            y=[cart2[1]],
            z=[cart2[2]],
            text=[concept2],
            mode="markers+text",
            marker=dict(
                size=15,
                color="blue",
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add interpolation points
        x_vals = []
        y_vals = []
        z_vals = []
        labels = []
        
        for i in range(1, steps):
            # Calculate weight
            weight = i / steps
            
            # Interpolate
            interp_pos = await self.vectorizer.interpolate_concepts(concept1, concept2, weight)
            interp_cart = interp_pos.to_cartesian()
            
            x_vals.append(interp_cart[0])
            y_vals.append(interp_cart[1])
            z_vals.append(interp_cart[2])
            labels.append(f"{concept1}_{concept2}_{weight:.2f}")
        
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            text=labels,
            mode="markers+text",
            marker=dict(
                size=10,
                color=list(range(len(x_vals))),
                colorscale="Viridis",
                opacity=0.8
            ),
            textposition="top center",
            hoverinfo="text"
        ))
        
        # Add interpolation line
        x_line = [cart1[0]] + x_vals + [cart2[0]]
        y_line = [cart1[1]] + y_vals + [cart2[1]]
        z_line = [cart1[2]] + z_vals + [cart2[2]]
        
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode="lines",
            line=dict(
                color="purple",
                width=3
            ),
            hoverinfo="none"
        ))
        
        # Set layout
        fig.update_layout(
            title=f"Concept Interpolation: '{concept1}' to '{concept2}'",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )
        
        # Save visualization
        vis_path = self.vis_dir / f"{concept1}_{concept2}_concept_interpolation_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated concept interpolation visualization: {vis_path}")
        
        return vis_path
