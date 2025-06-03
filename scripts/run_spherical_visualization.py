"""
Run Spherical Universe Visualization.

This script runs the visualization for the spherical universe,
loading core definitions and generating interactive visualizations.
"""

import asyncio
import logging
import os
import sys
import argparse
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.spherical_universe import BlochSphereUniverse, SphericalCoordinate
from src.core.null_gradient import NullGradientManager
from src.core.relative_type_system import RelativeTypeSystem, RelativeTypeHierarchy
from src.neural.spherical_vectorizer import SphericalRelationshipVectorizer
from src.services.sphere_visualization import SphericalUniverseVisualizer
from src.data.core_definitions import CORE_DEFINITIONS
from src.examples.spherical_integration_example import initialize_universe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def run_visualization(args):
    """Run the visualization."""
    logger.info("Running spherical universe visualization")
    
    # Initialize universe
    universe, null_gradient, vectorizer, type_system, visualizer = await initialize_universe()
    
    # Generate visualizations based on arguments
    if args.all:
        # Generate all visualizations
        logger.info("Generating all visualizations")
        
        # Generate sphere visualization
        sphere_path = await visualizer.generate_sphere_visualization()
        logger.info(f"Generated sphere visualization: {sphere_path}")
        
        # Generate null gradient visualization
        null_path = await visualizer.generate_null_gradient_visualization()
        logger.info(f"Generated null gradient visualization: {null_path}")
        
        # Generate relationship visualizations for key concepts
        key_concepts = ["concept", "meaning", "abstraction", "logic"]
        
        for concept in key_concepts:
            rel_path = await visualizer.generate_relationship_visualization(concept)
            logger.info(f"Generated relationship visualization for '{concept}': {rel_path}")
            
            type_path = await visualizer.generate_type_hierarchy_visualization(concept)
            logger.info(f"Generated type hierarchy visualization for '{concept}': {type_path}")
            
            cluster_path = await visualizer.generate_concept_cluster_visualization(concept)
            logger.info(f"Generated concept cluster visualization for '{concept}': {cluster_path}")
            
            nearest_path = await visualizer.generate_nearest_concepts_visualization(concept)
            logger.info(f"Generated nearest concepts visualization for '{concept}': {nearest_path}")
        
        # Generate interpolation visualization
        interp_path = await visualizer.generate_concept_interpolation_visualization("abstraction", "concrete")
        logger.info(f"Generated interpolation visualization: {interp_path}")
        
        # Open main visualization in browser
        if args.open:
            webbrowser.open(f"file://{os.path.abspath(sphere_path)}")
    
    elif args.concept:
        # Generate visualizations for a specific concept
        concept = args.concept
        
        # Check if concept exists
        if concept not in universe.concepts:
            logger.error(f"Concept '{concept}' not found")
            return
        
        # Generate visualizations
        rel_path = await visualizer.generate_relationship_visualization(concept)
        logger.info(f"Generated relationship visualization for '{concept}': {rel_path}")
        
        type_path = await visualizer.generate_type_hierarchy_visualization(concept)
        logger.info(f"Generated type hierarchy visualization for '{concept}': {type_path}")
        
        cluster_path = await visualizer.generate_concept_cluster_visualization(concept)
        logger.info(f"Generated concept cluster visualization for '{concept}': {cluster_path}")
        
        nearest_path = await visualizer.generate_nearest_concepts_visualization(concept)
        logger.info(f"Generated nearest concepts visualization for '{concept}': {nearest_path}")
        
        # Open relationship visualization in browser
        if args.open:
            webbrowser.open(f"file://{os.path.abspath(rel_path)}")
    
    elif args.interpolate:
        # Generate interpolation visualization
        concepts = args.interpolate.split(",")
        
        if len(concepts) != 2:
            logger.error("Interpolation requires exactly two concepts")
            return
        
        concept1, concept2 = concepts
        
        # Check if concepts exist
        if concept1 not in universe.concepts:
            logger.error(f"Concept '{concept1}' not found")
            return
        
        if concept2 not in universe.concepts:
            logger.error(f"Concept '{concept2}' not found")
            return
        
        # Generate visualization
        interp_path = await visualizer.generate_concept_interpolation_visualization(concept1, concept2)
        logger.info(f"Generated interpolation visualization: {interp_path}")
        
        # Open visualization in browser
        if args.open:
            webbrowser.open(f"file://{os.path.abspath(interp_path)}")
    
    elif args.null_gradient:
        # Generate null gradient visualization
        null_path = await visualizer.generate_null_gradient_visualization()
        logger.info(f"Generated null gradient visualization: {null_path}")
        
        # Open visualization in browser
        if args.open:
            webbrowser.open(f"file://{os.path.abspath(null_path)}")
    
    else:
        # Generate sphere visualization
        sphere_path = await visualizer.generate_sphere_visualization()
        logger.info(f"Generated sphere visualization: {sphere_path}")
        
        # Open visualization in browser
        if args.open:
            webbrowser.open(f"file://{os.path.abspath(sphere_path)}")

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run spherical universe visualization")
    
    # Visualization options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Generate all visualizations")
    group.add_argument("--concept", type=str, help="Generate visualizations for a specific concept")
    group.add_argument("--interpolate", type=str, help="Generate interpolation visualization (format: concept1,concept2)")
    group.add_argument("--null-gradient", action="store_true", help="Generate null gradient visualization")
    
    # Other options
    parser.add_argument("--open", action="store_true", help="Open visualization in browser")
    
    args = parser.parse_args()
    
    # Run visualization
    asyncio.run(run_visualization(args))

if __name__ == "__main__":
    main()
