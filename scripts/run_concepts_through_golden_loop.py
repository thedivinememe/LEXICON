"""
Run Concepts Through Golden Loop.

This script processes all concepts through the Empathetic Golden Loop
and visualizes the before/after results.
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
from src.core.empathetic_golden_loop import GoldenLoopProcessor
from src.data.core_definitions import CORE_DEFINITIONS
from src.examples.spherical_integration_example import initialize_universe

import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class GoldenLoopVisualizer:
    """
    Visualizer for concepts processed through the Empathetic Golden Loop.
    
    This visualizer shows the before and after positions of concepts
    after being processed through the Golden Loop.
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
        
        return hover_text
    
    def __init__(self, 
                universe: BlochSphereUniverse,
                golden_loop_processor: GoldenLoopProcessor,
                vis_dir: Path = Path("visualizations")):
        """
        Initialize the visualizer.
        
        Args:
            universe: Spherical universe
            golden_loop_processor: Golden Loop processor
            vis_dir: Directory for visualizations
        """
        self.universe = universe
        self.golden_loop_processor = golden_loop_processor
        self.vis_dir = vis_dir
        
        # Create visualizations directory if it doesn't exist
        self.vis_dir.mkdir(exist_ok=True)
        
        # Store original positions
        self.original_positions = {
            concept: position
            for concept, position in universe.concepts.items()
        }
        
        # Store for processed positions
        self.processed_positions = {}
        self.processing_results = {}
    
    async def process_all_concepts(self) -> Dict[str, Dict]:
        """
        Process all concepts through the Golden Loop.
        
        Returns:
            Dictionary mapping concept names to processing results
        """
        logger.info("Processing all concepts through the Golden Loop")
        
        # Process each concept
        for concept, position in self.original_positions.items():
            logger.info(f"Processing concept: {concept}")
            
            # Create context
            context = {
                "concept": concept,
                "max_expected_magnitude": 1.0,
                "self_vector": np.ones(3) / np.sqrt(3),  # Default self vector
                "other_vector": np.ones(3) / np.sqrt(3)  # Default other vector
            }
            
            # Process through Golden Loop
            result = await self.golden_loop_processor.process_golden_loop_spherical(
                position, context
            )
            
            # Store results
            self.processed_positions[concept] = SphericalCoordinate.from_dict(result["final_position"])
            self.processing_results[concept] = result
            
            logger.info(f"Processed {concept}: {result['loop_count']} loops, violations: {result['violations_found']}")
        
        return self.processing_results
    
    def generate_visualization(self, concept_filter: Optional[List[str]] = None) -> Path:
        """
        Generate a visualization showing before and after positions.
        
        Args:
            concept_filter: Optional list of concepts to include (if None, include all)
            
        Returns:
            Path to the visualization file
        """
        logger.info("Generating Golden Loop visualization")
        
        # Create figure
        fig = go.Figure()
        
        # Add sphere surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            colorscale="Blues",
            showscale=False
        ))
        
        # Filter concepts if needed
        concepts = list(self.original_positions.keys())
        if concept_filter:
            concepts = [c for c in concepts if c in concept_filter]
        
        # Add original positions
        x_vals_orig = []
        y_vals_orig = []
        z_vals_orig = []
        labels_orig = []
        hover_texts_orig = []
        
        for concept in concepts:
            position = self.original_positions[concept]
            cart = position.to_cartesian()
            
            x_vals_orig.append(cart[0])
            y_vals_orig.append(cart[1])
            z_vals_orig.append(cart[2])
            labels_orig.append(concept)
            
            # Format definition for hover
            concept_def = CORE_DEFINITIONS.get(concept, {})
            hover_text = self._format_definition_for_hover(concept, concept_def)
            hover_texts_orig.append(hover_text)
        
        fig.add_trace(go.Scatter3d(
            x=x_vals_orig,
            y=y_vals_orig,
            z=z_vals_orig,
            text=labels_orig,
            mode="markers+text",
            marker=dict(
                size=8,
                color="blue",
                opacity=0.7
            ),
            textposition="top center",
            name="Original Positions",
            hovertext=hover_texts_orig,
            hoverinfo="text"
        ))
        
        # Add processed positions
        x_vals_proc = []
        y_vals_proc = []
        z_vals_proc = []
        labels_proc = []
        hover_texts_proc = []
        
        for concept in concepts:
            if concept in self.processed_positions:
                position = self.processed_positions[concept]
                cart = position.to_cartesian()
                
                x_vals_proc.append(cart[0])
                y_vals_proc.append(cart[1])
                z_vals_proc.append(cart[2])
                labels_proc.append(concept)
                
                # Format definition for hover with processing results
                concept_def = CORE_DEFINITIONS.get(concept, {})
                hover_text = self._format_definition_for_hover(concept, concept_def)
                
                # Add processing results
                result = self.processing_results.get(concept, {})
                hover_text += "<br><b>Golden Loop Results:</b><br>"
                hover_text += f"Loops: {result.get('loop_count', 0)}<br>"
                hover_text += f"Violations: {result.get('violations_found', False)}<br>"
                
                hover_texts_proc.append(hover_text)
        
        fig.add_trace(go.Scatter3d(
            x=x_vals_proc,
            y=y_vals_proc,
            z=z_vals_proc,
            text=labels_proc,
            mode="markers+text",
            marker=dict(
                size=8,
                color="red",
                opacity=0.7
            ),
            textposition="top center",
            name="Processed Positions",
            hovertext=hover_texts_proc,
            hoverinfo="text"
        ))
        
        # Add lines connecting original and processed positions
        for concept in concepts:
            if concept in self.processed_positions:
                orig_pos = self.original_positions[concept]
                proc_pos = self.processed_positions[concept]
                
                orig_cart = orig_pos.to_cartesian()
                proc_cart = proc_pos.to_cartesian()
                
                x_line = [orig_cart[0], proc_cart[0]]
                y_line = [orig_cart[1], proc_cart[1]]
                z_line = [orig_cart[2], proc_cart[2]]
                
                fig.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode="lines",
                    line=dict(
                        color="green",
                        width=2
                    ),
                    showlegend=False,
                    hoverinfo="none"
                ))
        
        # Set layout
        fig.update_layout(
            title="Concepts Before and After Golden Loop Processing",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save visualization
        vis_path = self.vis_dir / "golden_loop_visualization.html"
        fig.write_html(str(vis_path))
        
        logger.info(f"Generated Golden Loop visualization: {vis_path}")
        
        return vis_path
    
    def generate_detailed_report(self) -> Path:
        """
        Generate a detailed HTML report of the Golden Loop processing.
        
        Returns:
            Path to the report file
        """
        logger.info("Generating detailed Golden Loop report")
        
        # Create HTML content
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Golden Loop Processing Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }
                h1, h2, h3 {
                    color: #333;
                }
                .concept {
                    margin-bottom: 30px;
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                }
                .concept-header {
                    background-color: #f5f5f5;
                    padding: 10px;
                    margin: -15px -15px 15px -15px;
                    border-bottom: 1px solid #ddd;
                    border-radius: 5px 5px 0 0;
                }
                .state {
                    margin-bottom: 15px;
                    padding: 10px;
                    background-color: #f9f9f9;
                    border-left: 3px solid #0066cc;
                }
                .violations {
                    color: #cc0000;
                }
                .no-violations {
                    color: #006600;
                }
                .summary {
                    font-weight: bold;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 15px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                .position-change {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 15px;
                }
                .position-block {
                    width: 48%;
                }
            </style>
        </head>
        <body>
            <h1>Golden Loop Processing Report</h1>
            <p>
                This report shows the results of processing all concepts through the Empathetic Golden Loop.
                The Golden Loop consists of six states that transform concept vectors to align with empathetic principles.
            </p>
        """
        
        # Add summary table
        html += """
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Concept</th>
                    <th>Loops</th>
                    <th>Violations Found</th>
                    <th>Position Change</th>
                </tr>
        """
        
        for concept, result in sorted(self.processing_results.items()):
            # Calculate position change
            orig_pos = self.original_positions[concept]
            proc_pos = self.processed_positions[concept]
            
            orig_cart = orig_pos.to_cartesian()
            proc_cart = proc_pos.to_cartesian()
            
            position_change = np.linalg.norm(proc_cart - orig_cart)
            
            # Add row
            violations_class = "violations" if result["violations_found"] else "no-violations"
            violations_text = "Yes" if result["violations_found"] else "No"
            
            html += f"""
                <tr>
                    <td>{concept}</td>
                    <td>{result["loop_count"]}</td>
                    <td class="{violations_class}">{violations_text}</td>
                    <td>{position_change:.4f}</td>
                </tr>
            """
        
        html += """
            </table>
        """
        
        # Add detailed results for each concept
        html += "<h2>Detailed Results</h2>"
        
        for concept, result in sorted(self.processing_results.items()):
            html += f"""
            <div class="concept">
                <div class="concept-header">
                    <h3>{concept}</h3>
                </div>
                
                <div class="summary">
                    Loops: {result["loop_count"]} | 
                    Violations: <span class="{'violations' if result['violations_found'] else 'no-violations'}">
                        {result["violations_found"]}
                    </span>
                </div>
                
                <div class="position-change">
                    <div class="position-block">
                        <h4>Original Position</h4>
                        <table>
                            <tr><th>r</th><td>{result["original_position"]["r"]:.4f}</td></tr>
                            <tr><th>theta</th><td>{result["original_position"]["theta"]:.4f}</td></tr>
                            <tr><th>phi</th><td>{result["original_position"]["phi"]:.4f}</td></tr>
                        </table>
                    </div>
                    
                    <div class="position-block">
                        <h4>Final Position</h4>
                        <table>
                            <tr><th>r</th><td>{result["final_position"]["r"]:.4f}</td></tr>
                            <tr><th>theta</th><td>{result["final_position"]["theta"]:.4f}</td></tr>
                            <tr><th>phi</th><td>{result["final_position"]["phi"]:.4f}</td></tr>
                        </table>
                    </div>
                </div>
                
                <h4>State Processing</h4>
            """
            
            # Add state results
            for state_name, state_result in result["states"].items():
                html += f"""
                <div class="state">
                    <h5>{state_name}</h5>
                    <table>
                """
                
                # Add state details
                for key, value in state_result.items():
                    if key != "vector" and key != "state":  # Skip vector and state name
                        if isinstance(value, dict):
                            # Format nested dictionary
                            html += f"<tr><th>{key}</th><td>"
                            for sub_key, sub_value in value.items():
                                html += f"{sub_key}: {sub_value}<br>"
                            html += "</td></tr>"
                        elif isinstance(value, list):
                            # Format list
                            html += f"<tr><th>{key}</th><td>{', '.join(str(v) for v in value)}</td></tr>"
                        else:
                            # Format simple value
                            html += f"<tr><th>{key}</th><td>{value}</td></tr>"
                
                html += """
                    </table>
                </div>
                """
            
            html += """
            </div>
            """
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        # Save report
        report_path = self.vis_dir / "golden_loop_report.html"
        with open(report_path, "w") as f:
            f.write(html)
        
        logger.info(f"Generated detailed Golden Loop report: {report_path}")
        
        return report_path

async def run_concepts_through_golden_loop(open_browser: bool = True) -> Tuple[Path, Path]:
    """
    Run all concepts through the Golden Loop and visualize the results.
    
    Args:
        open_browser: Whether to open the visualization in a browser
        
    Returns:
        Tuple of (visualization path, report path)
    """
    logger.info("Running all concepts through the Golden Loop")
    
    # Initialize universe
    universe, null_gradient, vectorizer, type_system, universe_visualizer = await initialize_universe()
    
    # Initialize Golden Loop processor
    golden_loop_processor = GoldenLoopProcessor()
    
    # Create Golden Loop visualizer
    visualizer = GoldenLoopVisualizer(universe, golden_loop_processor)
    
    # Process all concepts
    await visualizer.process_all_concepts()
    
    # Generate visualization
    vis_path = visualizer.generate_visualization()
    
    # Generate detailed report
    report_path = visualizer.generate_detailed_report()
    
    # Open in browser if requested
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(vis_path)}")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
    
    return vis_path, report_path

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run concepts through the Golden Loop")
    
    # Options
    parser.add_argument("--open", action="store_true", help="Open visualization in browser")
    
    args = parser.parse_args()
    
    # Run visualization
    asyncio.run(run_concepts_through_golden_loop(args.open))

if __name__ == "__main__":
    main()
