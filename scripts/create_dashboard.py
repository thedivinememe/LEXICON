"""
Create a dashboard for LEXICON test data visualization.
This script generates an HTML dashboard that displays the visualizations.
"""

import os
import sys
from pathlib import Path
import json
import shutil

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

def create_dashboard():
    """Create a dashboard HTML file"""
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # Create dashboard HTML
    dashboard_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LEXICON Concept Space Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            header {
                background-color: #333;
                color: white;
                padding: 20px;
                text-align: center;
            }
            h1 {
                margin: 0;
            }
            .dashboard {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-gap: 20px;
                margin-top: 20px;
            }
            .card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                padding: 20px;
            }
            .card h2 {
                margin-top: 0;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .viz-container {
                width: 100%;
                text-align: center;
            }
            .viz-container img {
                max-width: 100%;
                height: auto;
            }
            .viz-container iframe {
                width: 100%;
                height: 600px;
                border: none;
            }
            .full-width {
                grid-column: 1 / span 2;
            }
            .button {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                border-radius: 4px;
                margin: 10px 5px;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>LEXICON Concept Space Dashboard</h1>
            <p>Visualization of concept vectors in the LEXICON system</p>
        </header>
        
        <div class="container">
            <div class="dashboard">
                <div class="card full-width">
                    <h2>About LEXICON</h2>
                    <p>
                        LEXICON is a Python application implementing Null/Not-Null Logic theory through vectorized concept definitions. 
                        The system defines concepts through negation (X-shaped hole principle), generates semantic vectors, 
                        and evolves them using empathetic normalization.
                    </p>
                    <p>
                        This dashboard visualizes the test data concepts in 2D and 3D space, showing the relationships
                        between different concepts and how they cluster together.
                    </p>
                </div>
                
                <div class="card">
                    <h2>2D Visualization (t-SNE)</h2>
                    <div class="viz-container">
                        <img src="concepts_2d_tsne.png" alt="2D t-SNE Visualization">
                    </div>
                    <p>
                        This visualization uses t-SNE to reduce the 768-dimensional concept vectors to 2 dimensions.
                        t-SNE is good at preserving local structure, showing which concepts are similar to each other.
                    </p>
                </div>
                
                <div class="card">
                    <h2>2D Visualization (PCA)</h2>
                    <div class="viz-container">
                        <img src="concepts_2d_pca.png" alt="2D PCA Visualization">
                    </div>
                    <p>
                        This visualization uses PCA to reduce the 768-dimensional concept vectors to 2 dimensions.
                        PCA preserves global structure, showing the principal directions of variation in the data.
                    </p>
                </div>
                
                <div class="card full-width">
                    <h2>3D Visualization (t-SNE)</h2>
                    <div class="viz-container">
                        <iframe src="concepts_3d_tsne.html" title="3D t-SNE Visualization"></iframe>
                    </div>
                    <p>
                        This interactive 3D visualization uses t-SNE to reduce the 768-dimensional concept vectors to 3 dimensions.
                        You can rotate, zoom, and pan to explore the concept space. Hover over points to see concept names.
                    </p>
                </div>
                
                <div class="card full-width">
                    <h2>3D Visualization (PCA)</h2>
                    <div class="viz-container">
                        <iframe src="concepts_3d_pca.html" title="3D PCA Visualization"></iframe>
                    </div>
                    <p>
                        This interactive 3D visualization uses PCA to reduce the 768-dimensional concept vectors to 3 dimensions.
                        You can rotate, zoom, and pan to explore the concept space. Hover over points to see concept names.
                    </p>
                </div>
                
                <div class="card full-width">
                    <h2>Generate New Visualizations</h2>
                    <p>
                        To generate new visualizations, run the following command:
                    </p>
                    <pre>python scripts/visualize_test_data.py</pre>
                    <p>
                        This will create new visualizations based on the current test data.
                    </p>
                    <div style="text-align: center;">
                        <a href="concepts_3d_tsne.html" class="button" target="_blank">Open 3D t-SNE in Full Screen</a>
                        <a href="concepts_3d_pca.html" class="button" target="_blank">Open 3D PCA in Full Screen</a>
                    </div>
                </div>
            </div>
        </div>
        
        <footer style="text-align: center; margin-top: 50px; padding: 20px; background-color: #333; color: white;">
            <p>LEXICON Concept Space Dashboard &copy; 2025</p>
        </footer>
    </body>
    </html>
    """
    
    # Write dashboard HTML to file
    dashboard_path = output_dir / 'index.html'
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    print(f"Dashboard created at {dashboard_path}")
    print("Open this file in a web browser to view the dashboard")

if __name__ == "__main__":
    create_dashboard()
