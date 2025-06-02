"""
Prepare LEXICON visualizations for GitHub Pages deployment.
This script creates a GitHub Pages-ready version of the visualizations.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import json

def main():
    """Prepare GitHub Pages deployment"""
    print("=" * 80)
    print("LEXICON GitHub Pages Deployment Preparation")
    print("=" * 80)
    print()
    
    # Get the project root directory
    project_dir = Path(__file__).parent.parent
    
    # Create docs directory (GitHub Pages uses this by default)
    docs_dir = project_dir / 'docs'
    docs_dir.mkdir(exist_ok=True)
    
    # Create visualizations if they don't exist
    visualizations_dir = project_dir / 'visualizations'
    if not visualizations_dir.exists() or not any(visualizations_dir.iterdir()):
        print("Visualizations not found. Generating them now...")
        try:
            subprocess.run(
                [sys.executable, str(project_dir / 'scripts' / 'visualize_test_data.py')],
                check=True
            )
            subprocess.run(
                [sys.executable, str(project_dir / 'scripts' / 'create_dashboard.py')],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error generating visualizations: {e}")
            print("Please run the visualization scripts manually first:")
            print("1. python scripts/visualize_test_data.py")
            print("2. python scripts/create_dashboard.py")
            return
    
    # Copy visualizations to docs directory
    print("\nCopying visualizations to docs directory...")
    for file in visualizations_dir.glob('*'):
        if file.is_file():
            shutil.copy2(file, docs_dir)
            print(f"Copied {file.name}")
    
    # Create GitHub Pages index.html if it doesn't exist
    index_path = docs_dir / 'index.html'
    if not index_path.exists():
        print("\nCreating GitHub Pages index.html...")
        dashboard_path = visualizations_dir / 'index.html'
        if dashboard_path.exists():
            shutil.copy2(dashboard_path, index_path)
            print(f"Copied dashboard to {index_path}")
        else:
            # Create a simple index.html
            html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LEXICON Concept Space</title>
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
        .content {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-top: 20px;
        }
        .viz-container {
            width: 100%;
            text-align: center;
            margin: 20px 0;
        }
        .viz-container img {
            max-width: 100%;
            height: auto;
            margin-bottom: 10px;
        }
        .viz-container iframe {
            width: 100%;
            height: 600px;
            border: none;
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
        <h1>LEXICON Concept Space</h1>
        <p>Visualization of concept vectors in the LEXICON system</p>
    </header>
    
    <div class="container">
        <div class="content">
            <h2>About LEXICON</h2>
            <p>
                LEXICON is a Python application implementing Null/Not-Null Logic theory through vectorized concept definitions. 
                The system defines concepts through negation (X-shaped hole principle), generates semantic vectors, 
                and evolves them using empathetic normalization.
            </p>
            <p>
                This page visualizes the test data concepts in 2D and 3D space, showing the relationships
                between different concepts and how they cluster together.
            </p>
        </div>
        
        <div class="content">
            <h2>2D Visualizations</h2>
            
            <div class="viz-container">
                <h3>t-SNE Visualization</h3>
                <img src="concepts_2d_tsne.png" alt="2D t-SNE Visualization">
                <p>
                    This visualization uses t-SNE to reduce the 768-dimensional concept vectors to 2 dimensions.
                    t-SNE is good at preserving local structure, showing which concepts are similar to each other.
                </p>
            </div>
            
            <div class="viz-container">
                <h3>PCA Visualization</h3>
                <img src="concepts_2d_pca.png" alt="2D PCA Visualization">
                <p>
                    This visualization uses PCA to reduce the 768-dimensional concept vectors to 2 dimensions.
                    PCA preserves global structure, showing the principal directions of variation in the data.
                </p>
            </div>
        </div>
        
        <div class="content">
            <h2>3D Visualizations</h2>
            
            <div class="viz-container">
                <h3>t-SNE Visualization</h3>
                <iframe src="concepts_3d_tsne.html" title="3D t-SNE Visualization"></iframe>
                <p>
                    This interactive 3D visualization uses t-SNE to reduce the 768-dimensional concept vectors to 3 dimensions.
                    You can rotate, zoom, and pan to explore the concept space. Hover over points to see concept names.
                </p>
                <a href="concepts_3d_tsne.html" class="button" target="_blank">Open in Full Screen</a>
            </div>
            
            <div class="viz-container">
                <h3>PCA Visualization</h3>
                <iframe src="concepts_3d_pca.html" title="3D PCA Visualization"></iframe>
                <p>
                    This interactive 3D visualization uses PCA to reduce the 768-dimensional concept vectors to 3 dimensions.
                    You can rotate, zoom, and pan to explore the concept space. Hover over points to see concept names.
                </p>
                <a href="concepts_3d_pca.html" class="button" target="_blank">Open in Full Screen</a>
            </div>
        </div>
    </div>
    
    <footer style="text-align: center; margin-top: 50px; padding: 20px; background-color: #333; color: white;">
        <p>LEXICON Concept Space &copy; 2025</p>
    </footer>
</body>
</html>"""
            with open(index_path, 'w') as f:
                f.write(html_content)
            print("Created new index.html")
    
    # Create a README.md for the docs directory
    readme_path = docs_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write("""# LEXICON Concept Space

This directory contains the GitHub Pages deployment of the LEXICON concept space visualizations.

## Visualizations

- **2D Visualizations**:
  - [t-SNE Visualization](concepts_2d_tsne.png)
  - [PCA Visualization](concepts_2d_pca.png)

- **3D Visualizations**:
  - [t-SNE Visualization](concepts_3d_tsne.html)
  - [PCA Visualization](concepts_3d_pca.html)

## Dashboard

The [dashboard](index.html) provides an overview of all visualizations.

## About LEXICON

LEXICON is a Python application implementing Null/Not-Null Logic theory through vectorized concept definitions. The system defines concepts through negation (X-shaped hole principle), generates semantic vectors, and evolves them using empathetic normalization.
""")
    print(f"Created {readme_path}")
    
    # Create a .nojekyll file to disable Jekyll processing
    nojekyll_path = docs_dir / '.nojekyll'
    nojekyll_path.touch()
    print("Created .nojekyll file")
    
    # Final instructions
    print("\n" + "=" * 80)
    print("GitHub Pages Preparation Complete!")
    print("=" * 80)
    print("\nTo deploy to GitHub Pages:")
    print("1. Commit and push the 'docs' directory to your GitHub repository")
    print("2. Go to your repository settings on GitHub")
    print("3. Scroll down to the 'GitHub Pages' section")
    print("4. Select 'main branch /docs folder' as the source")
    print("5. Click 'Save'")
    print("\nYour visualizations will be available at:")
    print("https://[your-username].github.io/[repository-name]/")

if __name__ == "__main__":
    main()
