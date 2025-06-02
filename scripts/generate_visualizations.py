"""
Generate all visualizations and dashboard for LEXICON.
This script runs both the visualization and dashboard creation scripts.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run visualization and dashboard scripts"""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Get paths to the visualization and dashboard scripts
    visualize_script = script_dir / 'visualize_test_data.py'
    dashboard_script = script_dir / 'create_dashboard.py'
    
    # Check if the scripts exist
    if not visualize_script.exists():
        print(f"Error: {visualize_script} does not exist")
        return
    
    if not dashboard_script.exists():
        print(f"Error: {dashboard_script} does not exist")
        return
    
    # Create output directory
    output_dir = script_dir.parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("LEXICON Visualization Generator")
    print("=" * 80)
    print()
    
    # Run the visualization script
    print("Step 1: Generating visualizations...")
    try:
        subprocess.run([sys.executable, str(visualize_script)], check=True)
        print("Visualizations generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error generating visualizations: {e}")
        return
    
    print()
    
    # Run the dashboard script
    print("Step 2: Creating dashboard...")
    try:
        subprocess.run([sys.executable, str(dashboard_script)], check=True)
        print("Dashboard created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error creating dashboard: {e}")
        return
    
    print()
    print("=" * 80)
    print("Visualization generation complete!")
    print("=" * 80)
    print()
    print(f"Output files are in the {output_dir} directory")
    print("To view the dashboard, open the following file in a web browser:")
    print(f"{output_dir / 'index.html'}")
    print()
    print("You can also run the FastAPI application to access the visualizations through the API:")
    print("uvicorn src.main:app --reload --host 0.0.0.0 --port 8000")
    print("Then visit: http://localhost:8000/docs to access the API documentation")

if __name__ == "__main__":
    main()
