"""
Run all visualization tools for LEXICON.
This script runs all the visualization tools in sequence.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"{'=' * 80}")
    
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run all visualization tools"""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    print(f"{'=' * 80}")
    print("LEXICON Visualization Tools Runner")
    print(f"{'=' * 80}")
    print(f"Project directory: {project_dir}")
    
    # Create visualizations directory if it doesn't exist
    visualizations_dir = project_dir / 'visualizations'
    visualizations_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate visualizations
    success = run_command(
        [sys.executable, str(script_dir / 'visualize_test_data.py')],
        "Generating visualizations"
    )
    if not success:
        print("Failed to generate visualizations. Aborting.")
        return
    
    # Step 2: Create dashboard
    success = run_command(
        [sys.executable, str(script_dir / 'create_dashboard.py')],
        "Creating dashboard"
    )
    if not success:
        print("Failed to create dashboard. Aborting.")
        return
    
    # Step 3: Test API (if server is running)
    print("\nDo you want to test the API? This requires the LEXICON API server to be running.")
    print("If the server is not running, you can start it with:")
    print("    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000")
    
    choice = input("Test API? (y/n): ").strip().lower()
    if choice == 'y':
        # Test basic API endpoints
        run_command(
            [sys.executable, str(script_dir / 'test_api.py')],
            "Testing API endpoints"
        )
        
        # Test visualization API
        run_command(
            [sys.executable, str(script_dir / 'test_visualization_api.py')],
            "Testing visualization API"
        )
    
    # Final message
    print(f"\n{'=' * 80}")
    print("Visualization tools execution complete!")
    print(f"{'=' * 80}")
    print(f"Output files are in the {visualizations_dir} directory")
    print("To view the dashboard, open the following file in a web browser:")
    print(f"{visualizations_dir / 'index.html'}")
    
    # Open the dashboard if requested
    print("\nDo you want to open the dashboard now?")
    choice = input("Open dashboard? (y/n): ").strip().lower()
    if choice == 'y':
        dashboard_path = visualizations_dir / 'index.html'
        if dashboard_path.exists():
            if sys.platform == 'win32':
                os.startfile(dashboard_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(dashboard_path)])
            else:  # Linux
                subprocess.run(['xdg-open', str(dashboard_path)])
            print(f"Opened dashboard: {dashboard_path}")
        else:
            print(f"Dashboard file not found: {dashboard_path}")

if __name__ == "__main__":
    main()
