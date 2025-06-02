"""
Install dependencies for LEXICON visualization tools.
This script installs the required packages for the visualization tools.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Install dependencies"""
    print("=" * 80)
    print("LEXICON Visualization Dependencies Installer")
    print("=" * 80)
    print()
    
    # Required packages
    packages = [
        "matplotlib",
        "plotly",
        "scikit-learn",
        "numpy",
        "requests"
    ]
    
    print(f"Installing required packages: {', '.join(packages)}")
    print()
    
    # Install packages
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + packages,
            check=True
        )
        print("\nPackages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError installing packages: {e}")
        print("Please try installing the packages manually:")
        print(f"pip install {' '.join(packages)}")
        return
    
    print("\nYou can now run the visualization tools:")
    print("- scripts/run_all_visualization_tools.py")
    print("- scripts/visualize_test_data.py")
    print("- scripts/create_dashboard.py")
    
    print("\nOr use the batch/shell scripts:")
    if sys.platform == 'win32':
        print("- scripts\\run_all_visualization_tools.bat")
    else:
        print("- scripts/run_all_visualization_tools.sh")

if __name__ == "__main__":
    main()
