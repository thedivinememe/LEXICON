#!/bin/bash
echo "Installing LEXICON Visualization Dependencies..."
python "$(dirname "$0")/install_visualization_dependencies.py"
echo ""
echo "Press Enter to exit..."
read
