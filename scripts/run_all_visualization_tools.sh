#!/bin/bash
echo "Running LEXICON Visualization Tools..."
python "$(dirname "$0")/run_all_visualization_tools.py"
echo ""
echo "Press Enter to exit..."
read
