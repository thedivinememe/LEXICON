#!/bin/bash
echo "Running LEXICON Visualization Generator..."
python "$(dirname "$0")/generate_visualizations.py"
echo ""
echo "Press Enter to exit..."
read
