#!/bin/bash
echo "Preparing LEXICON visualizations for GitHub Pages..."
python "$(dirname "$0")/prepare_github_pages.py"
echo ""
echo "Press Enter to exit..."
read
