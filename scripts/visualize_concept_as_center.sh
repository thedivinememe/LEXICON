#!/bin/bash
# Run the concept-centric visualization script

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Run the script
if [ -z "$1" ]; then
    echo "Running with default concept (existence)"
    python3 "$SCRIPT_DIR/visualize_concept_as_center.py" --open
elif [ "$1" = "--all" ]; then
    echo "Visualizing all concepts as centers"
    python3 "$SCRIPT_DIR/visualize_concept_as_center.py" --all --open
else
    echo "Visualizing concept '$1' as center"
    python3 "$SCRIPT_DIR/visualize_concept_as_center.py" --concept "$1" --open
fi

if [ $? -ne 0 ]; then
    echo "Error running visualization script"
    exit 1
fi

echo "Visualization complete"
