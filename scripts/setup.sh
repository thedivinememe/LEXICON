#!/bin/bash
# Setup script for LEXICON on Unix-based systems

set -e  # Exit on error

echo "Setting up LEXICON..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python not found. Please install Python 3.11 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -U pip setuptools wheel
pip install -e .

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/faiss_index
mkdir -p models
mkdir -p logs

# Download BERT model
echo "Downloading BERT model..."
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('bert-base-uncased'); AutoTokenizer.from_pretrained('bert-base-uncased')"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# LEXICON Environment Variables
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lexicon
REDIS_URL=redis://localhost:6379
SECRET_KEY=development_secret_key
DEBUG=true
VECTOR_INDEX_PATH=./data/faiss_index/index.faiss
EOF
fi

# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.py

echo
echo "LEXICON setup complete!"
echo
echo "To start the application:"
echo "  1. Ensure PostgreSQL and Redis are running"
echo "  2. Initialize the database: python scripts/init_db.py --sample"
echo "  3. Create the vector index: python scripts/create_vector_index.py --test"
echo "  4. Start the server: uvicorn src.main:app --reload"
echo
echo "To run tests:"
echo "  pytest"
echo
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo
