"""
Run the COREE interface.

This script starts the FastAPI server with the COREE interface.
"""

import os
import sys
import asyncio
import uvicorn
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/coree.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the COREE interface."""
    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Ensure the OpenAI API key is set
    if not os.environ.get('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable is not set")
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set the OPENAI_API_KEY environment variable in the .env file")
        sys.exit(1)
    
    # Start the FastAPI server
    logger.info("Starting COREE interface")
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
