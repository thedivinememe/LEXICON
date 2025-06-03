#!/usr/bin/env python
"""
Run LEXICON LLM API Server.

This script runs the FastAPI server for the LEXICON LLM API.
"""

import argparse
import sys
import os
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run LEXICON LLM API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Set log level
    log_level = "info"
    if args.debug:
        log_level = "debug"
        
    # Run the server
    uvicorn.run(
        "src.llm.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
