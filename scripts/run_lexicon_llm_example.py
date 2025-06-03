#!/usr/bin/env python
"""
Run LEXICON LLM Example.

This script runs the LEXICON LLM integration example.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import and run the example
from src.examples.lexicon_llm_example import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
