"""
Shared pytest fixtures for LEXICON tests.
"""

import os
import sys
import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.storage.database import Database
from src.storage.cache import RedisCache
from src.storage.vector_store import FAISSStore

# Global fixtures

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def db():
    """Create a test database connection."""
    database = Database(settings.database_url)
    await database.connect()
    
    # Create test tables
    await database.execute("""
    CREATE TABLE IF NOT EXISTS concepts (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        atomic_pattern JSONB NOT NULL,
        not_space JSONB NOT NULL,
        confidence FLOAT NOT NULL,
        null_ratio FLOAT,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL
    )
    """)
    
    await database.execute("""
    CREATE TABLE IF NOT EXISTS concept_access (
        id SERIAL PRIMARY KEY,
        concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
        user_id TEXT,
        accessed_at TIMESTAMP WITH TIME ZONE NOT NULL,
        context TEXT
    )
    """)
    
    yield database
    
    # Clean up
    await database.execute("DROP TABLE IF EXISTS concept_access CASCADE")
    await database.execute("DROP TABLE IF EXISTS concepts CASCADE")
    await database.disconnect()

@pytest.fixture
async def redis_cache():
    """Create a test Redis cache."""
    cache = RedisCache(settings.redis_url)
    await cache.connect()
    
    # Clear test database
    await cache.client.flushdb()
    
    yield cache
    
    # Clean up
    await cache.client.flushdb()
    await cache.disconnect()

@pytest.fixture
def vector_store():
    """Create a test vector store."""
    # Create a temporary directory for the index
    os.makedirs(os.path.dirname(settings.vector_index_path), exist_ok=True)
    
    # Create the vector store
    store = FAISSStore(
        dimension=768,
        index_path=settings.vector_index_path
    )
    
    # Initialize the index
    store.initialize()
    
    yield store
    
    # Clean up
    if os.path.exists(settings.vector_index_path):
        os.remove(settings.vector_index_path)

@pytest.fixture
def app_state(db, redis_cache, vector_store):
    """Create a mock application state."""
    # Create mock vectorizer
    vectorizer = MagicMock()
    vectorizer.device = torch.device("cpu")
    
    # Mock vector generation
    def mock_forward(definition):
        from src.core.types import VectorizedObject
        return VectorizedObject(
            concept_id=definition.id,
            vector=np.random.randn(768).astype(np.float32),
            null_ratio=0.1,
            not_space_vector=np.random.randn(768).astype(np.float32),
            empathy_scores={"self_empathy": 0.8, "other_empathy": 0.2, "mutual_empathy": 0.6},
            cultural_variants={},
            metadata={"confidence": definition.confidence}
        )
    
    vectorizer.__call__ = mock_forward
    
    # Create app state
    state = {
        "db": db,
        "cache": redis_cache,
        "vector_store": vector_store,
        "vectorizer": vectorizer,
        "config": settings,
        "websocket_manager": MagicMock()
    }
    
    return state

@pytest.fixture
def sample_concept_data():
    """Sample concept data for tests."""
    return {
        "id": "test-concept-id",
        "name": "Tree",
        "atomic_pattern": {"pattern": ["1", "&&"]},
        "not_space": ["rock", "building", "animal"],
        "confidence": 0.95,
        "null_ratio": 0.05
    }
