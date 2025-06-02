"""
Tests for the FAISS vector store.
"""

import pytest
import os
import numpy as np
import faiss
from unittest.mock import patch, MagicMock

from src.storage.vector_store import FAISSStore

@pytest.fixture
def test_vectors():
    """Create test vectors for FAISS."""
    # Create 10 random vectors
    vectors = np.random.randn(10, 128).astype(np.float32)
    
    # Normalize vectors
    for i in range(vectors.shape[0]):
        vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
    
    # Create IDs
    ids = [f"test-id-{i}" for i in range(10)]
    
    return vectors, ids

@pytest.fixture
def vector_store(tmp_path):
    """Create a test vector store."""
    # Create a temporary file path for the index
    index_path = str(tmp_path / "test_index.faiss")
    
    # Create the vector store
    store = FAISSStore(
        dimension=128,
        index_path=index_path
    )
    
    # Initialize the index
    store.initialize()
    
    yield store
    
    # Clean up
    if os.path.exists(index_path):
        os.remove(index_path)

def test_initialization():
    """Test that the vector store initializes correctly."""
    # Create a vector store
    store = FAISSStore(dimension=128, index_path="test_index.faiss")
    
    # Check that the store is initialized correctly
    assert store.dimension == 128
    assert store.index_path == "test_index.faiss"
    assert store.id_map == {}
    assert store.next_id == 0
    
    # Initialize the index
    with patch('faiss.IndexFlatIP') as mock_index:
        store.initialize()
        mock_index.assert_called_once()

def test_add_vectors(vector_store, test_vectors):
    """Test adding vectors to the store."""
    vectors, ids = test_vectors
    
    # Add vectors to the store
    vector_store.add_vectors(vectors, ids)
    
    # Check that the vectors were added
    assert len(vector_store.id_map) == 10
    for i, id in enumerate(ids):
        assert id in vector_store.id_map.values()
    
    # Check that the next ID is updated
    assert vector_store.next_id == 10

def test_search(vector_store, test_vectors):
    """Test searching for similar vectors."""
    vectors, ids = test_vectors
    
    # Add vectors to the store
    vector_store.add_vectors(vectors, ids)
    
    # Search for a vector
    query = vectors[0]
    distances, indices = vector_store.search(query, k=3)
    
    # Check the results
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    
    # The most similar vector should be the query itself
    assert indices[0][0] == 0
    assert distances[0][0] == pytest.approx(1.0, abs=1e-5)

def test_get_vector(vector_store, test_vectors):
    """Test getting a vector by ID."""
    vectors, ids = test_vectors
    
    # Add vectors to the store
    vector_store.add_vectors(vectors, ids)
    
    # Get a vector by ID
    vector = vector_store.get_vector(ids[5])
    
    # Check the result
    assert vector is not None
    assert vector.shape == (128,)
    assert np.array_equal(vector, vectors[5])
    
    # Test getting a non-existent vector
    vector = vector_store.get_vector("non-existent-id")
    assert vector is None

def test_get_id(vector_store, test_vectors):
    """Test getting an ID by index."""
    vectors, ids = test_vectors
    
    # Add vectors to the store
    vector_store.add_vectors(vectors, ids)
    
    # Get an ID by index
    id = vector_store.get_id(5)
    
    # Check the result
    assert id == ids[5]
    
    # Test getting a non-existent ID
    id = vector_store.get_id(100)
    assert id is None

def test_save_and_load(tmp_path):
    """Test saving and loading the index."""
    # Create a temporary file path for the index
    index_path = str(tmp_path / "test_index.faiss")
    
    # Create the vector store
    store = FAISSStore(
        dimension=128,
        index_path=index_path
    )
    
    # Initialize the index
    store.initialize()
    
    # Create test vectors
    vectors = np.random.randn(10, 128).astype(np.float32)
    ids = [f"test-id-{i}" for i in range(10)]
    
    # Add vectors to the store
    store.add_vectors(vectors, ids)
    
    # Save the index
    store.save()
    
    # Check that the index file exists
    assert os.path.exists(index_path)
    
    # Create a new store and load the index
    new_store = FAISSStore(
        dimension=128,
        index_path=index_path
    )
    
    # Load the index
    new_store.load()
    
    # Check that the vectors were loaded
    assert len(new_store.id_map) == 10
    for i, id in enumerate(ids):
        assert id in new_store.id_map.values()
    
    # Search for a vector
    query = vectors[0]
    distances, indices = new_store.search(query, k=3)
    
    # Check the results
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    
    # The most similar vector should be the query itself
    assert indices[0][0] == 0
    assert distances[0][0] == pytest.approx(1.0, abs=1e-5)
    
    # Clean up
    os.remove(index_path)
