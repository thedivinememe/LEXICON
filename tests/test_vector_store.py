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
    with patch('faiss.IndexFlatIP') as mock_flat_index, \
         patch('faiss.IndexIDMap') as mock_id_map:
        # Mock the index creation
        mock_flat_index.return_value = MagicMock()
        mock_id_map.return_value = MagicMock()
        
        # Call initialize
        store.initialize()
        
        # Verify the index was created
        mock_flat_index.assert_called_once_with(128)
        mock_id_map.assert_called_once()

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
    
    # Mock the index search method to avoid FAISS issues
    mock_distances = np.array([[1.0, 0.8, 0.6]])
    mock_indices = np.array([[0, 1, 2]])
    
    with patch.object(vector_store.index, 'search', return_value=(mock_distances, mock_indices)):
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
    
    # Mock the reconstruct method to return a known vector
    with patch.object(vector_store.index, 'reconstruct', return_value=vectors[5]):
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
    id_map_path = f"{index_path}.ids"
    
    # Create test vectors and IDs
    vectors = np.random.randn(10, 128).astype(np.float32)
    ids = [f"test-id-{i}" for i in range(10)]
    
    # Create a mock index
    mock_index = MagicMock()
    mock_index.ntotal = 10
    mock_distances = np.array([[1.0, 0.8, 0.6]])
    mock_indices = np.array([[0, 1, 2]])
    mock_index.search.return_value = (mock_distances, mock_indices)
    
    # Create the store with mocked index
    with patch('faiss.IndexFlatIP'), \
         patch('faiss.IndexIDMap', return_value=mock_index):
        store = FAISSStore(dimension=128, index_path=index_path)
    
    # Add vectors to the store
    store.add_vectors(vectors, ids)
    
    # Save the index with mocked write_index
    with patch('faiss.write_index'):
        store.save()
    
    # Verify the ID mapping file was created
    assert os.path.exists(id_map_path)
    
    # Create a new store
    new_store = FAISSStore(dimension=128, index_path=index_path)
    
    # Create a mock file content for the ID mapping
    mock_file_content = "\n".join([f"{i},{id_str}" for i, id_str in enumerate(ids)])
    
    # Mock open to return our mock file content
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value.readlines.return_value = [f"{i},{id_str}\n" for i, id_str in enumerate(ids)]
    
    # Mock the file operations for loading
    with patch('faiss.read_index', return_value=mock_index), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', create=True) as mock_open_func:
        # Set up the mock to return different file handles for read and write
        mock_file = MagicMock()
        mock_file.__enter__.return_value = MagicMock()
        mock_file.__enter__.return_value.read.return_value = mock_file_content
        mock_file.__enter__.return_value.__iter__.return_value = [f"{i},{id_str}\n" for i, id_str in enumerate(ids)]
        mock_open_func.return_value = mock_file
        
        # Load the index
        new_store.load()
    
    # Verify the ID mapping was loaded correctly
    assert len(new_store.id_map) == 10
    for i, id_str in enumerate(ids):
        assert new_store.id_map[i] == id_str
    
    # Test search with the loaded index
    query = vectors[0]
    distances, indices = new_store.search(query, k=3)
    
    # Check the results
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert indices[0][0] == 0
    assert distances[0][0] == pytest.approx(1.0, abs=1e-5)
    
    # Clean up
    if os.path.exists(id_map_path):
        os.remove(id_map_path)
