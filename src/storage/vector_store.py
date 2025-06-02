"""
FAISS vector store interface for LEXICON.
"""

import os
import numpy as np
import faiss
from typing import Dict, List, Optional, Tuple, Union

class FAISSStore:
    """FAISS vector store for efficient similarity search"""
    
    def __init__(self, dimension: int = 768, index_path: Optional[str] = None):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Vector dimension (default: 768 for BERT)
            index_path: Path to load/save the index
        """
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.id_map = {}  # Maps FAISS internal IDs to concept IDs
        self.next_id = 0
        
        # Load index if it exists
        if index_path and os.path.exists(index_path):
            self.load()
        else:
            # Create a new index
            # Use IndexIDMap to support explicit IDs
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)
    
    def add_vectors(self, vectors: List[np.ndarray], ids: List[str]) -> None:
        """
        Add vectors to the index with their corresponding IDs.
        
        Args:
            vectors: List of vectors to add
            ids: List of string IDs corresponding to the vectors
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors and IDs must match")
        
        if not vectors:
            return
        
        # Convert to numpy array if needed
        if isinstance(vectors[0], list):
            vectors = [np.array(v, dtype=np.float32) for v in vectors]
        
        # Ensure vectors are 2D
        vectors_array = np.vstack([v.reshape(1, -1) if v.ndim == 1 else v for v in vectors])
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors_array)
        
        # Generate internal IDs
        internal_ids = np.array([self.next_id + i for i in range(len(vectors))], dtype=np.int64)
        
        # Update ID mapping
        for i, id_str in enumerate(ids):
            self.id_map[internal_ids[i]] = id_str
        
        # Add to index
        if hasattr(self.index, 'add_with_ids'):
            self.index.add_with_ids(vectors_array, internal_ids)
        else:
            # For indices that don't support explicit IDs
            self.index.add(vectors_array)
        
        # Update next ID
        self.next_id += len(vectors)
    
    def search(self, 
              query_vector: Union[List[float], np.ndarray], 
              k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Vector to search for
            k: Number of results to return
        
        Returns:
            Tuple of (distances, indices)
        """
        if self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        # Convert to numpy array if needed
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Perform search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        return distances, indices
    
    def get_id(self, index: int) -> Optional[str]:
        """Get the string ID for a FAISS internal index"""
        return self.id_map.get(int(index))
    
    def get_vector(self, id_str: str) -> Optional[np.ndarray]:
        """Get a vector by its string ID"""
        # Find the internal ID
        internal_id = None
        for idx, id_s in self.id_map.items():
            if id_s == id_str:
                internal_id = idx
                break
        
        if internal_id is None:
            return None
        
        # Reconstruct the vector if possible
        if hasattr(self.index, 'reconstruct'):
            return self.index.reconstruct(internal_id)
        
        # If reconstruction is not supported, return None
        return None
    
    def save(self) -> None:
        """Save the index to disk"""
        if not self.index_path:
            raise ValueError("No index path specified")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save the index
        faiss.write_index(self.index, self.index_path)
        
        # Save the ID mapping
        id_map_path = f"{self.index_path}.ids"
        with open(id_map_path, 'w') as f:
            for internal_id, id_str in self.id_map.items():
                f.write(f"{internal_id},{id_str}\n")
    
    def load(self) -> None:
        """Load the index from disk"""
        if not self.index_path or not os.path.exists(self.index_path):
            raise ValueError("Index file not found")
        
        # Load the index
        self.index = faiss.read_index(self.index_path)
        
        # Load the ID mapping
        id_map_path = f"{self.index_path}.ids"
        if os.path.exists(id_map_path):
            self.id_map = {}
            with open(id_map_path, 'r') as f:
                for line in f:
                    internal_id, id_str = line.strip().split(',', 1)
                    self.id_map[int(internal_id)] = id_str
            
            # Update next_id
            if self.id_map:
                self.next_id = max(self.id_map.keys()) + 1
    
    def clear(self) -> None:
        """Clear the index"""
        base_index = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        self.id_map = {}
        self.next_id = 0
    
    def __len__(self) -> int:
        """Get the number of vectors in the index"""
        return self.index.ntotal
