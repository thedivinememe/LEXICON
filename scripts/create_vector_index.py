#!/usr/bin/env python
"""
Script to create and initialize the FAISS vector index for LEXICON.
"""

import argparse
import os
import sys
import numpy as np
import faiss
import asyncio

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.storage.database import Database
from src.storage.vector_store import FAISSStore

async def create_index(dimension=768, test_data=False):
    """
    Create and initialize the FAISS vector index.
    
    Args:
        dimension: Dimension of the vectors
        test_data: Whether to add test data to the index
    """
    print(f"Creating FAISS index with dimension {dimension}...")
    
    # Create index directory if it doesn't exist
    index_dir = os.path.dirname(settings.vector_index_path)
    os.makedirs(index_dir, exist_ok=True)
    
    # Create FAISS store
    vector_store = FAISSStore(
        dimension=dimension,
        index_path=settings.vector_index_path
    )
    
    # Add test data if requested
    if test_data:
        print("Adding test data to index...")
        
        # Connect to database
        db = Database(settings.database_url)
        await db.connect()
        
        try:
            # Get concepts from database
            concepts = await db.fetch("SELECT id, name FROM concepts")
            
            if concepts:
                print(f"Found {len(concepts)} concepts in database")
                
                # Generate random vectors for testing
                vectors = []
                ids = []
                
                for concept in concepts:
                    # Create a random vector (in a real system, these would be generated by the vectorizer)
                    vector = np.random.randn(dimension).astype(np.float32)
                    vector = vector / np.linalg.norm(vector)  # Normalize
                    
                    vectors.append(vector)
                    ids.append(concept["id"])
                
                # Add vectors to index
                vector_store.add_vectors(vectors, ids)
                print(f"Added {len(vectors)} vectors to index")
            else:
                print("No concepts found in database. Add sample data with init_db.py --sample")
        
        finally:
            # Close database connection
            await db.disconnect()
    
    # Save the index
    vector_store.save()
    print(f"FAISS index created and saved to {settings.vector_index_path}")
    
    # Test search if test data was added
    if test_data and 'vectors' in locals():
        print("\nTesting search...")
        # Use the first vector as a query
        query_vector = vectors[0]
        distances, indices = vector_store.search(query_vector, k=3)
        
        print("Search results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            concept_id = vector_store.get_id(idx)
            print(f"  {i+1}. ID: {concept_id}, Distance: {dist:.4f}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Create FAISS vector index for LEXICON")
    parser.add_argument("--dimension", type=int, default=768, help="Vector dimension")
    parser.add_argument("--test", action="store_true", help="Add test data to index")
    
    args = parser.parse_args()
    
    # Run index creation
    asyncio.run(create_index(dimension=args.dimension, test_data=args.test))

if __name__ == "__main__":
    main()
