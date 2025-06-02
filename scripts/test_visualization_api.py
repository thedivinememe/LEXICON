"""
Test the visualization API endpoint with test data.
This script generates test data vectors and sends them to the API endpoint.
"""

import requests
import json
import sys
import numpy as np
from pathlib import Path
import time

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from tests.test_data import (
    PHILOSOPHICAL_CONCEPTS,
    ETHICAL_CONCEPTS,
    HIERARCHICAL_CONCEPTS,
    get_all_test_concepts
)

def generate_mock_vectors(concepts, dim=768):
    """Generate mock vectors for concepts"""
    vectors = {}
    for concept, negations in concepts:
        # Create a random unit vector
        vector = np.random.randn(dim)
        vector = vector / np.linalg.norm(vector)
        vectors[concept] = vector.tolist()
    return vectors

def test_visualization_api():
    """Test the visualization API endpoint with test data"""
    base_url = "http://localhost:8000"
    api_url = f"{base_url}/api/v1"
    
    print("=" * 80)
    print("LEXICON Visualization API Tester")
    print("=" * 80)
    print()
    
    # Get all test concepts
    all_concepts = get_all_test_concepts()
    print(f"Found {len(all_concepts)} test concepts")
    
    # Generate mock vectors
    print("Generating mock vectors...")
    vectors = generate_mock_vectors(all_concepts)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'visualizations' / 'api_test'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save vectors to file
    vectors_file = output_dir / 'test_vectors.json'
    with open(vectors_file, 'w') as f:
        json.dump(vectors, f, indent=2)
    print(f"Saved test vectors to {vectors_file}")
    
    # Define concepts to test
    concept_names = [concept for concept, _ in all_concepts]
    
    # Test visualization endpoint with default parameters
    print("\nTesting visualization endpoint with default parameters...")
    try:
        response = requests.get(
            f"{api_url}/vectors/visualize",
            params={"concept_ids": concept_names[:10]}  # Use first 10 concepts
        )
        
        if response.status_code == 200:
            print("Visualization endpoint is working!")
            data = response.json()
            print(f"Visualization method: {data.get('method')}")
            print(f"Visualization dimensions: {data.get('dimensions')}")
            print(f"Number of points: {len(data.get('points', []))}")
            
            # Save visualization data to file
            viz_file = output_dir / 'visualization_default.json'
            with open(viz_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved visualization data to {viz_file}")
        else:
            print(f"Visualization endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error testing visualization endpoint: {e}")
    
    # Test visualization endpoint with different parameters
    print("\nTesting visualization endpoint with PCA and 2D...")
    try:
        response = requests.get(
            f"{api_url}/vectors/visualize",
            params={
                "concept_ids": concept_names[:10],  # Use first 10 concepts
                "method": "pca",
                "dimensions": 2
            }
        )
        
        if response.status_code == 200:
            print("Visualization endpoint is working!")
            data = response.json()
            print(f"Visualization method: {data.get('method')}")
            print(f"Visualization dimensions: {data.get('dimensions')}")
            print(f"Number of points: {len(data.get('points', []))}")
            
            # Save visualization data to file
            viz_file = output_dir / 'visualization_pca_2d.json'
            with open(viz_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved visualization data to {viz_file}")
        else:
            print(f"Visualization endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error testing visualization endpoint: {e}")
    
    print()
    print("=" * 80)
    print("API testing complete!")
    print("=" * 80)
    print(f"Output files are in the {output_dir} directory")

if __name__ == "__main__":
    test_visualization_api()
