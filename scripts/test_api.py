"""
Test the LEXICON API endpoints.
This script tests the API endpoints for the LEXICON system.
"""

import requests
import json
import sys
from pathlib import Path

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("=" * 80)
    print("LEXICON API Tester")
    print("=" * 80)
    print()
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("Health endpoint is working!")
            print(f"Response: {response.json()}")
        else:
            print(f"Health endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
    
    print()
    
    # Test visualization endpoint
    print("Testing visualization endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/vectors/visualize")
        if response.status_code == 200:
            print("Visualization endpoint is working!")
            data = response.json()
            print(f"Visualization method: {data.get('method')}")
            print(f"Visualization dimensions: {data.get('dimensions')}")
            print(f"Number of points: {len(data.get('points', []))}")
        else:
            print(f"Visualization endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error testing visualization endpoint: {e}")
    
    print()
    print("=" * 80)
    print("API testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_api()
