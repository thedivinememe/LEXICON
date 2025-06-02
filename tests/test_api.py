"""
Tests for the REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.main import app
from src.services.definition import DefinitionService
from src.core.types import VectorizedObject

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def mock_app_state(app_state):
    """Mock the app_state dependency."""
    with patch('src.api.dependencies.get_app_state', return_value=app_state):
        yield app_state

@pytest.fixture
def mock_definition_service():
    """Mock the DefinitionService."""
    with patch('src.services.definition.DefinitionService', autospec=True) as mock_service:
        # Mock the define_concept method
        instance = mock_service.return_value
        instance.define_concept.return_value = VectorizedObject(
            concept_id="test-concept-id",
            vector=[0.1] * 768,
            null_ratio=0.05,
            not_space_vector=[0.2] * 768,
            empathy_scores={"self_empathy": 0.8, "other_empathy": 0.2, "mutual_empathy": 0.6},
            cultural_variants={},
            metadata={"atomic_pattern": "1 && !1", "confidence": 0.95}
        )
        
        # Mock the get_similar_concepts method
        instance.get_similar_concepts.return_value = [
            {
                "concept": "Plant",
                "similarity": 0.9,
                "null_ratio": 0.1,
                "shared_not_space": 1
            },
            {
                "concept": "Forest",
                "similarity": 0.8,
                "null_ratio": 0.15,
                "shared_not_space": 0
            }
        ]
        
        yield instance

@pytest.mark.asyncio
async def test_define_concept_endpoint(client, mock_app_state, mock_definition_service):
    """Test the define concept endpoint."""
    # Define request data
    request_data = {
        "concept": "Tree",
        "negations": ["rock", "building", "animal"],
        "cultural_context": "universal"
    }
    
    # Make the request
    response = client.post("/api/v1/concepts/define", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    
    # Check the response data
    assert data["concept_id"] == "test-concept-id"
    assert data["concept_name"] == "Tree"
    assert data["atomic_pattern"] == "1 && !1"
    assert data["null_ratio"] == 0.05
    assert "empathy_scores" in data
    assert "vector_preview" in data
    
    # Check that the service was called correctly
    mock_definition_service.define_concept.assert_called_once_with(
        concept="Tree",
        negations=["rock", "building", "animal"]
    )

@pytest.mark.asyncio
async def test_get_similar_concepts_endpoint(client, mock_app_state, mock_definition_service):
    """Test the get similar concepts endpoint."""
    # Make the request
    response = client.get("/api/v1/concepts/test-concept-id/similar?k=2")
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    
    # Check the response data
    assert data["concept_id"] == "test-concept-id"
    assert "similar_concepts" in data
    assert len(data["similar_concepts"]) == 2
    assert data["similar_concepts"][0]["concept"] == "Plant"
    assert data["similar_concepts"][0]["similarity"] == 0.9
    assert data["similar_concepts"][1]["concept"] == "Forest"
    
    # Check that the service was called correctly
    mock_definition_service.get_similar_concepts.assert_called_once_with(
        "test-concept-id", 2
    )

@pytest.mark.asyncio
async def test_define_concept_endpoint_error(client, mock_app_state, mock_definition_service):
    """Test error handling in the define concept endpoint."""
    # Mock an error in the service
    mock_definition_service.define_concept.side_effect = ValueError("Invalid concept")
    
    # Define request data
    request_data = {
        "concept": "",  # Empty concept name should cause an error
        "negations": []
    }
    
    # Make the request
    response = client.post("/api/v1/concepts/define", json=request_data)
    
    # Check the response
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Invalid concept"

@pytest.mark.asyncio
async def test_get_similar_concepts_endpoint_not_found(client, mock_app_state, mock_definition_service):
    """Test not found error in the get similar concepts endpoint."""
    # Mock a not found error
    mock_definition_service.get_similar_concepts.side_effect = ValueError("Concept not found")
    
    # Make the request
    response = client.get("/api/v1/concepts/nonexistent-id/similar")
    
    # Check the response
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Concept not found"

@pytest.mark.asyncio
async def test_health_check_endpoint(client):
    """Test the health check endpoint."""
    # Make the request
    response = client.get("/health")
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    
    # Check the response data
    assert data["status"] == "healthy"
    assert "version" in data
    assert "device" in data
    assert "features" in data
