"""
Tests for the DefinitionService.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.services.definition import DefinitionService
from src.core.types import ConceptDefinition, ExistencePattern, VectorizedObject

@pytest.fixture
def definition_service(app_state):
    """Create a DefinitionService instance with mocked dependencies."""
    # Mock the X-shaped hole engine
    x_hole_engine = MagicMock()
    x_hole_engine.define_through_negation.return_value = MagicMock(confidence=0.95)
    
    # Mock the primitive reducer
    reducer = MagicMock()
    reducer.reduce_to_primitives.return_value = ExistencePattern(
        pattern=["1", "&&"],
        confidence=0.95
    )
    
    # Create service with mocked components
    service = DefinitionService(app_state)
    service.x_hole_engine = x_hole_engine
    service.reducer = reducer
    
    return service

@pytest.mark.asyncio
async def test_define_concept(definition_service, app_state, monkeypatch):
    """Test defining a new concept."""
    # Mock UUID generation for consistent testing
    mock_uuid = "test-uuid-12345"
    monkeypatch.setattr(uuid, "uuid4", lambda: mock_uuid)
    
    # Define a concept
    concept_name = "Tree"
    negations = ["rock", "building", "animal"]
    
    # Call the service
    result = await definition_service.define_concept(concept_name, negations)
    
    # Check that the X-shaped hole engine was called
    definition_service.x_hole_engine.define_through_negation.assert_called_once_with(
        concept=concept_name,
        user_negations=negations
    )
    
    # Check that the reducer was called
    definition_service.reducer.reduce_to_primitives.assert_called_once_with(
        concept=concept_name,
        not_space=set(negations)
    )
    
    # Check that the vectorizer was called
    assert app_state["vectorizer"].__call__.called
    
    # Check that the result is a VectorizedObject
    assert isinstance(result, VectorizedObject)
    assert result.concept_id == mock_uuid
    
    # Check that the concept was stored in the database
    db = app_state["db"]
    db.concepts.insert_one.assert_called_once()
    
    # Check that the vector was added to the vector store
    vector_store = app_state["vector_store"]
    vector_store.add_vectors.assert_called_once()
    
    # Check that the result was cached
    cache = app_state["cache"]
    cache.set.assert_called_once()

@pytest.mark.asyncio
async def test_get_similar_concepts(definition_service, app_state):
    """Test getting similar concepts."""
    # Mock database response
    concept = {
        "id": "test-id",
        "name": "Tree",
        "vector": [0.1] * 768,
        "not_space": ["rock", "building"]
    }
    app_state["db"].concepts.find_one.return_value = concept
    
    # Mock vector store search
    app_state["vector_store"].search.return_value = (
        [[0.1, 0.2, 0.3]],  # Distances
        [[1, 2, 3]]         # Indices
    )
    
    # Mock vector store ID lookup
    app_state["vector_store"].get_id.side_effect = ["similar-id-1", "similar-id-2", "similar-id-3"]
    
    # Mock database responses for similar concepts
    similar_concepts = [
        {
            "id": "similar-id-1",
            "name": "Plant",
            "not_space": ["concrete", "building"]
        },
        {
            "id": "similar-id-2",
            "name": "Forest",
            "not_space": ["desert", "city"]
        },
        {
            "id": "similar-id-3",
            "name": "Garden",
            "not_space": ["building", "road"]
        }
    ]
    app_state["db"].concepts.find_one.side_effect = [concept] + similar_concepts
    
    # Get similar concepts
    result = await definition_service.get_similar_concepts("test-id", k=3)
    
    # Check the result
    assert len(result) == 3
    assert result[0]["concept"] == "Plant"
    assert result[1]["concept"] == "Forest"
    assert result[2]["concept"] == "Garden"
    
    # Check that similarity scores are included
    assert "similarity" in result[0]
    assert result[0]["similarity"] == 0.9  # 1 - distance
    
    # Check that shared not-space is calculated
    assert "shared_not_space" in result[0]

@pytest.mark.asyncio
async def test_define_concept_cached(definition_service, app_state):
    """Test that cached concepts are returned without recomputation."""
    # Mock cache hit
    cached_result = VectorizedObject(
        concept_id="cached-id",
        vector=[0.1] * 768,
        null_ratio=0.05,
        not_space_vector=[0.2] * 768,
        empathy_scores={"self_empathy": 0.8},
        cultural_variants={},
        metadata={}
    )
    app_state["cache"].get.return_value = cached_result
    
    # Define a concept that should be in cache
    result = await definition_service.define_concept("Cached", ["not-cached"])
    
    # Check that the result is the cached object
    assert result is cached_result
    
    # Check that the X-shaped hole engine was not called
    definition_service.x_hole_engine.define_through_negation.assert_not_called()
    
    # Check that the reducer was not called
    definition_service.reducer.reduce_to_primitives.assert_not_called()
    
    # Check that the vectorizer was not called
    assert not app_state["vectorizer"].__call__.called
