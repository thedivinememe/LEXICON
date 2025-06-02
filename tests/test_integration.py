"""
Integration tests for LEXICON.
Tests the interaction between different components of the system.
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Union
import asyncio
from datetime import datetime
import torch

from src.core.primitives import EXISTS, NOT_EXISTS, parse_pattern
from src.core.x_shaped_hole import XShapedHoleEngine
from src.neural.vectorizer import VectorizedObjectGenerator
from src.neural.empathy import EmpathyNormalizer
from src.core.types import ConceptDefinition, ExistencePattern
from tests.test_data import (
    PHILOSOPHICAL_CONCEPTS,
    ETHICAL_CONCEPTS,
    COMPLEX_DEFINITIONS,
    BOUNDARY_TEST_CASES,
    EVOLUTION_TEST_DATA
)

# Fixtures for integration tests
@pytest.fixture
def x_shaped_hole_engine():
    """Create an X-shaped hole engine for testing"""
    return XShapedHoleEngine()

@pytest.fixture
def vectorizer():
    """Create a vectorizer for testing"""
    return VectorizedObjectGenerator(model_name="bert-base-uncased", device="cpu")

@pytest.fixture
def empathy_normalizer():
    """Create an empathy normalizer for testing"""
    return EmpathyNormalizer(vector_dim=768, device="cpu")

@pytest.fixture
def mock_app_state(x_shaped_hole_engine, vectorizer, empathy_normalizer):
    """Create a mock application state for testing"""
    from unittest.mock import MagicMock
    
    # Create mock database
    mock_db = MagicMock()
    mock_db.concepts = MagicMock()
    mock_db.concepts.find_one.return_value = None
    mock_db.concepts.insert_one = MagicMock()
    
    # Create mock vector store
    mock_vector_store = MagicMock()
    mock_vector_store.add_vectors = MagicMock()
    mock_vector_store.search = MagicMock(return_value=[])
    
    # Create mock cache
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_cache.set = MagicMock()
    
    # Create app state
    app_state = {
        "db": mock_db,
        "vector_store": mock_vector_store,
        "cache": mock_cache,
        "x_shaped_hole_engine": x_shaped_hole_engine,
        "vectorizer": vectorizer,
        "empathy_normalizer": empathy_normalizer,
        "config": {
            "vector_dim": 768,
            "enable_gpu": False,
            "enable_meme_evolution": True
        }
    }
    
    return app_state

# Integration tests
def test_x_shaped_hole_principle(x_shaped_hole_engine):
    """Test that enough negations reveal the concept"""
    # Define "cat" through many not-cat examples
    negations = ["dog", "bird", "fish", "rock", "plant", "building", "abstract_idea"]
    
    # Create definition
    definition = x_shaped_hole_engine.define_through_negation("cat", negations)
    
    # Verify the definition
    assert definition.concept == "cat"
    assert len(definition.not_space) == len(negations)
    assert definition.confidence > 0.5  # Should have reasonable confidence
    
    # Calculate boundary
    boundary = x_shaped_hole_engine.calculate_boundary("cat", negations)
    
    # Test boundary inclusion probability
    assert boundary["inclusion_probability"]("cat") > 0.8  # High probability for self
    assert boundary["inclusion_probability"]("kitten") > 0.7  # High for similar
    assert boundary["inclusion_probability"]("dog") < 0.3  # Low for negation
    assert boundary["inclusion_probability"]("rock") < 0.2  # Very low for distant negation

def test_empathy_normalization(empathy_normalizer):
    """Test that empathetic concepts score higher"""
    # Create test vectors
    cooperative_vector = np.random.randn(768)
    cooperative_vector = cooperative_vector / np.linalg.norm(cooperative_vector)
    
    competitive_vector = -cooperative_vector + 0.1 * np.random.randn(768)
    competitive_vector = competitive_vector / np.linalg.norm(competitive_vector)
    
    neutral_vector = np.random.randn(768)
    neutral_vector = neutral_vector / np.linalg.norm(neutral_vector)
    
    # Calculate empathy scores
    coop_coop_score = empathy_normalizer.calculate_empathy_score(
        cooperative_vector, cooperative_vector
    )
    
    coop_comp_score = empathy_normalizer.calculate_empathy_score(
        cooperative_vector, competitive_vector
    )
    
    coop_neutral_score = empathy_normalizer.calculate_empathy_score(
        cooperative_vector, neutral_vector
    )
    
    # Verify empathy relationships
    assert coop_coop_score > 0.8  # High empathy with self
    assert coop_comp_score < 0.3  # Low empathy with opposite
    assert coop_neutral_score > coop_comp_score  # Higher with neutral than opposite
    
    # Test normalization
    vectors = [cooperative_vector, competitive_vector, neutral_vector]
    normalized_vectors, empathy_scores = empathy_normalizer.normalize_vectors(vectors)
    
    # Verify normalization preserves unit length
    for vec in normalized_vectors:
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
    
    # Verify group empathy improved
    original_group_empathy = empathy_normalizer.calculate_group_empathy(vectors)
    normalized_group_empathy = empathy_normalizer.calculate_group_empathy(normalized_vectors)
    
    assert normalized_group_empathy >= original_group_empathy

def test_null_ratio_refinement(x_shaped_hole_engine, vectorizer):
    """Test that null ratio decreases with more negations"""
    # Start with few negations
    few_negations = ["dog"]
    many_negations = ["dog", "bird", "fish", "rock", "plant", "building", "abstract_idea"]
    
    # Create definitions
    few_def = x_shaped_hole_engine.define_through_negation("cat", few_negations)
    many_def = x_shaped_hole_engine.define_through_negation("cat", many_negations)
    
    # Create concept definitions
    few_concept = ConceptDefinition(
        id="test-few",
        name="cat",
        atomic_pattern=ExistencePattern(pattern=[EXISTS], confidence=1.0),
        not_space=few_def.not_space,
        confidence=few_def.confidence,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    many_concept = ConceptDefinition(
        id="test-many",
        name="cat",
        atomic_pattern=ExistencePattern(pattern=[EXISTS], confidence=1.0),
        not_space=many_def.not_space,
        confidence=many_def.confidence,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Vectorize concepts
    with torch.no_grad():
        few_vector = vectorizer(few_concept)
        many_vector = vectorizer(many_concept)
    
    # Verify null ratio decreases with more negations
    assert few_vector.null_ratio > many_vector.null_ratio

@pytest.mark.asyncio
async def test_concept_definition_service(mock_app_state):
    """Test the full concept definition service"""
    from src.services.definition import DefinitionService
    
    # Create service
    service = DefinitionService(mock_app_state)
    
    # Define a concept
    concept = "democracy"
    negations = ["dictatorship", "monarchy", "anarchy", "totalitarianism"]
    
    # Define the concept
    result = await service.define_concept(concept, negations)
    
    # Verify result
    assert result.concept_id is not None
    assert result.null_ratio >= 0.0 and result.null_ratio <= 1.0
    assert len(result.empathy_scores) > 0
    
    # Mock the vector store to return results
    mock_app_state["vector_store"].search.return_value = [
        {"id": "test-1", "score": 0.9},
        {"id": "test-2", "score": 0.8}
    ]
    
    # Get similar concepts
    similar = await service.get_similar_concepts(result.concept_id, k=2)
    
    # Verify similar concepts
    assert len(similar) == 2
    assert similar[0]["id"] == "test-1"
    assert similar[0]["score"] > similar[1]["score"]

@pytest.mark.asyncio
async def test_evolution_service(mock_app_state):
    """Test the memetic evolution service"""
    from src.services.evolution import evolve_concepts
    
    # Create initial concepts in mock DB
    mock_concepts = []
    for concept_name, negations in EVOLUTION_TEST_DATA["initial_concepts"]:
        mock_concept = {
            "id": f"test-{concept_name}",
            "name": concept_name,
            "vector": np.random.randn(768).tolist(),
            "null_ratio": 0.5,
            "empathy_scores": {"self_empathy": 0.8, "other_empathy": 0.5},
            "negations": negations,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "atomic_pattern": "1"
        }
        mock_concepts.append(mock_concept)
    
    # Mock DB to return these concepts
    async def mock_find():
        return mock_concepts
    
    mock_app_state["db"].concepts.find = mock_find
    
    # Run evolution
    await evolve_concepts(mock_app_state)
    
    # Verify that insert_one was called (new concepts were created)
    assert mock_app_state["db"].concepts.insert_one.called
    
    # Verify that vector store was updated
    assert mock_app_state["vector_store"].add_vectors.called

def test_end_to_end_pipeline(x_shaped_hole_engine, vectorizer, empathy_normalizer):
    """Test the entire pipeline from definition to vectorization to normalization"""
    # Define a concept through negation
    concept = "cooperation"
    negations = ["competition", "conflict", "selfishness"]
    
    # 1. Create X-shaped hole definition
    definition = x_shaped_hole_engine.define_through_negation(concept, negations)
    
    # 2. Create concept definition
    concept_def = ConceptDefinition(
        id="test-cooperation",
        name=concept,
        atomic_pattern=ExistencePattern(pattern=[EXISTS], confidence=1.0),
        not_space=definition.not_space,
        confidence=definition.confidence,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # 3. Vectorize the concept
    with torch.no_grad():
        vectorized = vectorizer(concept_def)
    
    # 4. Create an opposing concept
    opposite_concept = "competition"
    opposite_negations = ["cooperation", "sharing", "altruism"]
    
    opposite_definition = x_shaped_hole_engine.define_through_negation(
        opposite_concept, opposite_negations
    )
    
    opposite_concept_def = ConceptDefinition(
        id="test-competition",
        name=opposite_concept,
        atomic_pattern=ExistencePattern(pattern=[EXISTS], confidence=1.0),
        not_space=opposite_definition.not_space,
        confidence=opposite_definition.confidence,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    with torch.no_grad():
        opposite_vectorized = vectorizer(opposite_concept_def)
    
    # 5. Calculate empathy between concepts
    empathy_score = empathy_normalizer.calculate_empathy_score(
        vectorized.vector, opposite_vectorized.vector
    )
    
    # 6. Normalize vectors
    normalized_vectors, _ = empathy_normalizer.normalize_vectors(
        [vectorized.vector, opposite_vectorized.vector]
    )
    
    # Verify results
    assert vectorized.null_ratio < 0.5  # Well-defined concept
    assert opposite_vectorized.null_ratio < 0.5  # Well-defined concept
    assert empathy_score < 0.5  # Low empathy between opposites
    
    # Verify normalized vectors are unit length
    for vec in normalized_vectors:
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

# Import torch here to avoid issues with pytest collection
import torch

if __name__ == "__main__":
    # Run tests manually
    engine = XShapedHoleEngine()
    test_x_shaped_hole_principle(engine)
    
    normalizer = EmpathyNormalizer(vector_dim=768, device="cpu")
    test_empathy_normalization(normalizer)
    
    print("All tests passed!")
