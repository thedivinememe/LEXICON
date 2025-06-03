"""
Tests for the VectorizedObjectGenerator.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.neural.vectorizer import VectorizedObjectGenerator
from src.core.types import ConceptDefinition, ExistencePattern

@pytest.fixture
def mock_bert():
    """Mock BERT model and tokenizer"""
    with patch('transformers.AutoModel.from_pretrained') as mock_model, \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        
        # Mock model output
        mock_last_hidden = MagicMock()
        mock_last_hidden.mean.return_value = torch.ones((1, 768))
        
        mock_output = MagicMock()
        mock_output.last_hidden_state = mock_last_hidden
        
        mock_model.return_value.return_value = mock_output
        mock_model.return_value.forward.return_value = mock_output
        
        # Mock tokenizer output with a to() method
        class MockTokenizerOutput(dict):
            def to(self, device):
                return self
                
        mock_tokens = MockTokenizerOutput({
            'input_ids': torch.ones((1, 10), dtype=torch.long),
            'attention_mask': torch.ones((1, 10), dtype=torch.long)
        })
        mock_tokenizer.return_value.return_value = mock_tokens
        
        yield

@pytest.fixture
def vectorizer(mock_bert):
    """Create a VectorizedObjectGenerator instance with mocked components"""
    generator = VectorizedObjectGenerator(device="cpu")
    
    # Mock the empathy attention mechanism
    mock_attended = torch.ones((1, 1, 768))
    mock_weights = torch.ones((1, 1, 1))
    # Patch the forward method instead of replacing the module
    generator.empathy_attention.forward = MagicMock(return_value=(mock_attended, mock_weights))
    
    # Mock the null predictor's forward method
    generator.null_predictor.forward = MagicMock(return_value=torch.tensor([0.1]))
    
    return generator

def test_vectorizer_initialization():
    """Test that the vectorizer initializes correctly"""
    with patch('transformers.AutoModel.from_pretrained'), \
         patch('transformers.AutoTokenizer.from_pretrained'):
        
        vectorizer = VectorizedObjectGenerator(device="cpu")
        
        # Check that primitive embeddings are initialized
        assert len(vectorizer.primitive_embeddings) == 4
        assert "1" in vectorizer.primitive_embeddings
        assert "!1" in vectorizer.primitive_embeddings
        assert "&&" in vectorizer.primitive_embeddings
        assert "||" in vectorizer.primitive_embeddings
        
        # Check that primitive embeddings are orthogonal
        for p1 in vectorizer.primitive_embeddings.values():
            for p2 in vectorizer.primitive_embeddings.values():
                if p1 is not p2:
                    # Dot product should be close to 0 for orthogonal vectors
                    assert torch.dot(p1, p2).abs() < 1e-6

def test_encode_pattern(vectorizer):
    """Test encoding an existence pattern"""
    # Create a simple pattern
    pattern = ExistencePattern(
        pattern=["1", "&&"],
        confidence=0.9
    )
    
    # Encode the pattern
    encoded = vectorizer.encode_pattern(pattern)
    
    # Check the result
    assert encoded.shape == (768,)
    assert torch.is_tensor(encoded)

def test_compute_x_shaped_hole(vectorizer):
    """Test computing the X-shaped hole"""
    # Compute X-shaped hole
    hole = vectorizer.compute_x_shaped_hole(
        concept="Tree",
        not_space=["rock", "building"]
    )
    
    # Check the result
    assert hole.shape == (768,)
    assert torch.is_tensor(hole)
    assert torch.norm(hole).item() == pytest.approx(1.0, abs=1e-6)

def test_forward(vectorizer):
    """Test the forward pass"""
    # Create a concept definition
    definition = ConceptDefinition(
        id="test-id",
        name="Tree",
        atomic_pattern=ExistencePattern(pattern=["1", "&&"], confidence=0.9),
        not_space={"rock", "building"},
        confidence=0.9,
        created_at=None,
        updated_at=None
    )
    
    # Generate vectorized object
    result = vectorizer(definition)
    
    # Check the result
    assert result.concept_id == "test-id"
    assert isinstance(result.vector, np.ndarray)
    assert result.vector.shape == (768,)
    assert result.null_ratio == pytest.approx(0.1, abs=1e-5)
    assert isinstance(result.empathy_scores, dict)
    assert "self_empathy" in result.empathy_scores
    assert isinstance(result.not_space_vector, np.ndarray)
    assert result.not_space_vector.shape == (1, 768)
    assert isinstance(result.metadata, dict)
