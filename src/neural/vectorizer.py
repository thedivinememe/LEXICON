import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Union
import asyncio
from dataclasses import dataclass, field

from src.core.types import VectorizedObject, ConceptDefinition, ExistencePattern
from src.core.existence_primitives import PrimitiveDefinition, ExistenceRelationship, LogicalConnector

class VectorizedObjectGenerator(nn.Module):
    """Neural network for generating vectorized objects from concepts"""
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Pre-trained language model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Primitive embeddings (orthogonal)
        self.primitive_embeddings = self._initialize_primitives()
        
        # X-shaped hole network
        self.hole_projector = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.Tanh()
        )
        
        # Null-ratio predictor
        self.null_predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Empathy calculator
        self.empathy_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.1
        )
        
        self.to(self.device)
    
    def _initialize_primitives(self) -> Dict[str, torch.Tensor]:
        """Create orthogonal embeddings for existence primitives"""
        # Use Hadamard matrix for orthogonality
        from scipy.linalg import hadamard
        # Hadamard matrix size must be a power of 2
        # Use 1024 (next power of 2 above 768) and truncate
        H = hadamard(1024)[:4, :768]  # Take first 4 rows, truncate to 768 columns
        
        primitives = {
            "1": torch.tensor(H[0], dtype=torch.float32),
            "!1": torch.tensor(H[1], dtype=torch.float32),
            "&&": torch.tensor(H[2], dtype=torch.float32),
            "||": torch.tensor(H[3], dtype=torch.float32)
        }
        
        # Normalize
        for k, v in primitives.items():
            primitives[k] = v / v.norm()
            
        return primitives
    
    def encode_pattern(self, pattern: ExistencePattern) -> torch.Tensor:
        """Encode existence pattern into vector"""
        # Handle Exists objects from primitives module
        if hasattr(pattern, 'symbol'):
            # This is an ExistencePrimitive object
            return self.primitive_embeddings[pattern.symbol].to(self.device)
        
        if len(pattern.pattern) == 0:
            # Empty pattern, return zero vector
            return torch.zeros(768).to(self.device)
            
        if isinstance(pattern.pattern[0], str):
            # Simple primitive
            return self.primitive_embeddings[pattern.pattern[0]].to(self.device)
        
        # Handle ExistencePrimitive objects
        if hasattr(pattern.pattern[0], 'symbol'):
            # This is an ExistencePrimitive object
            return self.primitive_embeddings[pattern.pattern[0].symbol].to(self.device)
        
        # Complex pattern - recursive encoding
        vectors = []
        for element in pattern.pattern:
            if isinstance(element, str):
                vectors.append(self.primitive_embeddings[element])
            elif hasattr(element, 'symbol'):
                # This is an ExistencePrimitive object
                vectors.append(self.primitive_embeddings[element.symbol])
            else:
                vectors.append(self.encode_pattern(element))
        
        # Combine using attention
        stacked = torch.stack(vectors).unsqueeze(0).to(self.device)
        attended, _ = self.empathy_attention(stacked, stacked, stacked)
        return attended.squeeze(0).mean(dim=0)
    
    def compute_x_shaped_hole(self, 
                             concept: str, 
                             not_space: List[str]) -> torch.Tensor:
        """Implement X-shaped hole principle in vector space"""
        # Encode concept text
        concept_tokens = self.tokenizer(
            concept, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        concept_encoding = self.encoder(**concept_tokens).last_hidden_state.mean(dim=1)
        
        # Encode not-space
        if not_space:
            not_tokens = self.tokenizer(
                not_space, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            not_encoding = self.encoder(**not_tokens).last_hidden_state.mean(dim=1)
            not_aggregate = not_encoding.mean(dim=0)
        else:
            not_aggregate = torch.zeros(768).to(self.device)
        
        # Create universal space (all ones, normalized)
        universal = torch.ones(768).to(self.device)
        universal = universal / universal.norm()
        
        # Compute hole: universal - not_space
        hole = universal - not_aggregate
        
        # Project through hole network
        hole_vector = self.hole_projector(hole)
        
        # Combine with concept encoding
        final_vector = hole_vector * concept_encoding.squeeze()
        
        return final_vector / (final_vector.norm() + 1e-8)
    
    def forward(self, definition: ConceptDefinition) -> VectorizedObject:
        """Generate complete vectorized object from definition"""
        # Encode atomic pattern
        pattern_vector = self.encode_pattern(definition.atomic_pattern)
        
        # Compute X-shaped hole
        hole_vector = self.compute_x_shaped_hole(
            definition.name, 
            list(definition.not_space)
        )
        
        # Combine pattern and hole
        combined = (pattern_vector + hole_vector) / 2
        
        # Predict null ratio
        null_ratio = self.null_predictor(combined).item()
        
        # Scale by null ratio (less null = stronger vector)
        final_vector = combined * (1 - null_ratio)
        
        # Compute empathy scores
        empathy_scores = self.compute_empathy_scores(final_vector)
        
        # Create not-space vector
        not_space_vector = self.encode_not_space(definition.not_space)
        
        return VectorizedObject(
            concept_id=definition.id,
            vector=final_vector.detach().cpu().numpy(),
            null_ratio=null_ratio,
            not_space_vector=not_space_vector.detach().cpu().numpy(),
            empathy_scores=empathy_scores,
            cultural_variants={},  # Computed separately
            metadata={
                "pattern_complexity": len(str(definition.atomic_pattern)),
                "not_space_size": len(definition.not_space),
                "confidence": definition.confidence
            }
        )
    
    def compute_empathy_scores(self, vector: torch.Tensor) -> Dict[str, float]:
        """Calculate empathy scores for co-existence optimization"""
        # Self-attention for empathy
        v = vector.unsqueeze(0).unsqueeze(0)
        empathy_output, attention_weights = self.empathy_attention(v, v, v)
        
        # Extract scores
        scores = {
            "self_empathy": attention_weights[0, 0, 0].item(),
            "other_empathy": 1 - attention_weights[0, 0, 0].item(),
            "mutual_empathy": empathy_output.squeeze().norm().item()
        }
        
        return scores
    
    def encode_not_space(self, not_space: Set[str]) -> torch.Tensor:
        """Encode the not-space into a single vector"""
        if not not_space:
            return torch.zeros(768)
            
        not_tokens = self.tokenizer(
            list(not_space), 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        not_encoding = self.encoder(**not_tokens).last_hidden_state
        return not_encoding.mean(dim=[0, 1])


class RelationshipAwareVectorizer:
    """
    Vectorizer that is aware of logical relationships between concepts.
    Used by the SphericalRelationshipVectorizer to generate base vectors.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cuda"):
        """
        Initialize the relationship-aware vectorizer
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to use (cuda or cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Pre-trained language model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move to device
        self.encoder.to(self.device)
        
        # Dimension of vectors
        self.dimension = 768
    
    async def generate_relationship_aware_vector(self, primitive_def: PrimitiveDefinition) -> np.ndarray:
        """
        Generate a vector that is aware of relationships
        
        Args:
            primitive_def: PrimitiveDefinition
            
        Returns:
            Vector
        """
        # Encode concept
        concept_vector = await self._encode_concept(primitive_def.concept)
        
        # Encode not space
        not_space = primitive_def.properties.get("not_space", [])
        not_space_vector = await self._encode_not_space(not_space)
        
        # Encode relationships
        and_vector = await self._encode_relationships(primitive_def.and_relationships)
        or_vector = await self._encode_relationships(primitive_def.or_relationships)
        not_vector = await self._encode_relationships(primitive_def.not_relationships)
        
        # Combine vectors
        combined_vector = concept_vector * 0.4 + not_space_vector * 0.2 + and_vector * 0.2 + or_vector * 0.1 + not_vector * 0.1
        
        # Normalize
        combined_vector = combined_vector / np.linalg.norm(combined_vector)
        
        return combined_vector
    
    async def _encode_concept(self, concept: str) -> np.ndarray:
        """
        Encode a concept
        
        Args:
            concept: Concept name
            
        Returns:
            Vector
        """
        # Tokenize
        tokens = self.tokenizer(
            concept,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            output = self.encoder(**tokens)
        
        # Get vector
        vector = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return vector
    
    async def _encode_not_space(self, not_space: List[str]) -> np.ndarray:
        """
        Encode not space
        
        Args:
            not_space: List of concepts in not space
            
        Returns:
            Vector
        """
        if not not_space:
            return np.zeros(self.dimension)
        
        # Tokenize
        tokens = self.tokenizer(
            not_space,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            output = self.encoder(**tokens)
        
        # Get vector
        vector = output.last_hidden_state.mean(dim=[0, 1]).cpu().numpy()
        
        return vector
    
    async def _encode_relationships(self, relationships: List[ExistenceRelationship]) -> np.ndarray:
        """
        Encode relationships
        
        Args:
            relationships: List of ExistenceRelationship
            
        Returns:
            Vector
        """
        if not relationships:
            return np.zeros(self.dimension)
        
        # Get concepts
        concepts = [rel.concept for rel in relationships]
        
        # Tokenize
        tokens = self.tokenizer(
            concepts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            output = self.encoder(**tokens)
        
        # Get vectors
        vectors = output.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Weight by strength
        weighted_vectors = np.zeros_like(vectors[0])
        
        for i, rel in enumerate(relationships):
            weighted_vectors += vectors[i] * rel.strength
        
        # Normalize
        if np.linalg.norm(weighted_vectors) > 0:
            weighted_vectors = weighted_vectors / np.linalg.norm(weighted_vectors)
        
        return weighted_vectors
    
    async def generate_vector_batch(self, primitive_defs: List[PrimitiveDefinition]) -> Dict[str, np.ndarray]:
        """
        Generate vectors for multiple concepts
        
        Args:
            primitive_defs: List of PrimitiveDefinition
            
        Returns:
            Dictionary of concept names to vectors
        """
        results = {}
        
        for primitive_def in primitive_defs:
            vector = await self.generate_relationship_aware_vector(primitive_def)
            results[primitive_def.concept] = vector
        
        return results
