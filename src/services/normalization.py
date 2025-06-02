"""
Normalization service for LEXICON.
Provides methods for normalizing vectors for comparison.
"""

from typing import Dict, List, Any, Optional, Set
import numpy as np
import torch
import torch.nn.functional as F

from src.core.types import NormalizationResult, VectorizedObject

class NormalizationService:
    """Service for normalizing vectors for comparison"""
    
    def __init__(self, app_state: Dict[str, Any]):
        """Initialize the normalization service"""
        self.app_state = app_state
        self.db = app_state["db"]
        self.cache = app_state["cache"]
        self.vector_store = app_state["vector_store"]
        self.vectorizer = app_state["vectorizer"]
        
        # Device for tensor operations
        self.device = torch.device(app_state["config"].device)
    
    async def normalize_set(self, concept_ids: List[str]) -> Dict[str, Any]:
        """
        Normalize a set of concepts for comparison.
        
        Args:
            concept_ids: List of concept IDs to normalize
        
        Returns:
            Dictionary with normalization results
        """
        # Check cache
        cache_key = f"norm:{'|'.join(sorted(concept_ids))}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get concept vectors
        concepts = []
        vectors = []
        
        for concept_id in concept_ids:
            # Get concept from database
            concept = await self.db.concepts.find_one({"id": concept_id})
            if not concept:
                continue
            
            # Get vector from vector store
            vector = self.vector_store.get_vector(concept_id)
            if vector is None:
                continue
            
            concepts.append(concept)
            vectors.append(vector)
        
        if not vectors:
            return {"normalized_concepts": []}
        
        # Convert to torch tensors
        vectors_tensor = torch.tensor(np.array(vectors), dtype=torch.float32).to(self.device)
        
        # Normalize vectors
        normalized_vectors, empathy_scores = self._normalize_vectors(vectors_tensor)
        
        # Create result
        normalized_concepts = []
        for i, concept in enumerate(concepts):
            normalized_concepts.append({
                "concept_id": concept["id"],
                "concept_name": concept["name"],
                "original_vector": vectors[i].tolist(),
                "normalized_vector": normalized_vectors[i].cpu().numpy().tolist(),
                "empathy_score": empathy_scores[i].item(),
                "null_ratio": concept.get("null_ratio", 0.0)
            })
        
        # Calculate pairwise similarities
        similarities = self._calculate_similarities(normalized_vectors)
        
        result = {
            "normalized_concepts": normalized_concepts,
            "similarities": similarities.cpu().numpy().tolist()
        }
        
        # Cache the result
        await self.cache.set(cache_key, result, expire=3600)  # 1 hour
        
        return result
    
    async def normalize_vector(self, vector: np.ndarray) -> NormalizationResult:
        """
        Normalize a single vector.
        
        Args:
            vector: Vector to normalize
        
        Returns:
            NormalizationResult with normalized vector and metadata
        """
        # Convert to torch tensor
        vector_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Normalize vector
        normalized_vectors, empathy_scores = self._normalize_vectors(vector_tensor)
        
        # Get comparable sets
        comparable_sets = await self._find_comparable_sets(normalized_vectors[0])
        
        return NormalizationResult(
            original_vector=vector,
            normalized_vector=normalized_vectors[0].cpu().numpy(),
            empathy_score=empathy_scores[0].item(),
            comparable_sets=comparable_sets,
            normalization_context={}
        )
    
    def _normalize_vectors(self, vectors: torch.Tensor) -> tuple:
        """
        Normalize vectors using empathy attention.
        
        Args:
            vectors: Tensor of vectors to normalize
        
        Returns:
            Tuple of (normalized_vectors, empathy_scores)
        """
        # Use the empathy attention mechanism from the vectorizer
        with torch.no_grad():
            # Apply self-attention for empathy normalization
            attended, attention_weights = self.vectorizer.empathy_attention(
                vectors.unsqueeze(0),
                vectors.unsqueeze(0),
                vectors.unsqueeze(0)
            )
            
            # Extract empathy scores from attention weights
            empathy_scores = attention_weights.squeeze(0).diag()
            
            # Normalize vectors
            normalized = F.normalize(attended.squeeze(0), p=2, dim=1)
            
            return normalized, empathy_scores
    
    def _calculate_similarities(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculate pairwise cosine similarities between vectors.
        
        Args:
            vectors: Tensor of vectors
        
        Returns:
            Tensor of pairwise similarities
        """
        # Normalize vectors
        normalized = F.normalize(vectors, p=2, dim=1)
        
        # Calculate cosine similarities
        similarities = torch.matmul(normalized, normalized.transpose(0, 1))
        
        return similarities
    
    async def _find_comparable_sets(self, normalized_vector: torch.Tensor) -> List[str]:
        """
        Find sets of concepts that are comparable to the normalized vector.
        
        Args:
            normalized_vector: Normalized vector
        
        Returns:
            List of comparable set names
        """
        # This is a placeholder implementation
        # In a real system, this would use a more sophisticated approach
        
        # Convert to numpy for FAISS search
        vector_np = normalized_vector.cpu().numpy()
        
        # Search for similar vectors
        distances, indices = self.vector_store.search(vector_np, k=10)
        
        # Get concept IDs
        concept_ids = [self.vector_store.get_id(idx) for idx in indices[0]]
        
        # Get concept names
        comparable_sets = []
        for concept_id in concept_ids:
            if concept_id:
                concept = await self.db.concepts.find_one({"id": concept_id})
                if concept:
                    comparable_sets.append(concept["name"])
        
        return comparable_sets
