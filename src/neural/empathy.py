"""
Empathy Normalizer for LEXICON.
Implements empathy-based normalization for concept vectors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class EmpathyNormalizer:
    """Calculate empathy scores for co-existence optimization"""
    
    def __init__(self, vector_dim: int = 768, device: str = None):
        """
        Initialize the empathy normalizer.
        
        Args:
            vector_dim: Dimension of concept vectors
            device: Computation device ('cuda' or 'cpu')
        """
        self.vector_dim = vector_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create attention mechanism for empathy calculation
        self.attention = nn.MultiheadAttention(
            embed_dim=vector_dim,
            num_heads=8,
            batch_first=True,
            device=self.device
        )
        
        # Empathy projection layers
        self.empathy_projector = nn.Sequential(
            nn.Linear(vector_dim, vector_dim // 2, device=self.device),
            nn.LayerNorm(vector_dim // 2, device=self.device),
            nn.ReLU(),
            nn.Linear(vector_dim // 2, vector_dim, device=self.device),
            nn.Tanh()
        )
        
        # Empathy score predictor
        self.score_predictor = nn.Sequential(
            nn.Linear(vector_dim * 2, 256, device=self.device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64, device=self.device),
            nn.ReLU(),
            nn.Linear(64, 1, device=self.device),
            nn.Sigmoid()
        )
    
    def calculate_empathy_score(self, vector1: Union[np.ndarray, torch.Tensor], 
                              vector2: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Return empathy score in [0, 1]
        
        1.0 = maximum co-existence benefit
        0.0 = maximum harm
        
        Args:
            vector1: First concept vector
            vector2: Second concept vector
            
        Returns:
            Empathy score between 0.0 and 1.0
        """
        # Convert to torch tensors if needed
        if isinstance(vector1, np.ndarray):
            vector1 = torch.tensor(vector1, dtype=torch.float32, device=self.device)
        if isinstance(vector2, np.ndarray):
            vector2 = torch.tensor(vector2, dtype=torch.float32, device=self.device)
        
        # Ensure vectors are on the correct device
        vector1 = vector1.to(self.device)
        vector2 = vector2.to(self.device)
        
        # Reshape for attention
        v1 = vector1.view(1, 1, -1)  # [1, 1, vector_dim]
        v2 = vector2.view(1, 1, -1)  # [1, 1, vector_dim]
        
        # Calculate attention between vectors
        combined = torch.cat([v1, v2], dim=1)  # [1, 2, vector_dim]
        attn_output, attn_weights = self.attention(combined, combined, combined)
        
        # Extract cross-attention score (how much they attend to each other)
        cross_attention = attn_weights[0, 0, 1].item()  # Attention from v1 to v2
        
        # Project through empathy layers
        v1_projected = self.empathy_projector(v1)
        v2_projected = self.empathy_projector(v2)
        
        # Calculate cosine similarity after projection
        v1_norm = torch.nn.functional.normalize(v1_projected, p=2, dim=2)
        v2_norm = torch.nn.functional.normalize(v2_projected, p=2, dim=2)
        cosine_sim = torch.sum(v1_norm * v2_norm).item()
        
        # Combine features for final score
        combined_features = torch.cat([
            v1.view(-1),
            v2.view(-1)
        ])
        
        # Predict empathy score
        with torch.no_grad():
            empathy_score = self.score_predictor(combined_features).item()
        
        # Adjust score based on attention and similarity
        adjusted_score = (
            0.5 * empathy_score +
            0.3 * cross_attention +
            0.2 * (cosine_sim + 1) / 2  # Map from [-1, 1] to [0, 1]
        )
        
        return max(0.0, min(1.0, adjusted_score))
    
    def normalize_vectors(self, vectors: List[Union[np.ndarray, torch.Tensor]]) -> Tuple[List[np.ndarray], Dict[Tuple[int, int], float]]:
        """
        Normalize a set of vectors for optimal co-existence.
        
        This method adjusts vectors to maximize their collective empathy scores,
        allowing concepts to co-exist more harmoniously in the vector space.
        
        Args:
            vectors: List of concept vectors to normalize
            
        Returns:
            Tuple containing:
            - List of normalized vectors
            - Dictionary mapping (i, j) pairs to their empathy scores
        """
        if not vectors:
            return [], {}
        
        # Convert to torch tensors if needed
        torch_vectors = []
        for v in vectors:
            if isinstance(v, np.ndarray):
                torch_vectors.append(torch.tensor(v, dtype=torch.float32, device=self.device))
            else:
                torch_vectors.append(v.to(self.device))
        
        # Calculate pairwise empathy scores
        n = len(torch_vectors)
        empathy_scores = {}
        
        for i in range(n):
            for j in range(i+1, n):
                score = self.calculate_empathy_score(torch_vectors[i], torch_vectors[j])
                empathy_scores[(i, j)] = score
                empathy_scores[(j, i)] = score
        
        # Normalize vectors to maximize collective empathy
        normalized_vectors = []
        
        for i, vec in enumerate(torch_vectors):
            # Calculate weighted average direction based on empathy
            weighted_sum = torch.zeros_like(vec)
            weight_sum = 0.0
            
            for j, other_vec in enumerate(torch_vectors):
                if i == j:
                    continue
                
                # Get empathy score
                score = empathy_scores[(i, j)]
                
                # Higher empathy = stronger influence
                weight = score ** 2  # Square to emphasize high empathy
                weighted_sum += weight * other_vec
                weight_sum += weight
            
            # If no meaningful weights, keep original
            if weight_sum < 1e-6:
                normalized = vec
            else:
                # Calculate direction adjustment
                direction = weighted_sum / weight_sum
                
                # Mix original with empathy direction (80% original, 20% empathy)
                normalized = 0.8 * vec + 0.2 * direction
                
                # Renormalize to unit length
                normalized = torch.nn.functional.normalize(normalized, p=2, dim=0)
            
            # Convert back to numpy
            normalized_vectors.append(normalized.cpu().numpy())
        
        return normalized_vectors, empathy_scores
    
    def calculate_group_empathy(self, vectors: List[Union[np.ndarray, torch.Tensor]]) -> float:
        """
        Calculate the overall empathy score for a group of vectors.
        
        Higher scores indicate better collective co-existence.
        
        Args:
            vectors: List of concept vectors
            
        Returns:
            Group empathy score between 0.0 and 1.0
        """
        if len(vectors) <= 1:
            return 1.0  # Perfect empathy for single vector or empty list
        
        # Calculate all pairwise empathy scores
        n = len(vectors)
        total_score = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                score = self.calculate_empathy_score(vectors[i], vectors[j])
                total_score += score
                count += 1
        
        # Average empathy score
        return total_score / count if count > 0 else 0.0
    
    def find_optimal_addition(self, existing_vectors: List[Union[np.ndarray, torch.Tensor]], 
                            candidate_vectors: List[Union[np.ndarray, torch.Tensor]]) -> Tuple[int, float]:
        """
        Find the candidate vector that would maximize group empathy if added.
        
        Args:
            existing_vectors: List of vectors already in the group
            candidate_vectors: List of vectors to consider adding
            
        Returns:
            Tuple containing:
            - Index of the optimal candidate
            - New group empathy score if that candidate is added
        """
        if not candidate_vectors:
            return -1, self.calculate_group_empathy(existing_vectors)
        
        best_index = -1
        best_score = -1.0
        
        # Try adding each candidate
        for i, candidate in enumerate(candidate_vectors):
            # Create test group with this candidate
            test_group = existing_vectors + [candidate]
            
            # Calculate group empathy
            score = self.calculate_group_empathy(test_group)
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_index = i
        
        return best_index, best_score
