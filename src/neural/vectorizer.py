import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Set
from src.core.types import VectorizedObject, ConceptDefinition, ExistencePattern

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
        if isinstance(pattern.pattern[0], str):
            # Simple primitive
            return self.primitive_embeddings[pattern.pattern[0]].to(self.device)
        
        # Complex pattern - recursive encoding
        vectors = []
        for element in pattern.pattern:
            if isinstance(element, str):
                vectors.append(self.primitive_embeddings[element])
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
