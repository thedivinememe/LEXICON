"""
X-Shaped Hole Engine for LEXICON.
Implements the X-shaped hole principle for concept definition through negation.
"""

from typing import Set, List, Dict, Any, Optional
import re
import numpy as np
from dataclasses import dataclass

@dataclass
class XShapedHoleDefinition:
    """Definition created through the X-shaped hole principle"""
    concept: str
    not_space: Set[str]
    confidence: float
    semantic_distance: Dict[str, float]

class XShapedHoleEngine:
    """Engine for defining concepts through negation (X-shaped hole principle)"""
    
    def __init__(self):
        """Initialize the X-shaped hole engine"""
        # Semantic distance cache
        self.distance_cache = {}
        
        # Confidence thresholds
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
    
    def define_through_negation(self, concept: str, user_negations: List[str]) -> XShapedHoleDefinition:
        """
        Define a concept through negation (what it is not).
        
        Args:
            concept: The concept to define
            user_negations: List of concepts that the target concept is not
        
        Returns:
            XShapedHoleDefinition: The definition created through negation
        """
        # Clean and normalize inputs
        concept = self._normalize_concept(concept)
        normalized_negations = [self._normalize_concept(neg) for neg in user_negations]
        
        # Remove duplicates and empty strings
        not_space = set(neg for neg in normalized_negations if neg)
        
        # Calculate semantic distances between concept and negations
        semantic_distances = {}
        for neg in not_space:
            distance = self._calculate_semantic_distance(concept, neg)
            semantic_distances[neg] = distance
        
        # Calculate definition confidence
        confidence = self._calculate_definition_confidence(concept, not_space, semantic_distances)
        
        return XShapedHoleDefinition(
            concept=concept,
            not_space=not_space,
            confidence=confidence,
            semantic_distance=semantic_distances
        )
    
    def expand_not_space(self, definition: XShapedHoleDefinition, 
                        additional_negations: List[str]) -> XShapedHoleDefinition:
        """
        Expand the not-space of an existing definition with additional negations.
        
        Args:
            definition: Existing definition
            additional_negations: Additional concepts that the target concept is not
        
        Returns:
            XShapedHoleDefinition: Updated definition with expanded not-space
        """
        # Normalize additional negations
        normalized_negations = [self._normalize_concept(neg) for neg in additional_negations]
        
        # Add to existing not-space
        expanded_not_space = definition.not_space.union(
            set(neg for neg in normalized_negations if neg)
        )
        
        # Calculate semantic distances for new negations
        semantic_distances = dict(definition.semantic_distance)
        for neg in expanded_not_space:
            if neg not in semantic_distances:
                distance = self._calculate_semantic_distance(definition.concept, neg)
                semantic_distances[neg] = distance
        
        # Recalculate confidence
        confidence = self._calculate_definition_confidence(
            definition.concept, expanded_not_space, semantic_distances
        )
        
        return XShapedHoleDefinition(
            concept=definition.concept,
            not_space=expanded_not_space,
            confidence=confidence,
            semantic_distance=semantic_distances
        )
    
    def refine_not_space(self, definition: XShapedHoleDefinition) -> XShapedHoleDefinition:
        """
        Refine the not-space by removing redundant or contradictory negations.
        
        Args:
            definition: Existing definition
        
        Returns:
            XShapedHoleDefinition: Refined definition with optimized not-space
        """
        # Identify redundant negations (those that are very similar to others)
        redundant = set()
        negations = list(definition.not_space)
        
        for i, neg1 in enumerate(negations):
            for neg2 in negations[i+1:]:
                similarity = self._calculate_similarity(neg1, neg2)
                
                # If very similar, keep the one with higher semantic distance
                if similarity > 0.8:
                    if definition.semantic_distance[neg1] < definition.semantic_distance[neg2]:
                        redundant.add(neg1)
                    else:
                        redundant.add(neg2)
        
        # Remove redundant negations
        refined_not_space = definition.not_space - redundant
        
        # Recalculate confidence
        refined_distances = {
            neg: dist for neg, dist in definition.semantic_distance.items()
            if neg in refined_not_space
        }
        
        confidence = self._calculate_definition_confidence(
            definition.concept, refined_not_space, refined_distances
        )
        
        return XShapedHoleDefinition(
            concept=definition.concept,
            not_space=refined_not_space,
            confidence=confidence,
            semantic_distance=refined_distances
        )
    
    def _normalize_concept(self, concept: str) -> str:
        """Normalize a concept string"""
        # Convert to lowercase
        normalized = concept.lower()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _calculate_semantic_distance(self, concept1: str, concept2: str) -> float:
        """
        Calculate semantic distance between two concepts.
        
        This is a simplified implementation. In a real system, this would use
        word embeddings or other semantic similarity measures.
        """
        # Check cache
        cache_key = f"{concept1}|{concept2}"
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Simple character-based distance
        concept1_chars = set(concept1.lower())
        concept2_chars = set(concept2.lower())
        
        # Jaccard distance
        intersection = len(concept1_chars.intersection(concept2_chars))
        union = len(concept1_chars.union(concept2_chars))
        
        if union == 0:
            distance = 1.0  # Maximum distance for empty sets
        else:
            # Jaccard distance = 1 - Jaccard similarity
            distance = 1.0 - (intersection / union)
        
        # Add some randomness to simulate semantic nuances
        # In a real implementation, this would be based on actual semantic analysis
        np.random.seed(hash(cache_key) % 2**32)
        noise = np.random.normal(0, 0.1)  # Small Gaussian noise
        distance = max(0.0, min(1.0, distance + noise))  # Clamp to [0, 1]
        
        # Cache the result
        self.distance_cache[cache_key] = distance
        
        return distance
    
    def _calculate_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between two concepts (inverse of distance)"""
        return 1.0 - self._calculate_semantic_distance(concept1, concept2)
    
    def _calculate_definition_confidence(self, concept: str, 
                                       not_space: Set[str],
                                       distances: Dict[str, float]) -> float:
        """
        Calculate the confidence of a definition based on its not-space.
        
        Higher confidence when:
        1. Not-space has sufficient elements
        2. Not-space elements are semantically distant from the concept
        3. Not-space elements cover diverse semantic areas
        """
        if not not_space:
            return 0.3  # Low confidence for empty not-space
        
        # Factor 1: Size of not-space
        size_factor = min(1.0, len(not_space) / 5)  # Max out at 5 elements
        
        # Factor 2: Average semantic distance
        avg_distance = sum(distances.values()) / len(distances)
        distance_factor = avg_distance  # Higher distance = higher confidence
        
        # Factor 3: Diversity of not-space
        # Calculate pairwise similarities between negations
        diversity_sum = 0.0
        diversity_count = 0
        
        negations = list(not_space)
        for i, neg1 in enumerate(negations):
            for neg2 in negations[i+1:]:
                similarity = self._calculate_similarity(neg1, neg2)
                diversity_sum += (1.0 - similarity)  # Convert to diversity
                diversity_count += 1
        
        # Average diversity (higher is better)
        diversity_factor = diversity_sum / max(1, diversity_count)
        
        # Combine factors with weights
        confidence = (
            0.3 * size_factor +
            0.4 * distance_factor +
            0.3 * diversity_factor
        )
        
        # Clamp to reasonable range
        return max(0.1, min(0.95, confidence))
