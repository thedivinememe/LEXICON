"""
Primitive Reducer Engine for LEXICON.
Reduces complex concepts to atomic existence patterns.
"""

from typing import Set, List, Dict, Any
import re
import numpy as np
from src.core.types import ExistencePattern, Primitive

class PrimitiveReducer:
    """Engine for reducing concepts to atomic existence patterns"""
    
    def __init__(self):
        """Initialize the primitive reducer"""
        # Primitive patterns
        self.primitives = {
            "exists": Primitive.EXISTS,
            "not_exists": Primitive.NOT_EXISTS,
            "co_exists": Primitive.CO_EXISTS,
            "alt_exists": Primitive.ALT_EXISTS
        }
        
        # Pattern templates
        self.templates = {
            "binary": [
                # A and B
                (r"(\w+)\s+and\s+(\w+)", "co_exists"),
                # A or B
                (r"(\w+)\s+or\s+(\w+)", "alt_exists"),
                # A with B
                (r"(\w+)\s+with\s+(\w+)", "co_exists"),
                # A without B
                (r"(\w+)\s+without\s+(\w+)", "not_co_exists")
            ],
            "unary": [
                # not A
                (r"not\s+(\w+)", "not_exists"),
                # no A
                (r"no\s+(\w+)", "not_exists")
            ]
        }
    
    def reduce_to_primitives(self, concept: str, not_space: Set[str]) -> ExistencePattern:
        """
        Reduce a concept to its atomic existence pattern.
        
        Args:
            concept: The concept name
            not_space: Set of concepts that define what this concept is not
        
        Returns:
            ExistencePattern: The atomic pattern representing the concept
        """
        # Simple case: single word concept
        if re.match(r"^\w+$", concept) and not self._is_negation(concept):
            return ExistencePattern(
                pattern=[Primitive.EXISTS],
                confidence=1.0
            )
        
        # Check for compound patterns
        pattern_elements = []
        confidence = 1.0
        
        # Try to match binary patterns
        for regex, pattern_type in self.templates["binary"]:
            matches = re.findall(regex, concept, re.IGNORECASE)
            if matches:
                for match in matches:
                    a, b = match
                    if pattern_type == "co_exists":
                        pattern_elements.append(Primitive.CO_EXISTS)
                        sub_pattern_a = self.reduce_to_primitives(a, not_space)
                        sub_pattern_b = self.reduce_to_primitives(b, not_space)
                        pattern_elements.append(sub_pattern_a)
                        pattern_elements.append(sub_pattern_b)
                        confidence *= 0.9  # Reduce confidence for compound patterns
                    elif pattern_type == "alt_exists":
                        pattern_elements.append(Primitive.ALT_EXISTS)
                        sub_pattern_a = self.reduce_to_primitives(a, not_space)
                        sub_pattern_b = self.reduce_to_primitives(b, not_space)
                        pattern_elements.append(sub_pattern_a)
                        pattern_elements.append(sub_pattern_b)
                        confidence *= 0.85  # Alternatives are less confident
                    elif pattern_type == "not_co_exists":
                        pattern_elements.append(Primitive.CO_EXISTS)
                        sub_pattern_a = self.reduce_to_primitives(a, not_space)
                        sub_pattern_b = ExistencePattern(
                            pattern=[Primitive.NOT_EXISTS],
                            confidence=0.9
                        )
                        pattern_elements.append(sub_pattern_a)
                        pattern_elements.append(sub_pattern_b)
                        confidence *= 0.8  # Negations reduce confidence
        
        # Try to match unary patterns
        for regex, pattern_type in self.templates["unary"]:
            matches = re.findall(regex, concept, re.IGNORECASE)
            if matches:
                for match in matches:
                    if pattern_type == "not_exists":
                        pattern_elements.append(Primitive.NOT_EXISTS)
                        sub_pattern = self.reduce_to_primitives(match, not_space)
                        pattern_elements.append(sub_pattern)
                        confidence *= 0.75  # Negations reduce confidence more
        
        # If no patterns matched, use the not-space to infer pattern
        if not pattern_elements:
            # Calculate overlap with not-space
            overlap_score = self._calculate_not_space_overlap(concept, not_space)
            
            if overlap_score > 0.7:
                # High overlap with not-space suggests negation
                pattern_elements.append(Primitive.NOT_EXISTS)
                confidence *= (1 - overlap_score)  # Lower confidence based on overlap
            else:
                # Default to EXISTS with confidence based on not-space distance
                pattern_elements.append(Primitive.EXISTS)
                confidence *= (1 - overlap_score * 0.5)  # Moderate confidence reduction
        
        # If still no elements, default to EXISTS
        if not pattern_elements:
            pattern_elements.append(Primitive.EXISTS)
            confidence = 0.5  # Low confidence for default case
        
        return ExistencePattern(
            pattern=pattern_elements,
            confidence=max(0.1, min(1.0, confidence))  # Clamp to [0.1, 1.0]
        )
    
    def _is_negation(self, word: str) -> bool:
        """Check if a word represents a negation"""
        negation_prefixes = ["non", "un", "in", "dis", "anti"]
        return any(word.lower().startswith(prefix) for prefix in negation_prefixes)
    
    def _calculate_not_space_overlap(self, concept: str, not_space: Set[str]) -> float:
        """
        Calculate the semantic overlap between a concept and its not-space.
        
        This is a simplified implementation. In a real system, this would use
        word embeddings or other semantic similarity measures.
        """
        if not not_space:
            return 0.0
        
        # Simple character-based similarity
        concept_chars = set(concept.lower())
        
        similarities = []
        for neg_concept in not_space:
            neg_chars = set(neg_concept.lower())
            
            # Jaccard similarity
            intersection = len(concept_chars.intersection(neg_chars))
            union = len(concept_chars.union(neg_chars))
            
            if union > 0:
                similarities.append(intersection / union)
        
        return max(similarities) if similarities else 0.0
    
    def simplify_pattern(self, pattern: ExistencePattern) -> ExistencePattern:
        """
        Simplify a complex pattern by applying reduction rules.
        
        For example:
        - NOT(NOT(A)) -> A
        - AND(A, A) -> A
        - OR(A, NOT(A)) -> EXISTS (tautology)
        """
        # Base case: primitive pattern
        if len(pattern.pattern) == 1 and isinstance(pattern.pattern[0], Primitive):
            return pattern
        
        # Recursive simplification
        simplified_elements = []
        for element in pattern.pattern:
            if isinstance(element, ExistencePattern):
                simplified_elements.append(self.simplify_pattern(element))
            else:
                simplified_elements.append(element)
        
        # Apply reduction rules
        if len(simplified_elements) >= 3:
            op = simplified_elements[0]
            
            # Double negation: NOT(NOT(A)) -> A
            if op == Primitive.NOT_EXISTS and len(simplified_elements) == 2:
                sub_pattern = simplified_elements[1]
                if isinstance(sub_pattern, ExistencePattern) and len(sub_pattern.pattern) >= 1:
                    if sub_pattern.pattern[0] == Primitive.NOT_EXISTS:
                        # Extract the inner pattern
                        inner_pattern = sub_pattern.pattern[1:]
                        return ExistencePattern(
                            pattern=inner_pattern,
                            confidence=pattern.confidence * 0.9  # Reduce confidence slightly
                        )
            
            # Idempotence: AND(A, A) -> A or OR(A, A) -> A
            if (op == Primitive.CO_EXISTS or op == Primitive.ALT_EXISTS) and len(simplified_elements) == 3:
                if simplified_elements[1] == simplified_elements[2]:
                    return ExistencePattern(
                        pattern=[simplified_elements[1]],
                        confidence=pattern.confidence
                    )
            
            # Tautology: OR(A, NOT(A)) -> EXISTS
            if op == Primitive.ALT_EXISTS and len(simplified_elements) == 3:
                a = simplified_elements[1]
                b = simplified_elements[2]
                
                if isinstance(b, ExistencePattern) and len(b.pattern) >= 2:
                    if b.pattern[0] == Primitive.NOT_EXISTS and b.pattern[1] == a:
                        return ExistencePattern(
                            pattern=[Primitive.EXISTS],
                            confidence=1.0  # Tautologies are certain
                        )
                
                if isinstance(a, ExistencePattern) and len(a.pattern) >= 2:
                    if a.pattern[0] == Primitive.NOT_EXISTS and a.pattern[1] == b:
                        return ExistencePattern(
                            pattern=[Primitive.EXISTS],
                            confidence=1.0  # Tautologies are certain
                        )
        
        # If no rules applied, return with simplified elements
        return ExistencePattern(
            pattern=simplified_elements,
            confidence=pattern.confidence
        )
