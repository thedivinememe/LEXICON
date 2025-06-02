"""
Existence Primitives for LEXICON.
Defines the fundamental building blocks of existence patterns.
"""

from enum import Enum
from typing import List, Dict, Set, Optional, Union, Tuple
import numpy as np

class ExistencePrimitive:
    """Base class for existence primitives"""
    
    def __init__(self, symbol: str, name: str, description: str):
        self.symbol = symbol
        self.name = name
        self.description = description
    
    def __str__(self) -> str:
        return self.symbol
    
    def __repr__(self) -> str:
        return f"{self.name}({self.symbol})"
    
    def to_vector(self, dimension: int = 768) -> np.ndarray:
        """Convert primitive to vector representation"""
        # This would be implemented with proper orthogonal vectors
        # Here we just create a random unit vector for demonstration
        np.random.seed(hash(self.symbol) % 2**32)
        vector = np.random.randn(dimension)
        return vector / np.linalg.norm(vector)
    
    def reduce_to_primitive(self, pattern: Union[str, List, Dict]) -> Union[str, List]:
        """
        Reduce any pattern to 1/!1 representation.
        
        This method takes a complex pattern and reduces it to its most basic
        existence representation (1 or !1).
        
        Args:
            pattern: The pattern to reduce, can be a string, list, or dictionary
                    representing a complex existence pattern
        
        Returns:
            The reduced pattern as either "1", "!1", or a list of primitives
        """
        # Base case: pattern is already a primitive
        if isinstance(pattern, str) and pattern in ["1", "!1"]:
            return pattern
        
        # If pattern is a string but not a primitive, interpret as existing concept
        if isinstance(pattern, str):
            return "1"  # Anything named exists by default
        
        # If pattern is a list with a primitive as first element
        if isinstance(pattern, list) and len(pattern) > 0:
            if pattern[0] == "!1" or pattern[0] == NOT_EXISTS:
                # Negation inverts existence
                if len(pattern) > 1:
                    sub_reduction = self.reduce_to_primitive(pattern[1])
                    if sub_reduction == "1":
                        return "!1"
                    else:
                        return "1"
                return "!1"
            
            elif pattern[0] == "&&" or pattern[0] == CO_EXISTS:
                # Co-existence: all must exist, so if any doesn't exist, result is !1
                for element in pattern[1:]:
                    if self.reduce_to_primitive(element) == "!1":
                        return "!1"
                return "1"
            
            elif pattern[0] == "||" or pattern[0] == ALT_EXISTS:
                # Alternative existence: at least one must exist
                all_not_exist = True
                for element in pattern[1:]:
                    if self.reduce_to_primitive(element) == "1":
                        all_not_exist = False
                        break
                return "!1" if all_not_exist else "1"
        
        # Default case: if we can't determine non-existence, assume existence
        return "1"
    
    def calculate_existence_ratio(self, pattern: Union[str, List, Dict]) -> float:
        """
        Return ratio of 1 to !1 in pattern.
        
        This method calculates how much of a pattern represents existence vs. non-existence.
        A ratio of 1.0 means pure existence, 0.0 means pure non-existence,
        and values in between represent mixed patterns.
        
        Args:
            pattern: The pattern to analyze, can be a string, list, or dictionary
        
        Returns:
            Float between 0.0 and 1.0 representing the existence ratio
        """
        # Base cases
        if isinstance(pattern, str):
            if pattern == "1":
                return 1.0
            elif pattern == "!1":
                return 0.0
            else:
                return 1.0  # Named concepts exist by default
        
        # For lists with primitive as first element
        if isinstance(pattern, list) and len(pattern) > 0:
            if pattern[0] == "!1" or pattern[0] == NOT_EXISTS:
                # Negation inverts the existence ratio
                if len(pattern) > 1:
                    sub_ratio = self.calculate_existence_ratio(pattern[1])
                    return 1.0 - sub_ratio
                return 0.0
            
            elif pattern[0] == "&&" or pattern[0] == CO_EXISTS:
                # Co-existence: average of all sub-patterns
                if len(pattern) == 1:
                    return 1.0
                
                total_ratio = 0.0
                for element in pattern[1:]:
                    total_ratio += self.calculate_existence_ratio(element)
                
                return total_ratio / (len(pattern) - 1)
            
            elif pattern[0] == "||" or pattern[0] == ALT_EXISTS:
                # Alternative existence: maximum of all sub-patterns
                if len(pattern) == 1:
                    return 0.0
                
                max_ratio = 0.0
                for element in pattern[1:]:
                    ratio = self.calculate_existence_ratio(element)
                    max_ratio = max(max_ratio, ratio)
                
                return max_ratio
        
        # Default case
        return 0.5  # Uncertain patterns are 50/50

class Exists(ExistencePrimitive):
    """The fundamental primitive of existence"""
    
    def __init__(self):
        super().__init__(
            symbol="1",
            name="EXISTS",
            description="Fundamental assertion of existence"
        )
    
    def apply(self, concept: str) -> str:
        """Apply EXISTS to a concept"""
        return concept
    
    def to_vector(self, dimension: int = 768) -> np.ndarray:
        """Convert EXISTS to vector representation"""
        # EXISTS is represented as a unit vector along the first dimension
        vector = np.zeros(dimension)
        vector[0] = 1.0
        return vector

class NotExists(ExistencePrimitive):
    """Negation of existence"""
    
    def __init__(self):
        super().__init__(
            symbol="!1",
            name="NOT_EXISTS",
            description="Negation of existence"
        )
    
    def apply(self, concept: str) -> str:
        """Apply NOT_EXISTS to a concept"""
        return f"not {concept}"
    
    def to_vector(self, dimension: int = 768) -> np.ndarray:
        """Convert NOT_EXISTS to vector representation"""
        # NOT_EXISTS is represented as the negative of the EXISTS vector
        exists_vector = Exists().to_vector(dimension)
        return -exists_vector

class CoExists(ExistencePrimitive):
    """Co-existence of multiple concepts"""
    
    def __init__(self):
        super().__init__(
            symbol="&&",
            name="CO_EXISTS",
            description="Co-existence of multiple concepts"
        )
    
    def apply(self, *concepts: str) -> str:
        """Apply CO_EXISTS to multiple concepts"""
        if not concepts:
            return ""
        elif len(concepts) == 1:
            return concepts[0]
        else:
            return " and ".join(concepts)
    
    def to_vector(self, dimension: int = 768) -> np.ndarray:
        """Convert CO_EXISTS to vector representation"""
        # CO_EXISTS is represented as a unit vector along the second dimension
        vector = np.zeros(dimension)
        vector[1] = 1.0
        return vector

class AltExists(ExistencePrimitive):
    """Alternative existence (one of multiple concepts)"""
    
    def __init__(self):
        super().__init__(
            symbol="||",
            name="ALT_EXISTS",
            description="Alternative existence (one of multiple concepts)"
        )
    
    def apply(self, *concepts: str) -> str:
        """Apply ALT_EXISTS to multiple concepts"""
        if not concepts:
            return ""
        elif len(concepts) == 1:
            return concepts[0]
        else:
            return " or ".join(concepts)
    
    def to_vector(self, dimension: int = 768) -> np.ndarray:
        """Convert ALT_EXISTS to vector representation"""
        # ALT_EXISTS is represented as a unit vector along the third dimension
        vector = np.zeros(dimension)
        vector[2] = 1.0
        return vector

# Singleton instances
EXISTS = Exists()
NOT_EXISTS = NotExists()
CO_EXISTS = CoExists()
ALT_EXISTS = AltExists()

# Dictionary of all primitives
PRIMITIVES = {
    "1": EXISTS,
    "!1": NOT_EXISTS,
    "&&": CO_EXISTS,
    "||": ALT_EXISTS
}

def get_primitive(symbol: str) -> Optional[ExistencePrimitive]:
    """Get a primitive by its symbol"""
    return PRIMITIVES.get(symbol)

def combine_primitives(op: ExistencePrimitive, 
                     *elements: Union[ExistencePrimitive, str]) -> str:
    """Combine primitives into a pattern string"""
    if op == EXISTS:
        if len(elements) == 0:
            return "1"
        elif len(elements) == 1:
            return str(elements[0])
        else:
            raise ValueError("EXISTS can only be applied to a single element")
    
    elif op == NOT_EXISTS:
        if len(elements) == 0:
            return "!1"
        elif len(elements) == 1:
            return f"!({elements[0]})"
        else:
            raise ValueError("NOT_EXISTS can only be applied to a single element")
    
    elif op == CO_EXISTS:
        if len(elements) == 0:
            return "&&"
        else:
            return f"&&({', '.join(str(e) for e in elements)})"
    
    elif op == ALT_EXISTS:
        if len(elements) == 0:
            return "||"
        else:
            return f"||({', '.join(str(e) for e in elements)})"
    
    else:
        raise ValueError(f"Unknown primitive: {op}")

def parse_pattern(pattern_str: str) -> Union[ExistencePrimitive, List]:
    """Parse a pattern string into primitives and nested patterns"""
    pattern_str = pattern_str.strip()
    
    # Simple primitive
    if pattern_str in PRIMITIVES:
        return PRIMITIVES[pattern_str]
    
    # Negation
    if pattern_str.startswith("!(") and pattern_str.endswith(")"):
        inner = pattern_str[2:-1]
        return [NOT_EXISTS, parse_pattern(inner)]
    
    # Co-existence
    if pattern_str.startswith("&&(") and pattern_str.endswith(")"):
        inner = pattern_str[3:-1]
        elements = _split_pattern_elements(inner)
        return [CO_EXISTS] + [parse_pattern(e) for e in elements]
    
    # Alternative existence
    if pattern_str.startswith("||(") and pattern_str.endswith(")"):
        inner = pattern_str[3:-1]
        elements = _split_pattern_elements(inner)
        return [ALT_EXISTS] + [parse_pattern(e) for e in elements]
    
    # If not a pattern, treat as a concept string
    return pattern_str

def _split_pattern_elements(pattern_str: str) -> List[str]:
    """Split a pattern string into its elements, respecting nested patterns"""
    elements = []
    current = ""
    paren_depth = 0
    
    for char in pattern_str:
        if char == ',' and paren_depth == 0:
            elements.append(current.strip())
            current = ""
        else:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            current += char
    
    if current:
        elements.append(current.strip())
    
    return elements
