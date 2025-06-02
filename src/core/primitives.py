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
