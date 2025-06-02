"""
Core engine components for LEXICON.
"""

from src.core.primitives import (
    ExistencePrimitive, 
    Exists, 
    NotExists, 
    CoExists, 
    AltExists,
    EXISTS,
    NOT_EXISTS,
    CO_EXISTS,
    ALT_EXISTS,
    PRIMITIVES,
    get_primitive,
    combine_primitives,
    parse_pattern
)

from src.core.types import (
    Primitive,
    ExistencePattern,
    ConceptDefinition,
    VectorizedObject,
    NormalizationResult,
    MemeticState
)

from src.core.reducer import PrimitiveReducer
from src.core.x_shaped_hole import XShapedHoleEngine, XShapedHoleDefinition

__all__ = [
    # Primitives
    'ExistencePrimitive', 'Exists', 'NotExists', 'CoExists', 'AltExists',
    'EXISTS', 'NOT_EXISTS', 'CO_EXISTS', 'ALT_EXISTS', 'PRIMITIVES',
    'get_primitive', 'combine_primitives', 'parse_pattern',
    
    # Types
    'Primitive', 'ExistencePattern', 'ConceptDefinition', 'VectorizedObject',
    'NormalizationResult', 'MemeticState',
    
    # Engines
    'PrimitiveReducer', 'XShapedHoleEngine', 'XShapedHoleDefinition'
]
