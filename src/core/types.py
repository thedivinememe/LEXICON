from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Union
from enum import Enum
import numpy as np
from datetime import datetime

class Primitive(Enum):
    EXISTS = "1"
    NOT_EXISTS = "!1"
    CO_EXISTS = "&&"
    ALT_EXISTS = "||"

@dataclass
class ExistencePattern:
    """Represents a pattern of existence primitives"""
    pattern: List[Union[Primitive, 'ExistencePattern']]
    confidence: float = 1.0
    
    def to_vector(self) -> np.ndarray:
        """Convert pattern to vector representation"""
        # Implementation in vectorizer
        pass

@dataclass
class ConceptDefinition:
    """Core concept definition"""
    id: str
    name: str
    atomic_pattern: ExistencePattern
    not_space: Set[str]
    confidence: float
    created_at: datetime
    updated_at: datetime
    
@dataclass
class VectorizedObject:
    """Vectorized representation of a concept"""
    concept_id: str
    vector: np.ndarray  # 768-dimensional
    null_ratio: float
    not_space_vector: np.ndarray
    empathy_scores: Dict[str, float]
    cultural_variants: Dict[str, np.ndarray]
    metadata: Dict[str, any]

@dataclass
class NormalizationResult:
    """Result of empathetic normalization"""
    original_vector: np.ndarray
    normalized_vector: np.ndarray
    empathy_score: float
    comparable_sets: List[str]
    normalization_context: Dict[str, any]

@dataclass
class MemeticState:
    """Memetic evolution state"""
    concept_id: str
    generation: int
    fitness_score: float
    replication_count: int
    mutation_history: List[Dict]
    cultural_adaptations: Dict[str, any]
