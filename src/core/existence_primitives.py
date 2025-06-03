"""
Existence Primitives for the Spherical Universe.
Defines primitive concepts and their relationships.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio
import random

from src.core.existence_types import ExistenceLevel, ExistenceType, ExistenceTypeSystem

class LogicalConnector(Enum):
    """Logical connectors for relationships"""
    AND = auto()
    OR = auto()
    NOT = auto()
    XOR = auto()
    IMPLIES = auto()
    EQUIVALENT = auto()

@dataclass
class ExistenceRelationship:
    """
    Relationship between concepts.
    
    Attributes:
        concept: Related concept
        connector: Logical connector
        strength: Relationship strength (0 to 1)
        description: Description of the relationship
    """
    concept: str
    connector: LogicalConnector
    strength: float = 1.0
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "concept": self.concept,
            "connector": self.connector.name,
            "strength": self.strength,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExistenceRelationship':
        """Create from dictionary"""
        return cls(
            concept=data["concept"],
            connector=LogicalConnector[data["connector"]],
            strength=data.get("strength", 1.0),
            description=data.get("description", "")
        )

@dataclass
class PrimitiveDefinition:
    """
    Definition of a primitive concept.
    
    Attributes:
        concept: Concept name
        existence_level: Level of existence
        and_relationships: AND relationships
        or_relationships: OR relationships
        not_relationships: NOT relationships
        implies_relationships: IMPLIES relationships
        equivalent_relationships: EQUIVALENT relationships
        xor_relationships: XOR relationships
        spherical_properties: Properties for spherical positioning
    """
    concept: str
    existence_level: int
    and_relationships: List[ExistenceRelationship] = field(default_factory=list)
    or_relationships: List[ExistenceRelationship] = field(default_factory=list)
    not_relationships: List[ExistenceRelationship] = field(default_factory=list)
    implies_relationships: List[ExistenceRelationship] = field(default_factory=list)
    equivalent_relationships: List[ExistenceRelationship] = field(default_factory=list)
    xor_relationships: List[ExistenceRelationship] = field(default_factory=list)
    spherical_properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "concept": self.concept,
            "existence_level": self.existence_level,
            "and_relationships": [rel.to_dict() for rel in self.and_relationships],
            "or_relationships": [rel.to_dict() for rel in self.or_relationships],
            "not_relationships": [rel.to_dict() for rel in self.not_relationships],
            "implies_relationships": [rel.to_dict() for rel in self.implies_relationships],
            "equivalent_relationships": [rel.to_dict() for rel in self.equivalent_relationships],
            "xor_relationships": [rel.to_dict() for rel in self.xor_relationships],
            "spherical_properties": self.spherical_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PrimitiveDefinition':
        """Create from dictionary"""
        return cls(
            concept=data["concept"],
            existence_level=data["existence_level"],
            and_relationships=[ExistenceRelationship.from_dict(rel) for rel in data.get("and_relationships", [])],
            or_relationships=[ExistenceRelationship.from_dict(rel) for rel in data.get("or_relationships", [])],
            not_relationships=[ExistenceRelationship.from_dict(rel) for rel in data.get("not_relationships", [])],
            implies_relationships=[ExistenceRelationship.from_dict(rel) for rel in data.get("implies_relationships", [])],
            equivalent_relationships=[ExistenceRelationship.from_dict(rel) for rel in data.get("equivalent_relationships", [])],
            xor_relationships=[ExistenceRelationship.from_dict(rel) for rel in data.get("xor_relationships", [])],
            spherical_properties=data.get("spherical_properties", {})
        )

class ExistencePrimitiveEngine:
    """
    Engine for generating primitive definitions.
    
    Generates primitive definitions based on:
    - Existence level
    - Relationships to other concepts
    - Spherical properties
    """
    
    def __init__(self, type_system: Optional[ExistenceTypeSystem] = None):
        self.type_system = type_system if type_system else ExistenceTypeSystem()
        self.primitives = {}  # concept -> PrimitiveDefinition
        self.logical_operators = {}  # operator -> PrimitiveDefinition
        self._initialize_logical_operators()
    
    def _initialize_logical_operators(self) -> None:
        """Initialize logical operators"""
        # AND operator
        and_op = PrimitiveDefinition(
            concept="&&",
            existence_level=ExistenceLevel.INFORMATIONAL.value,
            and_relationships=[
                ExistenceRelationship(
                    concept="conjunction",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Logical conjunction"
                ),
                ExistenceRelationship(
                    concept="both",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Both conditions must be true"
                )
            ],
            or_relationships=[
                ExistenceRelationship(
                    concept="operator",
                    connector=LogicalConnector.OR,
                    strength=1.0,
                    description="Logical operator"
                )
            ],
            not_relationships=[
                ExistenceRelationship(
                    concept="||",
                    connector=LogicalConnector.NOT,
                    strength=0.8,
                    description="Not OR"
                )
            ],
            spherical_properties={
                "preferred_r": 0.4,
                "growth_pattern": "crystalline"
            }
        )
        
        # OR operator
        or_op = PrimitiveDefinition(
            concept="||",
            existence_level=ExistenceLevel.INFORMATIONAL.value,
            and_relationships=[
                ExistenceRelationship(
                    concept="disjunction",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Logical disjunction"
                ),
                ExistenceRelationship(
                    concept="either",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Either condition can be true"
                )
            ],
            or_relationships=[
                ExistenceRelationship(
                    concept="operator",
                    connector=LogicalConnector.OR,
                    strength=1.0,
                    description="Logical operator"
                )
            ],
            not_relationships=[
                ExistenceRelationship(
                    concept="&&",
                    connector=LogicalConnector.NOT,
                    strength=0.8,
                    description="Not AND"
                )
            ],
            spherical_properties={
                "preferred_r": 0.4,
                "growth_pattern": "crystalline"
            }
        )
        
        # NOT operator
        not_op = PrimitiveDefinition(
            concept="!",
            existence_level=ExistenceLevel.INFORMATIONAL.value,
            and_relationships=[
                ExistenceRelationship(
                    concept="negation",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Logical negation"
                ),
                ExistenceRelationship(
                    concept="opposite",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Opposite of a condition"
                )
            ],
            or_relationships=[
                ExistenceRelationship(
                    concept="operator",
                    connector=LogicalConnector.OR,
                    strength=1.0,
                    description="Logical operator"
                )
            ],
            not_relationships=[
                ExistenceRelationship(
                    concept="identity",
                    connector=LogicalConnector.NOT,
                    strength=1.0,
                    description="Not identity"
                )
            ],
            spherical_properties={
                "preferred_r": 0.4,
                "growth_pattern": "crystalline"
            }
        )
        
        # XOR operator
        xor_op = PrimitiveDefinition(
            concept="^",
            existence_level=ExistenceLevel.INFORMATIONAL.value,
            and_relationships=[
                ExistenceRelationship(
                    concept="exclusive_or",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Exclusive OR"
                ),
                ExistenceRelationship(
                    concept="difference",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Difference between conditions"
                )
            ],
            or_relationships=[
                ExistenceRelationship(
                    concept="operator",
                    connector=LogicalConnector.OR,
                    strength=1.0,
                    description="Logical operator"
                )
            ],
            not_relationships=[
                ExistenceRelationship(
                    concept="equivalence",
                    connector=LogicalConnector.NOT,
                    strength=1.0,
                    description="Not equivalence"
                )
            ],
            spherical_properties={
                "preferred_r": 0.4,
                "growth_pattern": "crystalline"
            }
        )
        
        # IMPLIES operator
        implies_op = PrimitiveDefinition(
            concept="->",
            existence_level=ExistenceLevel.INFORMATIONAL.value,
            and_relationships=[
                ExistenceRelationship(
                    concept="implication",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Logical implication"
                ),
                ExistenceRelationship(
                    concept="if_then",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="If-then relationship"
                )
            ],
            or_relationships=[
                ExistenceRelationship(
                    concept="operator",
                    connector=LogicalConnector.OR,
                    strength=1.0,
                    description="Logical operator"
                )
            ],
            not_relationships=[
                ExistenceRelationship(
                    concept="independence",
                    connector=LogicalConnector.NOT,
                    strength=1.0,
                    description="Not independence"
                )
            ],
            spherical_properties={
                "preferred_r": 0.4,
                "growth_pattern": "crystalline"
            }
        )
        
        # EQUIVALENT operator
        equiv_op = PrimitiveDefinition(
            concept="<->",
            existence_level=ExistenceLevel.INFORMATIONAL.value,
            and_relationships=[
                ExistenceRelationship(
                    concept="equivalence",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="Logical equivalence"
                ),
                ExistenceRelationship(
                    concept="if_and_only_if",
                    connector=LogicalConnector.AND,
                    strength=1.0,
                    description="If and only if relationship"
                )
            ],
            or_relationships=[
                ExistenceRelationship(
                    concept="operator",
                    connector=LogicalConnector.OR,
                    strength=1.0,
                    description="Logical operator"
                )
            ],
            not_relationships=[
                ExistenceRelationship(
                    concept="^",
                    connector=LogicalConnector.NOT,
                    strength=1.0,
                    description="Not XOR"
                )
            ],
            spherical_properties={
                "preferred_r": 0.4,
                "growth_pattern": "crystalline"
            }
        )
        
        # Store operators
        self.logical_operators[LogicalConnector.AND] = and_op
        self.logical_operators[LogicalConnector.OR] = or_op
        self.logical_operators[LogicalConnector.NOT] = not_op
        self.logical_operators[LogicalConnector.XOR] = xor_op
        self.logical_operators[LogicalConnector.IMPLIES] = implies_op
        self.logical_operators[LogicalConnector.EQUIVALENT] = equiv_op
    
    async def generate_primitive_definition(self, 
                                          concept: str, 
                                          negations: List[str] = None, 
                                          existence_level_name: str = None) -> PrimitiveDefinition:
        """
        Generate a primitive definition for a concept.
        
        Args:
            concept: Concept name
            negations: List of negation concepts
            existence_level_name: Name of existence level
            
        Returns:
            PrimitiveDefinition: Generated primitive definition
        """
        # Set defaults
        if negations is None:
            negations = []
        
        # Determine existence level
        existence_level = ExistenceLevel.INFORMATIONAL.value
        if existence_level_name:
            level_value = self.type_system.get_level_value(existence_level_name)
            if level_value >= 0:
                existence_level = level_value
        
        # Get existence type
        existence_type = self.type_system.get_type(
            self.type_system.get_level_name(existence_level)
        )
        
        # Create primitive definition
        primitive_def = PrimitiveDefinition(
            concept=concept,
            existence_level=existence_level
        )
        
        # Add AND relationships
        primitive_def.and_relationships.append(
            ExistenceRelationship(
                concept="existence",
                connector=LogicalConnector.AND,
                strength=1.0,
                description="All concepts exist"
            )
        )
        
        primitive_def.and_relationships.append(
            ExistenceRelationship(
                concept="pattern",
                connector=LogicalConnector.AND,
                strength=0.9,
                description="All concepts have patterns"
            )
        )
        
        # Add existence level specific AND relationships
        if existence_type:
            for dep_name in existence_type.dependencies:
                dep_type = self.type_system.get_type(dep_name)
                if dep_type:
                    primitive_def.and_relationships.append(
                        ExistenceRelationship(
                            concept=dep_name,
                            connector=LogicalConnector.AND,
                            strength=0.8,
                            description=f"Depends on {dep_name}"
                        )
                    )
        
        # Add OR relationships
        primitive_def.or_relationships.append(
            ExistenceRelationship(
                concept="void",
                connector=LogicalConnector.OR,
                strength=0.5,
                description="Concepts can be void"
            )
        )
        
        # Add existence level specific OR relationships
        if existence_level > ExistenceLevel.VOID.value:
            # Add lower levels as OR relationships
            for level in range(existence_level):
                level_name = self.type_system.get_level_name(level)
                if level_name != "unknown":
                    primitive_def.or_relationships.append(
                        ExistenceRelationship(
                            concept=level_name,
                            connector=LogicalConnector.OR,
                            strength=0.7 - (level / 10),
                            description=f"Can be {level_name}"
                        )
                    )
        
        # Add NOT relationships
        # Add negations
        for negation in negations:
            primitive_def.not_relationships.append(
                ExistenceRelationship(
                    concept=negation,
                    connector=LogicalConnector.NOT,
                    strength=1.0,
                    description=f"Negation of {concept}"
                )
            )
        
        # Add existence level specific NOT relationships
        if existence_type:
            for neg_name in existence_type.negations:
                neg_type = self.type_system.get_type(neg_name)
                if neg_type:
                    primitive_def.not_relationships.append(
                        ExistenceRelationship(
                            concept=neg_name,
                            connector=LogicalConnector.NOT,
                            strength=0.9,
                            description=f"Negates {neg_name}"
                        )
                    )
        
        # Add IMPLIES relationships
        if existence_level > ExistenceLevel.VOID.value:
            # Higher levels imply lower levels
            for level in range(existence_level):
                level_name = self.type_system.get_level_name(level)
                if level_name != "unknown" and level_name != "void":
                    primitive_def.implies_relationships.append(
                        ExistenceRelationship(
                            concept=level_name,
                            connector=LogicalConnector.IMPLIES,
                            strength=0.8,
                            description=f"Implies {level_name}"
                        )
                    )
        
        # Add spherical properties
        primitive_def.spherical_properties = {
            "preferred_r": min(0.4, 0.2 + (existence_level / 20)),
            "growth_pattern": random.choice(["radial", "spiral", "branching", "crystalline"])
        }
        
        # Store primitive
        self.primitives[concept] = primitive_def
        
        return primitive_def
    
    def get_logical_operator_definition(self, operator: LogicalConnector) -> PrimitiveDefinition:
        """Get the definition of a logical operator"""
        return self.logical_operators.get(operator)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "primitives": {
                concept: prim.to_dict()
                for concept, prim in self.primitives.items()
            },
            "logical_operators": {
                op.name: prim.to_dict()
                for op, prim in self.logical_operators.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExistencePrimitiveEngine':
        """Create from dictionary"""
        engine = cls()
        
        # Load primitives
        for concept, prim_data in data.get("primitives", {}).items():
            engine.primitives[concept] = PrimitiveDefinition.from_dict(prim_data)
        
        # Load logical operators
        for op_name, op_data in data.get("logical_operators", {}).items():
            try:
                op = LogicalConnector[op_name]
                engine.logical_operators[op] = PrimitiveDefinition.from_dict(op_data)
            except KeyError:
                pass
        
        return engine
