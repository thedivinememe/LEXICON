"""
Existence Types for the Spherical Universe.
Defines types of existence and their properties.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio

class ExistenceLevel(Enum):
    """Levels of existence"""
    VOID = 0
    POTENTIAL = 1
    PHYSICAL = 2
    ENERGETIC = 3
    INFORMATIONAL = 4
    BIOLOGICAL = 5
    MENTAL = 6
    LINGUISTIC = 7
    SOCIAL = 8
    CONSCIOUS = 9
    TRANSCENDENT = 10

@dataclass
class ExistenceType:
    """
    Type of existence with properties.
    
    Attributes:
        name: Name of the existence type
        level: Level of existence
        existence_ratio: Ratio of existence to non-existence (0 to 1)
        persistence: Temporal stability (0 to 1)
        complexity: Structural complexity (0 to 1)
        dependencies: Set of types this type depends on
        negations: List of types this type negates
        is_material: Whether this type has material form
        is_temporal: Whether this type exists in time
        is_spatial: Whether this type exists in space
        is_relational: Whether this type involves relationships
        is_self_aware: Whether this type has self-awareness
    """
    name: str
    level: ExistenceLevel
    existence_ratio: float
    persistence: float
    complexity: float
    dependencies: Set[str] = field(default_factory=set)
    negations: List[str] = field(default_factory=list)
    is_material: bool = False
    is_temporal: bool = True
    is_spatial: bool = False
    is_relational: bool = False
    is_self_aware: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "level": self.level.value,
            "existence_ratio": self.existence_ratio,
            "persistence": self.persistence,
            "complexity": self.complexity,
            "dependencies": list(self.dependencies),
            "negations": self.negations,
            "is_material": self.is_material,
            "is_temporal": self.is_temporal,
            "is_spatial": self.is_spatial,
            "is_relational": self.is_relational,
            "is_self_aware": self.is_self_aware
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExistenceType':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            level=ExistenceLevel(data["level"]),
            existence_ratio=data["existence_ratio"],
            persistence=data["persistence"],
            complexity=data["complexity"],
            dependencies=set(data.get("dependencies", [])),
            negations=data.get("negations", []),
            is_material=data.get("is_material", False),
            is_temporal=data.get("is_temporal", True),
            is_spatial=data.get("is_spatial", False),
            is_relational=data.get("is_relational", False),
            is_self_aware=data.get("is_self_aware", False)
        )

class ExistenceTypeRegistry:
    """Registry of existence types"""
    
    def __init__(self):
        self.types = {}  # name -> ExistenceType
    
    async def register_type(self, existence_type: ExistenceType) -> None:
        """Register an existence type"""
        self.types[existence_type.name] = existence_type
    
    def get_type(self, name: str) -> Optional[ExistenceType]:
        """Get an existence type by name"""
        return self.types.get(name)
    
    def get_types_by_level(self, level: ExistenceLevel) -> List[ExistenceType]:
        """Get all types at a specific level"""
        return [t for t in self.types.values() if t.level == level]
    
    def get_dependencies(self, type_name: str) -> List[ExistenceType]:
        """Get all types that a type depends on"""
        type_obj = self.get_type(type_name)
        if not type_obj:
            return []
        
        return [self.get_type(dep) for dep in type_obj.dependencies if self.get_type(dep)]
    
    def get_negations(self, type_name: str) -> List[ExistenceType]:
        """Get all types that a type negates"""
        type_obj = self.get_type(type_name)
        if not type_obj:
            return []
        
        return [self.get_type(neg) for neg in type_obj.negations if self.get_type(neg)]
    
    def is_compatible(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible"""
        type1_obj = self.get_type(type1)
        type2_obj = self.get_type(type2)
        
        if not type1_obj or not type2_obj:
            return False
        
        # Check if either type negates the other
        if type2 in type1_obj.negations or type1 in type2_obj.negations:
            return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "types": {
                name: type_obj.to_dict()
                for name, type_obj in self.types.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExistenceTypeRegistry':
        """Create from dictionary"""
        registry = cls()
        
        for name, type_data in data.get("types", {}).items():
            registry.types[name] = ExistenceType.from_dict(type_data)
        
        return registry

class ExistenceTypeSystem:
    """System for managing existence types"""
    
    def __init__(self):
        self.registry = ExistenceTypeRegistry()
        self._initialize_types()
    
    def _initialize_types(self) -> None:
        """Initialize built-in existence types"""
        # VOID
        void_type = ExistenceType(
            name="void",
            level=ExistenceLevel.VOID,
            existence_ratio=0.0,
            persistence=0.0,
            complexity=0.0,
            dependencies=set(),
            negations=["physical", "energetic", "informational"],
            is_material=False,
            is_temporal=False,
            is_spatial=False,
            is_relational=False,
            is_self_aware=False
        )
        
        # POTENTIAL
        potential_type = ExistenceType(
            name="potential",
            level=ExistenceLevel.POTENTIAL,
            existence_ratio=0.2,
            persistence=0.3,
            complexity=0.2,
            dependencies={"void"},
            negations=["physical", "deterministic"],
            is_material=False,
            is_temporal=True,
            is_spatial=False,
            is_relational=False,
            is_self_aware=False
        )
        
        # PHYSICAL
        physical_type = ExistenceType(
            name="physical",
            level=ExistenceLevel.PHYSICAL,
            existence_ratio=0.8,
            persistence=0.9,
            complexity=0.5,
            dependencies={"potential"},
            negations=["void", "energetic", "mental"],
            is_material=True,
            is_temporal=True,
            is_spatial=True,
            is_relational=False,
            is_self_aware=False
        )
        
        # ENERGETIC
        energetic_type = ExistenceType(
            name="energetic",
            level=ExistenceLevel.ENERGETIC,
            existence_ratio=0.7,
            persistence=0.6,
            complexity=0.6,
            dependencies={"physical", "potential"},
            negations=["void", "physical"],
            is_material=False,
            is_temporal=True,
            is_spatial=True,
            is_relational=True,
            is_self_aware=False
        )
        
        # INFORMATIONAL
        informational_type = ExistenceType(
            name="informational",
            level=ExistenceLevel.INFORMATIONAL,
            existence_ratio=0.6,
            persistence=0.7,
            complexity=0.7,
            dependencies={"energetic", "physical"},
            negations=["void", "random"],
            is_material=False,
            is_temporal=True,
            is_spatial=False,
            is_relational=True,
            is_self_aware=False
        )
        
        # BIOLOGICAL
        biological_type = ExistenceType(
            name="biological",
            level=ExistenceLevel.BIOLOGICAL,
            existence_ratio=0.75,
            persistence=0.5,
            complexity=0.8,
            dependencies={"physical", "energetic", "informational"},
            negations=["void", "mechanical"],
            is_material=True,
            is_temporal=True,
            is_spatial=True,
            is_relational=True,
            is_self_aware=False
        )
        
        # MENTAL
        mental_type = ExistenceType(
            name="mental",
            level=ExistenceLevel.MENTAL,
            existence_ratio=0.65,
            persistence=0.4,
            complexity=0.85,
            dependencies={"biological", "informational"},
            negations=["void", "physical"],
            is_material=False,
            is_temporal=True,
            is_spatial=False,
            is_relational=True,
            is_self_aware=False
        )
        
        # LINGUISTIC
        linguistic_type = ExistenceType(
            name="linguistic",
            level=ExistenceLevel.LINGUISTIC,
            existence_ratio=0.6,
            persistence=0.8,
            complexity=0.9,
            dependencies={"mental", "informational"},
            negations=["void", "physical"],
            is_material=False,
            is_temporal=True,
            is_spatial=False,
            is_relational=True,
            is_self_aware=False
        )
        
        # SOCIAL
        social_type = ExistenceType(
            name="social",
            level=ExistenceLevel.SOCIAL,
            existence_ratio=0.7,
            persistence=0.6,
            complexity=0.95,
            dependencies={"mental", "linguistic", "biological"},
            negations=["void", "isolated"],
            is_material=False,
            is_temporal=True,
            is_spatial=True,
            is_relational=True,
            is_self_aware=False
        )
        
        # CONSCIOUS
        conscious_type = ExistenceType(
            name="conscious",
            level=ExistenceLevel.CONSCIOUS,
            existence_ratio=0.8,
            persistence=0.4,
            complexity=1.0,
            dependencies={"mental", "social", "biological"},
            negations=["void", "unconscious"],
            is_material=False,
            is_temporal=True,
            is_spatial=False,
            is_relational=True,
            is_self_aware=True
        )
        
        # TRANSCENDENT
        transcendent_type = ExistenceType(
            name="transcendent",
            level=ExistenceLevel.TRANSCENDENT,
            existence_ratio=0.9,
            persistence=1.0,
            complexity=1.0,
            dependencies={"conscious", "informational"},
            negations=["void", "physical", "limited"],
            is_material=False,
            is_temporal=False,
            is_spatial=False,
            is_relational=True,
            is_self_aware=True
        )
        
        # Register types
        self.registry.types["void"] = void_type
        self.registry.types["potential"] = potential_type
        self.registry.types["physical"] = physical_type
        self.registry.types["energetic"] = energetic_type
        self.registry.types["informational"] = informational_type
        self.registry.types["biological"] = biological_type
        self.registry.types["mental"] = mental_type
        self.registry.types["linguistic"] = linguistic_type
        self.registry.types["social"] = social_type
        self.registry.types["conscious"] = conscious_type
        self.registry.types["transcendent"] = transcendent_type
    
    def get_type(self, name: str) -> Optional[ExistenceType]:
        """Get an existence type by name"""
        return self.registry.get_type(name)
    
    def get_dependencies(self, name: str) -> List[ExistenceType]:
        """Get dependencies for a type"""
        return self.registry.get_dependencies(name)
    
    def is_compatible(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible"""
        return self.registry.is_compatible(type1, type2)
    
    def get_level_name(self, level: int) -> str:
        """Get the name of an existence level"""
        try:
            return ExistenceLevel(level).name.lower()
        except ValueError:
            return "unknown"
    
    def get_level_value(self, name: str) -> int:
        """Get the value of an existence level by name"""
        try:
            return ExistenceLevel[name.upper()].value
        except KeyError:
            return -1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return self.registry.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExistenceTypeSystem':
        """Create from dictionary"""
        system = cls()
        system.registry = ExistenceTypeRegistry.from_dict(data)
        return system
