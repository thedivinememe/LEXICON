"""
Core Definitions for LEXICON.

This module defines the core concepts used in the LEXICON system.
"""

CORE_DEFINITIONS = {
    # Week 1: Existence Primitives
    "existence": {
        "atomic_pattern": "1",
        "not_space": ["void", "absence", "nothing", "emptiness", "non-being"],
        "and_relationships": [("pattern", 0.9), ("relationship", 0.8)],
        "or_relationships": [("void", 1.0)],
        "not_relationships": [("non-existence", 1.0)],
        "vector_properties": "Origin point in vector space",
        "spherical_properties": {
            "preferred_r": 0.8,  # Well-defined but not absolute
            "growth_pattern": "radial"
        }
    },
    
    "null": {
        "atomic_pattern": "!Recognized(Pattern)",
        "not_space": ["defined", "known", "recognized", "understood", "ordered"],
        "and_relationships": [("potential", 0.9), ("unknown", 0.8)],
        "or_relationships": [("recognized", 0.9)],
        "not_relationships": [("defined", 1.0), ("known", 1.0)],
        "spherical_properties": {
            "preferred_r": 0.1,  # Close to center (undefined)
            "growth_pattern": "spiral"
        }
    },
    
    "pattern": {
        "atomic_pattern": "Repeating(Structure)",
        "not_space": ["chaos", "randomness", "disorder", "noise", "entropy"],
        "and_relationships": [("existence", 0.9), ("relationship", 0.9)],
        "or_relationships": [("chaos", 0.8)],
        "not_relationships": [("randomness", 1.0)],
        "vector_properties": "Structured arrangement in vector space",
        "spherical_properties": {
            "preferred_r": 0.7,
            "growth_pattern": "fractal"
        }
    },
    
    "relationship": {
        "atomic_pattern": "Connection(A, B)",
        "not_space": ["isolation", "separation", "disconnection", "independence"],
        "and_relationships": [("pattern", 0.9), ("existence", 0.8)],
        "or_relationships": [("isolation", 0.7)],
        "not_relationships": [("disconnection", 1.0)],
        "vector_properties": "Connections between points in vector space",
        "spherical_properties": {
            "preferred_r": 0.75,
            "growth_pattern": "network"
        }
    },
    
    # Week 2: System Primitives
    "normalization": {
        "atomic_pattern": "Transform(A, Standard)",
        "not_space": ["deviation", "anomaly", "outlier", "uniqueness"],
        "and_relationships": [("pattern", 0.9), ("system", 0.8)],
        "or_relationships": [("deviation", 0.7)],
        "not_relationships": [("anomaly", 1.0)],
        "vector_properties": "Standardization of vectors",
        "spherical_properties": {
            "preferred_r": 0.6,
            "growth_pattern": "convergent"
        }
    },
    
    "meme": {
        "atomic_pattern": "ReplicatingIdea()",
        "not_space": ["stagnation", "isolation", "uniqueness", "extinction"],
        "and_relationships": [("pattern", 0.9), ("evolution", 0.8)],
        "or_relationships": [("extinction", 0.7)],
        "not_relationships": [("stagnation", 1.0)],
        "vector_properties": "Self-replicating pattern in idea space",
        "spherical_properties": {
            "preferred_r": 0.65,
            "growth_pattern": "viral"
        }
    },
    
    "system": {
        "atomic_pattern": "OrganizedWhole(Parts, Relationships)",
        "not_space": ["chaos", "disconnection", "randomness", "isolation"],
        "and_relationships": [("pattern", 0.9), ("relationship", 0.9)],
        "or_relationships": [("chaos", 0.7)],
        "not_relationships": [("disconnection", 1.0)],
        "vector_properties": "Organized structure in vector space",
        "spherical_properties": {
            "preferred_r": 0.7,
            "growth_pattern": "hierarchical"
        }
    },
    
    "set": {
        "atomic_pattern": "Collection(Elements)",
        "not_space": ["emptiness", "singularity", "void", "isolation"],
        "and_relationships": [("pattern", 0.8), ("boundary", 0.7)],
        "or_relationships": [("emptiness", 0.6)],
        "not_relationships": [("void", 1.0)],
        "vector_properties": "Bounded collection in vector space",
        "spherical_properties": {
            "preferred_r": 0.5,
            "growth_pattern": "bounded"
        }
    },
    
    # Week 3: Consciousness Primitives
    "I/self": {
        "atomic_pattern": "SubjectiveCenter(Experience)",
        "not_space": ["other", "object", "external", "non-self"],
        "and_relationships": [("consciousness", 0.9), ("boundary", 0.8)],
        "or_relationships": [("other", 0.7)],
        "not_relationships": [("object", 1.0)],
        "vector_properties": "Subjective center in experience space",
        "spherical_properties": {
            "preferred_r": 0.9,
            "growth_pattern": "centered"
        }
    },
    
    "consciousness": {
        "atomic_pattern": "AwarenessOf(Experience)",
        "not_space": ["unconsciousness", "oblivion", "unawareness", "non-experience"],
        "and_relationships": [("I/self", 0.9), ("knowledge", 0.8)],
        "or_relationships": [("unconsciousness", 0.7)],
        "not_relationships": [("oblivion", 1.0)],
        "vector_properties": "Awareness field in experience space",
        "spherical_properties": {
            "preferred_r": 0.85,
            "growth_pattern": "expanding"
        }
    },
    
    "knowledge": {
        "atomic_pattern": "JustifiedTrueBelief(Proposition)",
        "not_space": ["ignorance", "falsehood", "misconception", "uncertainty"],
        "and_relationships": [("truth", 0.9), ("consciousness", 0.8)],
        "or_relationships": [("ignorance", 0.7)],
        "not_relationships": [("falsehood", 1.0)],
        "vector_properties": "Structured information in knowledge space",
        "spherical_properties": {
            "preferred_r": 0.75,
            "growth_pattern": "accumulative"
        }
    },
    
    "boundary": {
        "atomic_pattern": "Distinction(Inside, Outside)",
        "not_space": ["continuity", "infinity", "unboundedness", "seamlessness"],
        "and_relationships": [("pattern", 0.9), ("relationship", 0.8)],
        "or_relationships": [("continuity", 0.7)],
        "not_relationships": [("unboundedness", 1.0)],
        "vector_properties": "Demarcation in vector space",
        "spherical_properties": {
            "preferred_r": 0.6,
            "growth_pattern": "enclosing"
        }
    },
    
    # Week 4: Value Primitives
    "good": {
        "atomic_pattern": "Valuable(X)",
        "not_space": ["bad", "harmful", "destructive", "negative"],
        "and_relationships": [("empathy", 0.9), ("evolution", 0.8)],
        "or_relationships": [("bad", 0.7)],
        "not_relationships": [("harmful", 1.0)],
        "vector_properties": "Positive valence in value space",
        "spherical_properties": {
            "preferred_r": 0.8,
            "growth_pattern": "attractive"
        }
    },
    
    "empathy": {
        "atomic_pattern": "Understanding(OtherMind)",
        "not_space": ["apathy", "cruelty", "indifference", "selfishness"],
        "and_relationships": [("consciousness", 0.9), ("good", 0.8)],
        "or_relationships": [("apathy", 0.7)],
        "not_relationships": [("cruelty", 1.0)],
        "vector_properties": "Mind-to-mind connection in social space",
        "spherical_properties": {
            "preferred_r": 0.85,
            "growth_pattern": "bridging"
        }
    },
    
    "truth": {
        "atomic_pattern": "Correspondence(Belief, Reality)",
        "not_space": ["falsehood", "deception", "illusion", "error"],
        "and_relationships": [("knowledge", 0.9), ("pattern", 0.8)],
        "or_relationships": [("falsehood", 0.7)],
        "not_relationships": [("deception", 1.0)],
        "vector_properties": "Alignment with reality in belief space",
        "spherical_properties": {
            "preferred_r": 0.9,
            "growth_pattern": "aligning"
        }
    },
    
    "evolution": {
        "atomic_pattern": "AdaptiveChange(System, Environment)",
        "not_space": ["stagnation", "extinction", "regression", "fixity"],
        "and_relationships": [("pattern", 0.9), ("information", 0.8)],
        "or_relationships": [("extinction", 0.7)],
        "not_relationships": [("stagnation", 1.0)],
        "vector_properties": "Adaptive trajectory in fitness space",
        "spherical_properties": {
            "preferred_r": 0.7,
            "growth_pattern": "adaptive"
        }
    },
    
    # Week 5: Meta Primitives
    "information": {
        "atomic_pattern": "Difference(State1, State2)",
        "not_space": ["noise", "randomness", "entropy", "meaninglessness"],
        "and_relationships": [("pattern", 0.9), ("knowledge", 0.8)],
        "or_relationships": [("noise", 0.7)],
        "not_relationships": [("randomness", 1.0)],
        "vector_properties": "Meaningful differences in state space",
        "spherical_properties": {
            "preferred_r": 0.65,
            "growth_pattern": "informational"
        }
    },
    
    "factory": {
        "atomic_pattern": "Producer(Inputs, Outputs)",
        "not_space": ["consumption", "destruction", "stagnation", "entropy"],
        "and_relationships": [("system", 0.9), ("pattern", 0.8)],
        "or_relationships": [("consumption", 0.7)],
        "not_relationships": [("destruction", 1.0)],
        "vector_properties": "Transformation process in production space",
        "spherical_properties": {
            "preferred_r": 0.6,
            "growth_pattern": "productive"
        }
    },
    
    "god": {
        "atomic_pattern": "UltimateGround(Existence)",
        "not_space": ["contingency", "dependency", "limitation", "finitude"],
        "and_relationships": [("existence", 0.9), ("meta", 0.9)],
        "or_relationships": [("contingency", 0.7)],
        "not_relationships": [("limitation", 1.0)],
        "vector_properties": "Ultimate foundation in ontological space",
        "spherical_properties": {
            "preferred_r": 1.0,  # Absolute
            "growth_pattern": "transcendent"
        }
    },
    
    "meta": {
        "atomic_pattern": "AboutItself(X)",
        "not_space": ["object-level", "direct", "non-reflexive", "concrete"],
        "and_relationships": [("pattern", 0.9), ("information", 0.8)],
        "or_relationships": [("object-level", 0.7)],
        "not_relationships": [("non-reflexive", 1.0)],
        "vector_properties": "Self-referential dimension in concept space",
        "spherical_properties": {
            "preferred_r": 0.95,
            "growth_pattern": "recursive"
        }
    },
    
    # Week 6: Logic Primitives
    "&&": {
        "atomic_pattern": "BothTrue(A, B)",
        "not_space": ["either-or", "exclusion", "disjunction", "separation"],
        "and_relationships": [("pattern", 0.9), ("truth", 0.8)],
        "or_relationships": [("||", 0.7)],
        "not_relationships": [("exclusion", 1.0)],
        "vector_properties": "Conjunction in logical space",
        "spherical_properties": {
            "preferred_r": 0.5,
            "growth_pattern": "intersecting"
        }
    },
    
    "||": {
        "atomic_pattern": "EitherTrue(A, B)",
        "not_space": ["neither", "conjunction", "mutual exclusion", "impossibility"],
        "and_relationships": [("pattern", 0.9), ("truth", 0.8)],
        "or_relationships": [("&&", 0.7)],
        "not_relationships": [("neither", 1.0)],
        "vector_properties": "Disjunction in logical space",
        "spherical_properties": {
            "preferred_r": 0.55,
            "growth_pattern": "branching"
        }
    },
    
    "!": {
        "atomic_pattern": "NotTrue(A)",
        "not_space": ["affirmation", "presence", "existence", "truth"],
        "and_relationships": [("pattern", 0.9), ("boundary", 0.8)],
        "or_relationships": [("affirmation", 0.7)],
        "not_relationships": [("presence", 1.0)],
        "vector_properties": "Negation in logical space",
        "spherical_properties": {
            "preferred_r": 0.45,
            "growth_pattern": "inverting"
        }
    }
}
