"""
Narrative Concepts for LEXICON.

This module defines narrative-related concepts used in the LEXICON system.
"""

# Narrative Concepts
NARRATIVE_CONCEPTS = {
    "heroes_journey": {
        "atomic_pattern": "CyclicalTransformation(Ordinary, Adventure, Return)",
        "not_space": ["stasis", "linearity", "monotony", "unchanging", "isolation"],
        "and_relationships": [("story", 0.95), ("change", 0.9), ("growth", 0.85)],
        "or_relationships": [("story_circle", 0.9)],
        "not_relationships": [("stasis", 1.0)],
        "vector_properties": "Transformative cycle in narrative space",
        "spherical_properties": {
            "preferred_r": 0.8,
            "growth_pattern": "cyclical"
        }
    },
    
    "story_circle": {
        "atomic_pattern": "EightStageLoop(Comfort, Desire, Unfamiliar, Adaptation, Attainment, Price, Return, Change)",
        "not_space": ["linearity", "stasis", "fragmentation", "randomness", "incompleteness"],
        "and_relationships": [("heroes_journey", 0.9), ("loop", 0.85), ("story", 0.95)],
        "or_relationships": [("heroes_journey", 0.9)],
        "not_relationships": [("linearity", 1.0)],
        "vector_properties": "Eight-stage cycle in narrative space",
        "spherical_properties": {
            "preferred_r": 0.75,
            "growth_pattern": "octagonal"
        }
    },
    
    "narrative_loop": {
        "atomic_pattern": "RecursiveStory(Self, Experience, Meaning)",
        "not_space": ["linearity", "endpoint", "fragmentation", "meaninglessness", "randomness"],
        "and_relationships": [("loop", 0.95), ("story", 0.9), ("consciousness", 0.85)],
        "or_relationships": [("linearity", 0.6)],
        "not_relationships": [("endpoint", 1.0)],
        "vector_properties": "Self-referential cycle in narrative space",
        "spherical_properties": {
            "preferred_r": 0.85,
            "growth_pattern": "recursive"
        }
    },
    
    "self_narrative": {
        "atomic_pattern": "IdentityStory(Self, Experience, Coherence)",
        "not_space": ["fragmentation", "incoherence", "anonymity", "objectivity", "detachment"],
        "and_relationships": [("I/self", 0.95), ("story", 0.9), ("identity", 0.85)],
        "or_relationships": [("fragmentation", 0.6)],
        "not_relationships": [("anonymity", 1.0)],
        "vector_properties": "Identity construction in narrative space",
        "spherical_properties": {
            "preferred_r": 0.9,
            "growth_pattern": "self-organizing"
        }
    },
    
    "cognitive_narratology": {
        "atomic_pattern": "MindStoryStructure(Cognition, Narrative, Meaning)",
        "not_space": ["randomness", "meaninglessness", "objectivity", "mechanism", "determinism"],
        "and_relationships": [("mind", 0.95), ("story", 0.9), ("consciousness", 0.85)],
        "or_relationships": [("mechanism", 0.6)],
        "not_relationships": [("randomness", 1.0)],
        "vector_properties": "Cognitive story-making in mental space",
        "spherical_properties": {
            "preferred_r": 0.8,
            "growth_pattern": "cognitive"
        }
    },
    
    "emergent_complexity": {
        "atomic_pattern": "SimpleToComplex(Rules, Interactions, Emergence)",
        "not_space": ["reduction", "simplicity", "linearity", "predictability", "mechanism"],
        "and_relationships": [("pattern", 0.9), ("system", 0.85), ("evolution", 0.8)],
        "or_relationships": [("reduction", 0.6)],
        "not_relationships": [("simplicity", 1.0)],
        "vector_properties": "Complexity emergence in system space",
        "spherical_properties": {
            "preferred_r": 0.75,
            "growth_pattern": "emergent"
        }
    },
    
    "feedback_loop": {
        "atomic_pattern": "OutputToInput(System, Environment, Adaptation)",
        "not_space": ["linearity", "isolation", "one-way", "stasis", "endpoint"],
        "and_relationships": [("loop", 0.95), ("system", 0.9), ("change", 0.85)],
        "or_relationships": [("linearity", 0.6)],
        "not_relationships": [("one-way", 1.0)],
        "vector_properties": "Circular causality in system space",
        "spherical_properties": {
            "preferred_r": 0.7,
            "growth_pattern": "feedback"
        }
    },
    
    "metaphor": {
        "atomic_pattern": "UnderstandingVia(Source, Target, Mapping)",
        "not_space": ["literalness", "directness", "exactness", "precision", "denotation"],
        "and_relationships": [("pattern", 0.9), ("relationship", 0.85), ("meaning", 0.8)],
        "or_relationships": [("literalness", 0.6)],
        "not_relationships": [("exactness", 1.0)],
        "vector_properties": "Conceptual mapping in meaning space",
        "spherical_properties": {
            "preferred_r": 0.8,
            "growth_pattern": "connective"
        }
    },
    
    "strange_loop": {
        "atomic_pattern": "SelfReferentialHierarchy(Levels, Paradox)",
        "not_space": ["linearity", "hierarchy", "simplicity", "non-recursion", "endpoint"],
        "and_relationships": [("loop", 0.95), ("meta", 0.9), ("consciousness", 0.85)],
        "or_relationships": [("hierarchy", 0.7)],
        "not_relationships": [("linearity", 1.0)],
        "vector_properties": "Paradoxical self-reference in conceptual space",
        "spherical_properties": {
            "preferred_r": 0.9,
            "growth_pattern": "tangled"
        }
    }
}
