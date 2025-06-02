"""
Test data for LEXICON.
Provides standardized test data for unit and integration tests.
"""

from typing import List, Tuple, Dict, Set

# Core philosophical concepts to test
PHILOSOPHICAL_CONCEPTS = [
    ("existence", ["non-existence", "void", "absence"]),
    ("something", ["nothing", "null", "void"]),
    ("knowledge", ["ignorance", "uncertainty", "confusion"]),
    ("empathy", ["apathy", "indifference", "cruelty"]),
    ("self", ["other", "external", "foreign"])
]

# Hierarchical concepts (S_Conscious ⊆ S_Life ⊆ S_Exist)
HIERARCHICAL_CONCEPTS = {
    "existence_level": [
        ("rock", ["void", "nothing", "absence"]),
        ("energy", ["void", "stillness", "nothingness"])
    ],
    "life_level": [
        ("organism", ["mineral", "artificial", "dead"]),
        ("growth", ["decay", "stasis", "death"])
    ],
    "consciousness_level": [
        ("awareness", ["unconsciousness", "ignorance", "oblivion"]),
        ("thought", ["instinct", "reflex", "automation"])
    ]
}

# Ethical concepts for Golden Loop testing
ETHICAL_CONCEPTS = [
    ("cooperation", ["competition", "conflict", "selfishness"]),
    ("golden_rule", ["exploitation", "harm", "cruelty"]),
    ("mutual_benefit", ["zero_sum", "parasitism", "predation"])
]

# Opposing concept pairs for boundary testing
OPPOSING_CONCEPTS = [
    ("light", "darkness"),
    ("hot", "cold"),
    ("wet", "dry"),
    ("hard", "soft"),
    ("fast", "slow"),
    ("large", "small"),
    ("complex", "simple"),
    ("order", "chaos"),
    ("creation", "destruction"),
    ("unity", "division")
]

# Concept clusters for empathy testing
CONCEPT_CLUSTERS = {
    "cooperative_cluster": [
        "cooperation", "sharing", "altruism", "teamwork", "community"
    ],
    "competitive_cluster": [
        "competition", "rivalry", "contest", "struggle", "conflict"
    ],
    "abstract_cluster": [
        "idea", "concept", "theory", "hypothesis", "principle"
    ],
    "concrete_cluster": [
        "object", "material", "substance", "matter", "element"
    ]
}

# Gradient concepts for null ratio testing
NULL_RATIO_GRADIENT = [
    # (concept, expected_null_ratio, negations)
    ("pure_existence", 0.0, ["non-existence", "void", "nothingness", "absence", "emptiness"]),
    ("partial_existence", 0.3, ["non-existence", "void"]),
    ("ambiguous_existence", 0.5, ["partial_non_existence"]),
    ("partial_non_existence", 0.7, ["pure_existence"]),
    ("pure_non_existence", 1.0, [])
]

# Cultural variants for testing cultural adaptation
CULTURAL_VARIANTS = {
    "freedom": {
        "western": ["constraint", "imprisonment", "control"],
        "eastern": ["attachment", "desire", "ego"],
        "indigenous": ["separation", "isolation", "disconnection"]
    },
    "success": {
        "western": ["failure", "poverty", "mediocrity"],
        "eastern": ["imbalance", "attachment", "excess"],
        "indigenous": ["disharmony", "waste", "selfishness"]
    }
}

# Test patterns for pattern reduction
TEST_PATTERNS = [
    # (pattern_string, expected_reduction, expected_ratio)
    ("1", "1", 1.0),
    ("!1", "!1", 0.0),
    ("&&(1, 1)", "1", 1.0),
    ("&&(1, !1)", "!1", 0.5),
    ("||(1, !1)", "1", 0.5),
    ("!(!1)", "1", 1.0),
    ("&&(||(1, !1), 1)", "1", 0.75),
    ("!(!(!1))", "!1", 0.0)
]

# Complex concept definitions for integration testing
COMPLEX_DEFINITIONS = [
    {
        "concept": "cat",
        "negations": ["dog", "bird", "fish", "rock", "plant", "building", "abstract_idea"],
        "expected_null_ratio": 0.2,
        "expected_empathy": {
            "dog": 0.4,  # Some empathy with other animals
            "rock": 0.1  # Very little empathy with inanimate objects
        }
    },
    {
        "concept": "democracy",
        "negations": ["dictatorship", "monarchy", "anarchy", "totalitarianism", "oligarchy"],
        "expected_null_ratio": 0.3,
        "expected_empathy": {
            "republic": 0.8,  # High empathy with similar systems
            "dictatorship": 0.2  # Low empathy with opposites
        }
    }
]

# Test vectors for vector operations
def get_test_vectors() -> Dict[str, List[float]]:
    """Generate test vectors for various concepts"""
    import numpy as np
    
    # Use fixed seed for reproducibility
    np.random.seed(42)
    
    # Generate unit vectors with controlled similarities
    vectors = {}
    
    # Base vectors for different concept clusters
    base_cooperative = np.random.randn(768)
    base_cooperative = base_cooperative / np.linalg.norm(base_cooperative)
    
    base_competitive = np.random.randn(768)
    base_competitive = base_competitive / np.linalg.norm(base_competitive)
    
    # Make them somewhat orthogonal
    base_competitive = base_competitive - 0.1 * base_cooperative
    base_competitive = base_competitive / np.linalg.norm(base_competitive)
    
    # Generate vectors for cooperative concepts (similar to each other)
    for concept in CONCEPT_CLUSTERS["cooperative_cluster"]:
        # Add small random variations to base vector
        noise = np.random.randn(768) * 0.1
        vector = base_cooperative + noise
        vector = vector / np.linalg.norm(vector)
        vectors[concept] = vector.tolist()
    
    # Generate vectors for competitive concepts (similar to each other)
    for concept in CONCEPT_CLUSTERS["competitive_cluster"]:
        # Add small random variations to base vector
        noise = np.random.randn(768) * 0.1
        vector = base_competitive + noise
        vector = vector / np.linalg.norm(vector)
        vectors[concept] = vector.tolist()
    
    # Generate vectors for other concepts
    for concept_pair in PHILOSOPHICAL_CONCEPTS:
        concept = concept_pair[0]
        if concept not in vectors:
            vector = np.random.randn(768)
            vector = vector / np.linalg.norm(vector)
            vectors[concept] = vector.tolist()
    
    return vectors

# Boundary test cases
BOUNDARY_TEST_CASES = [
    {
        "concept": "fruit",
        "clear_members": ["apple", "banana", "orange"],
        "clear_non_members": ["car", "computer", "book"],
        "boundary_cases": ["tomato", "avocado", "coconut"],
        "negations": ["vegetable", "meat", "dairy", "grain"]
    },
    {
        "concept": "vehicle",
        "clear_members": ["car", "truck", "motorcycle"],
        "clear_non_members": ["house", "tree", "mountain"],
        "boundary_cases": ["skateboard", "horse", "elevator"],
        "negations": ["building", "furniture", "clothing", "food"]
    }
]

# Evolution test data
EVOLUTION_TEST_DATA = {
    "initial_concepts": [
        ("basic_cooperation", ["competition", "conflict"]),
        ("basic_altruism", ["selfishness", "greed"])
    ],
    "expected_evolved_concepts": [
        "advanced_cooperation",
        "mutual_aid",
        "collective_benefit"
    ],
    "generations": 5,
    "expected_fitness_improvement": 0.2  # Minimum expected improvement
}

# Get all test concepts
def get_all_test_concepts() -> List[Tuple[str, List[str]]]:
    """Return all test concepts with their negations"""
    all_concepts = []
    
    # Add philosophical concepts
    all_concepts.extend(PHILOSOPHICAL_CONCEPTS)
    
    # Add hierarchical concepts
    for level, concepts in HIERARCHICAL_CONCEPTS.items():
        all_concepts.extend(concepts)
    
    # Add ethical concepts
    all_concepts.extend(ETHICAL_CONCEPTS)
    
    # Add opposing concepts
    for concept, opposite in OPPOSING_CONCEPTS:
        all_concepts.append((concept, [opposite]))
        all_concepts.append((opposite, [concept]))
    
    # Add null ratio gradient concepts
    for concept, _, negations in NULL_RATIO_GRADIENT:
        all_concepts.append((concept, negations))
    
    # Add cultural variant concepts
    for concept, variants in CULTURAL_VARIANTS.items():
        for culture, negations in variants.items():
            all_concepts.append((f"{concept}_{culture}", negations))
    
    # Add complex definition concepts
    for definition in COMPLEX_DEFINITIONS:
        all_concepts.append((definition["concept"], definition["negations"]))
    
    # Add boundary test concepts
    for test_case in BOUNDARY_TEST_CASES:
        all_concepts.append((test_case["concept"], test_case["negations"]))
    
    return all_concepts
