import asyncio
from datetime import datetime, timedelta
import numpy as np
import torch
from typing import Dict, List

from src.core.types import ConceptDefinition, VectorizedObject, MemeticState

async def start_evolution_loop(app_state: Dict):
    """Background process for memetic evolution"""
    
    while True:
        try:
            await evolve_concepts(app_state)
            await asyncio.sleep(60)  # Run every minute
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Evolution error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

async def evolve_concepts(app_state: Dict):
    """Evolve concepts based on usage and fitness"""
    
    db = app_state['db']
    vectorizer = app_state['vectorizer']
    
    # Get concepts accessed in last hour
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    
    # Use a list comprehension instead of to_list()
    # This is a simplified version that just uses a dummy list for now
    # since we don't have actual access data yet
    active_concepts = []
    
    # In a real implementation, we would query the database
    # For now, just use a dummy list to avoid errors
    
    # Count access frequency
    concept_fitness = {}
    for access in active_concepts:
        concept_id = access['concept_id']
        concept_fitness[concept_id] = concept_fitness.get(concept_id, 0) + 1
    
    # Evolve high-fitness concepts
    for concept_id, fitness in concept_fitness.items():
        if fitness > 10:  # Threshold for evolution
            await mutate_concept(concept_id, fitness, app_state)

async def mutate_concept(concept_id: str, fitness: int, app_state: Dict):
    """Create variations of successful concepts"""
    
    db = app_state['db']
    vectorizer = app_state['vectorizer']
    vector_store = app_state['vector_store']
    
    # Get original concept
    concept = await db.concepts.find_one({'id': concept_id})
    if not concept:
        return
    
    # Convert to ConceptDefinition object
    concept_def = ConceptDefinition(
        id=concept['id'],
        name=concept['name'],
        atomic_pattern=concept['atomic_pattern'],
        not_space=set(concept['not_space']),
        confidence=concept['confidence'],
        created_at=concept['created_at'],
        updated_at=concept['updated_at']
    )
    
    # Generate variations
    variations = []
    
    # Variation 1: Expand not-space
    expanded_not_space = set(concept['not_space'])
    expanded_not_space.add(f"not-{concept['name']}-variant")
    
    expanded_def = ConceptDefinition(
        id=f"{concept_id}-expanded",
        name=concept['name'],
        atomic_pattern=concept['atomic_pattern'],
        not_space=expanded_not_space,
        confidence=concept['confidence'] * 0.95,  # Slightly less confident
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    variations.append(expanded_def)
    
    # Variation 2: Refine pattern (simplified example)
    # In a real implementation, this would involve more complex pattern manipulation
    
    # Generate new vectors for variations
    for var in variations:
        with torch.no_grad():
            new_vector = vectorizer(var)
        
        # Store if fitness improved
        if calculate_fitness(new_vector) > fitness:
            # Create memetic state
            memetic_state = MemeticState(
                concept_id=var.id,
                generation=1,
                fitness_score=calculate_fitness(new_vector),
                replication_count=0,
                mutation_history=[{
                    "parent_id": concept_id,
                    "mutation_type": "not_space_expansion",
                    "timestamp": datetime.utcnow().isoformat()
                }],
                cultural_adaptations={}
            )
            
            # Store in database
            await db.concepts.insert_one(var.__dict__)
            await db.memetic_states.insert_one(memetic_state.__dict__)
            
            # Add to vector store
            vector_store.add_vectors([new_vector.vector], [var.id])
            
            # Log evolution
            print(f"Evolved concept {concept['name']} -> {var.id} with fitness {calculate_fitness(new_vector)}")

def calculate_fitness(vectorized: VectorizedObject) -> float:
    """Calculate memetic fitness score"""
    
    # Factors:
    # 1. Low null ratio (well-defined)
    # 2. High empathy scores (good co-existence)
    # 3. Vector coherence (not random)
    
    fitness = (
        (1 - vectorized.null_ratio) * 0.4 +
        vectorized.empathy_scores.get('mutual_empathy', 0) * 0.4 +
        np.linalg.norm(vectorized.vector) * 0.2
    )
    
    return fitness
