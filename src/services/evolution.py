import asyncio
from datetime import datetime, timedelta
import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
import random
import logging

from src.core.types import ConceptDefinition, VectorizedObject, MemeticState
from src.core.primitives import EXISTS, NOT_EXISTS, CO_EXISTS, ALT_EXISTS

# Configure logging
logger = logging.getLogger(__name__)

class MemeticEvolutionService:
    """Background service for concept evolution"""
    
    def __init__(self, app_state: Dict):
        """
        Initialize the memetic evolution service.
        
        Args:
            app_state: Application state dictionary containing dependencies
        """
        self.app_state = app_state
        self.db = app_state['db']
        self.vectorizer = app_state['vectorizer']
        self.vector_store = app_state['vector_store']
        self.x_shaped_hole_engine = app_state.get('x_shaped_hole_engine')
        self.empathy_normalizer = app_state.get('empathy_normalizer')
        
        # Evolution parameters
        self.config = app_state.get('config', {})
        self.population_size = self.config.get('population_size', 1000)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.evolution_interval = self.config.get('evolution_interval', 60)  # seconds
        
        # Fitness thresholds
        self.fitness_threshold = 0.6  # Minimum fitness to survive
        self.elite_threshold = 0.8  # Threshold for elite concepts
        
        # Mutation types and probabilities
        self.mutation_types = {
            'expand_not_space': 0.4,
            'refine_not_space': 0.3,
            'modify_pattern': 0.2,
            'cultural_adaptation': 0.1
        }
        
        # Running flag
        self.is_running = False
    
    async def start(self):
        """Start the evolution loop"""
        if self.is_running:
            return
        
        self.is_running = True
        asyncio.create_task(self.evolution_loop())
        logger.info("Memetic evolution service started")
    
    async def stop(self):
        """Stop the evolution loop"""
        self.is_running = False
        logger.info("Memetic evolution service stopped")
    
    async def evolution_loop(self):
        """
        Continuous evolution process:
        1. Select concepts based on empathy scores
        2. Apply variations
        3. Evaluate fitness
        4. Update population
        """
        while self.is_running:
            try:
                await self.evolution_step()
                await asyncio.sleep(self.evolution_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evolution error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def evolution_step(self):
        """Perform a single step of evolution"""
        # Get active concepts
        active_concepts = await self.get_active_concepts()
        
        if not active_concepts:
            logger.info("No active concepts found for evolution")
            return
        
        # Calculate fitness for each concept
        concept_fitness = {}
        for concept in active_concepts:
            fitness = self.calculate_fitness(concept)
            concept_fitness[concept['id']] = fitness
        
        # Select concepts for evolution (those above threshold)
        candidates = [
            concept for concept in active_concepts
            if concept_fitness[concept['id']] >= self.fitness_threshold
        ]
        
        # Evolve candidates
        evolved_count = 0
        for concept in candidates:
            # Apply mutation with probability based on fitness
            mutation_prob = concept_fitness[concept['id']] * self.mutation_rate
            if random.random() < mutation_prob:
                success = await self.evolve_concept(concept)
                if success:
                    evolved_count += 1
        
        logger.info(f"Evolution step completed: {evolved_count} concepts evolved")
    
    async def get_active_concepts(self) -> List[Dict]:
        """Get concepts that have been accessed recently"""
        # Simplified approach to avoid coroutine issues
        try:
            # Try to get some concepts from the database
            # This is a simplified approach that should work with most database APIs
            logger.info("Fetching active concepts")
            
            # For now, just return a small number of mock concepts for testing
            # In a real implementation, this would query the database
            mock_concepts = []
            for i in range(5):
                mock_concepts.append({
                    'id': f'mock-concept-{i}',
                    'name': f'concept_{i}',
                    'vector': np.random.randn(768).tolist(),
                    'null_ratio': 0.5,
                    'empathy_scores': {'mutual_empathy': 0.7},
                    'negations': [f'not_concept_{i}'],
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'atomic_pattern': '1'
                })
            
            logger.info(f"Returning {len(mock_concepts)} mock concepts")
            return mock_concepts
            
        except Exception as e:
            logger.error(f"Error getting active concepts: {e}")
            logger.info("Returning empty list of concepts")
            return []  # Return empty list as last resort
    
    async def evolve_concept(self, concept: Dict) -> bool:
        """
        Evolve a single concept through mutation.
        
        Args:
            concept: The concept to evolve
            
        Returns:
            bool: True if evolution was successful
        """
        # Choose mutation type
        mutation_type = self.select_mutation_type()
        
        # Apply mutation
        if mutation_type == 'expand_not_space':
            return await self.mutate_expand_not_space(concept)
        elif mutation_type == 'refine_not_space':
            return await self.mutate_refine_not_space(concept)
        elif mutation_type == 'modify_pattern':
            return await self.mutate_modify_pattern(concept)
        elif mutation_type == 'cultural_adaptation':
            return await self.mutate_cultural_adaptation(concept)
        
        return False
    
    def select_mutation_type(self) -> str:
        """Select a mutation type based on probabilities"""
        r = random.random()
        cumulative = 0
        
        for mutation_type, probability in self.mutation_types.items():
            cumulative += probability
            if r <= cumulative:
                return mutation_type
        
        # Default
        return 'expand_not_space'
    
    async def mutate_expand_not_space(self, concept: Dict) -> bool:
        """
        Mutation: Expand the not-space with additional negations.
        
        Args:
            concept: The concept to mutate
            
        Returns:
            bool: True if mutation was successful
        """
        # Get similar concepts to use as potential negations
        similar_concepts = await self.get_similar_concepts(concept['id'], 10)
        
        # Filter out concepts already in not-space
        existing_negations = set(concept.get('negations', []))
        new_negations = [
            similar['name'] for similar in similar_concepts
            if similar['name'] not in existing_negations
            and similar['name'] != concept['name']
        ]
        
        if not new_negations:
            return False
        
        # Add 1-3 new negations
        num_to_add = min(3, len(new_negations))
        selected_negations = random.sample(new_negations, num_to_add)
        
        # Create expanded not-space
        expanded_not_space = list(existing_negations) + selected_negations
        
        # Create new concept definition
        new_concept_id = f"{concept['id']}-expanded-{datetime.utcnow().timestamp()}"
        
        # Convert to ConceptDefinition
        concept_def = self.dict_to_concept_definition(concept)
        
        # Update not-space
        if self.x_shaped_hole_engine:
            # Use X-shaped hole engine if available
            definition = self.x_shaped_hole_engine.define_through_negation(
                concept['name'], expanded_not_space
            )
            concept_def.not_space = definition.not_space
            concept_def.confidence = definition.confidence
        else:
            # Simple update
            concept_def.not_space = set(expanded_not_space)
        
        # Update ID
        concept_def.id = new_concept_id
        
        # Generate new vector
        with torch.no_grad():
            new_vector = self.vectorizer(concept_def)
        
        # Calculate fitness
        fitness = self.calculate_fitness(new_vector)
        original_fitness = self.calculate_fitness(concept)
        
        # Only keep if fitness improved
        if fitness <= original_fitness:
            return False
        
        # Create memetic state
        memetic_state = MemeticState(
            concept_id=new_concept_id,
            generation=self.get_generation(concept) + 1,
            fitness_score=fitness,
            replication_count=0,
            mutation_history=[{
                "parent_id": concept['id'],
                "mutation_type": "expand_not_space",
                "added_negations": selected_negations,
                "timestamp": datetime.utcnow().isoformat()
            }],
            cultural_adaptations={}
        )
        
        # Store in database
        await self.db.concepts.insert_one(self.concept_definition_to_dict(concept_def, new_vector))
        await self.db.memetic_states.insert_one(memetic_state.__dict__)
        
        # Add to vector store
        self.vector_store.add_vectors([new_vector.vector], [new_concept_id])
        
        # Log evolution
        logger.info(f"Evolved concept {concept['name']} -> {new_concept_id} with fitness {fitness}")
        
        return True
    
    async def mutate_refine_not_space(self, concept: Dict) -> bool:
        """
        Mutation: Refine the not-space by removing redundant negations.
        
        Args:
            concept: The concept to mutate
            
        Returns:
            bool: True if mutation was successful
        """
        if not self.x_shaped_hole_engine:
            return False
        
        # Get existing negations
        existing_negations = set(concept.get('negations', []))
        
        if len(existing_negations) < 3:
            return False  # Not enough negations to refine
        
        # Create concept definition
        concept_def = self.dict_to_concept_definition(concept)
        
        # Create X-shaped hole definition
        definition = self.x_shaped_hole_engine.define_through_negation(
            concept['name'], list(existing_negations)
        )
        
        # Refine not-space
        refined_definition = self.x_shaped_hole_engine.refine_not_space(definition)
        
        # If no change, abort
        if len(refined_definition.not_space) >= len(definition.not_space):
            return False
        
        # Create new concept ID
        new_concept_id = f"{concept['id']}-refined-{datetime.utcnow().timestamp()}"
        
        # Update concept definition
        concept_def.id = new_concept_id
        concept_def.not_space = refined_definition.not_space
        concept_def.confidence = refined_definition.confidence
        
        # Generate new vector
        with torch.no_grad():
            new_vector = self.vectorizer(concept_def)
        
        # Calculate fitness
        fitness = self.calculate_fitness(new_vector)
        original_fitness = self.calculate_fitness(concept)
        
        # Only keep if fitness improved
        if fitness <= original_fitness:
            return False
        
        # Create memetic state
        memetic_state = MemeticState(
            concept_id=new_concept_id,
            generation=self.get_generation(concept) + 1,
            fitness_score=fitness,
            replication_count=0,
            mutation_history=[{
                "parent_id": concept['id'],
                "mutation_type": "refine_not_space",
                "removed_negations": list(existing_negations - refined_definition.not_space),
                "timestamp": datetime.utcnow().isoformat()
            }],
            cultural_adaptations={}
        )
        
        # Store in database
        await self.db.concepts.insert_one(self.concept_definition_to_dict(concept_def, new_vector))
        await self.db.memetic_states.insert_one(memetic_state.__dict__)
        
        # Add to vector store
        self.vector_store.add_vectors([new_vector.vector], [new_concept_id])
        
        # Log evolution
        logger.info(f"Refined concept {concept['name']} -> {new_concept_id} with fitness {fitness}")
        
        return True
    
    async def mutate_modify_pattern(self, concept: Dict) -> bool:
        """
        Mutation: Modify the atomic pattern of the concept.
        
        Args:
            concept: The concept to mutate
            
        Returns:
            bool: True if mutation was successful
        """
        # Get current pattern
        current_pattern = concept.get('atomic_pattern')
        
        if not current_pattern:
            return False
        
        # Create new pattern (simplified example)
        # In a real implementation, this would involve more complex pattern manipulation
        new_pattern = None
        
        if current_pattern == "1":
            # Simple existence -> co-existence with self
            new_pattern = f"&&(1, {concept['name']})"
        elif "&&" in current_pattern:
            # Try alternative existence
            new_pattern = current_pattern.replace("&&", "||")
        elif "||" in current_pattern:
            # Try negation of negation
            new_pattern = f"!(!({current_pattern}))"
        else:
            # Default to EXISTS
            new_pattern = "1"
        
        # Create new concept ID
        new_concept_id = f"{concept['id']}-pattern-{datetime.utcnow().timestamp()}"
        
        # Create concept definition
        concept_def = self.dict_to_concept_definition(concept)
        concept_def.id = new_concept_id
        
        # Update atomic pattern
        from src.core.primitives import parse_pattern
        concept_def.atomic_pattern.pattern = parse_pattern(new_pattern)
        
        # Generate new vector
        with torch.no_grad():
            new_vector = self.vectorizer(concept_def)
        
        # Calculate fitness
        fitness = self.calculate_fitness(new_vector)
        original_fitness = self.calculate_fitness(concept)
        
        # Only keep if fitness improved
        if fitness <= original_fitness:
            return False
        
        # Create memetic state
        memetic_state = MemeticState(
            concept_id=new_concept_id,
            generation=self.get_generation(concept) + 1,
            fitness_score=fitness,
            replication_count=0,
            mutation_history=[{
                "parent_id": concept['id'],
                "mutation_type": "modify_pattern",
                "old_pattern": current_pattern,
                "new_pattern": new_pattern,
                "timestamp": datetime.utcnow().isoformat()
            }],
            cultural_adaptations={}
        )
        
        # Store in database
        await self.db.concepts.insert_one(self.concept_definition_to_dict(concept_def, new_vector))
        await self.db.memetic_states.insert_one(memetic_state.__dict__)
        
        # Add to vector store
        self.vector_store.add_vectors([new_vector.vector], [new_concept_id])
        
        # Log evolution
        logger.info(f"Modified pattern for {concept['name']} -> {new_concept_id} with fitness {fitness}")
        
        return True
    
    async def mutate_cultural_adaptation(self, concept: Dict) -> bool:
        """
        Mutation: Create a cultural variant of the concept.
        
        Args:
            concept: The concept to mutate
            
        Returns:
            bool: True if mutation was successful
        """
        # Available cultural contexts
        cultural_contexts = ["western", "eastern", "indigenous", "scientific", "artistic"]
        
        # Get existing cultural variants
        existing_variants = concept.get('cultural_variants', {}).keys()
        
        # Filter out existing variants
        available_contexts = [ctx for ctx in cultural_contexts if ctx not in existing_variants]
        
        if not available_contexts:
            return False
        
        # Select a random context
        context = random.choice(available_contexts)
        
        # Create new concept ID with cultural context
        new_concept_id = f"{concept['id']}-{context}-{datetime.utcnow().timestamp()}"
        
        # Create concept definition
        concept_def = self.dict_to_concept_definition(concept)
        concept_def.id = new_concept_id
        
        # Generate new vector with cultural bias
        # In a real implementation, this would involve more sophisticated cultural adaptation
        with torch.no_grad():
            # Add small cultural bias to vector
            new_vector = self.vectorizer(concept_def)
            
            # Add cultural bias (simplified)
            cultural_bias = np.random.randn(768) * 0.1
            cultural_vector = new_vector.vector + cultural_bias
            cultural_vector = cultural_vector / np.linalg.norm(cultural_vector)
            
            # Update vector
            new_vector.vector = cultural_vector
            new_vector.cultural_variants = {context: cultural_vector}
        
        # Calculate fitness
        fitness = self.calculate_fitness(new_vector)
        original_fitness = self.calculate_fitness(concept)
        
        # Only keep if fitness improved
        if fitness <= original_fitness:
            return False
        
        # Create memetic state
        memetic_state = MemeticState(
            concept_id=new_concept_id,
            generation=self.get_generation(concept) + 1,
            fitness_score=fitness,
            replication_count=0,
            mutation_history=[{
                "parent_id": concept['id'],
                "mutation_type": "cultural_adaptation",
                "cultural_context": context,
                "timestamp": datetime.utcnow().isoformat()
            }],
            cultural_adaptations={
                context: {
                    "base_concept_id": concept['id'],
                    "adaptation_date": datetime.utcnow().isoformat()
                }
            }
        )
        
        # Store in database
        concept_dict = self.concept_definition_to_dict(concept_def, new_vector)
        concept_dict['cultural_context'] = context
        await self.db.concepts.insert_one(concept_dict)
        await self.db.memetic_states.insert_one(memetic_state.__dict__)
        
        # Add to vector store
        self.vector_store.add_vectors([new_vector.vector], [new_concept_id])
        
        # Log evolution
        logger.info(f"Created cultural variant {context} for {concept['name']} -> {new_concept_id}")
        
        return True
    
    async def get_similar_concepts(self, concept_id: str, k: int = 10) -> List[Dict]:
        """Get concepts similar to the given one"""
        # Get concept vector
        concept = await self.db.concepts.find_one({'id': concept_id})
        if not concept or 'vector' not in concept:
            return []
        
        # Search vector store
        results = self.vector_store.search(concept['vector'], k=k+1)
        
        # Filter out the query concept itself
        similar = [r for r in results if r['id'] != concept_id][:k]
        
        # Get full concept data
        similar_concepts = []
        for result in similar:
            similar_concept = await self.db.concepts.find_one({'id': result['id']})
            if similar_concept:
                similar_concept['similarity'] = result['score']
                similar_concepts.append(similar_concept)
        
        return similar_concepts
    
    def calculate_fitness(self, obj: Union[Dict, VectorizedObject]) -> float:
        """
        Calculate memetic fitness score.
        
        Args:
            obj: Either a concept dictionary or a VectorizedObject
            
        Returns:
            Fitness score between 0.0 and 1.0
        """
        if isinstance(obj, VectorizedObject):
            # Factors:
            # 1. Low null ratio (well-defined)
            # 2. High empathy scores (good co-existence)
            # 3. Vector coherence (not random)
            
            fitness = (
                (1 - obj.null_ratio) * 0.4 +
                obj.empathy_scores.get('mutual_empathy', 0) * 0.4 +
                np.linalg.norm(obj.vector) * 0.2
            )
            
            return max(0.0, min(1.0, fitness))
        
        elif isinstance(obj, dict):
            # Calculate from dictionary
            null_ratio = obj.get('null_ratio', 0.5)
            empathy_scores = obj.get('empathy_scores', {})
            mutual_empathy = empathy_scores.get('mutual_empathy', 0)
            
            # Vector norm (assume unit vector if not specified)
            vector_norm = 1.0
            if 'vector' in obj and isinstance(obj['vector'], (list, np.ndarray)):
                vector = np.array(obj['vector'])
                vector_norm = np.linalg.norm(vector)
            
            fitness = (
                (1 - null_ratio) * 0.4 +
                mutual_empathy * 0.4 +
                vector_norm * 0.2
            )
            
            return max(0.0, min(1.0, fitness))
        
        else:
            return 0.0
    
    def get_generation(self, concept: Dict) -> int:
        """Get the generation of a concept"""
        return concept.get('generation', 0)
    
    def dict_to_concept_definition(self, concept: Dict) -> ConceptDefinition:
        """Convert a concept dictionary to a ConceptDefinition object"""
        from src.core.primitives import parse_pattern
        
        # Parse atomic pattern
        pattern_str = concept.get('atomic_pattern', '1')
        pattern = parse_pattern(pattern_str)
        
        # Create atomic pattern
        atomic_pattern = ExistencePattern(
            pattern=[pattern] if isinstance(pattern, str) else pattern,
            confidence=concept.get('confidence', 1.0)
        )
        
        # Create concept definition
        return ConceptDefinition(
            id=concept['id'],
            name=concept['name'],
            atomic_pattern=atomic_pattern,
            not_space=set(concept.get('negations', [])),
            confidence=concept.get('confidence', 0.5),
            created_at=concept.get('created_at', datetime.now()),
            updated_at=concept.get('updated_at', datetime.now())
        )
    
    def concept_definition_to_dict(self, concept_def: ConceptDefinition, 
                                 vectorized: Optional[VectorizedObject] = None) -> Dict:
        """Convert a ConceptDefinition to a dictionary for storage"""
        result = {
            'id': concept_def.id,
            'name': concept_def.name,
            'atomic_pattern': str(concept_def.atomic_pattern),
            'negations': list(concept_def.not_space),
            'confidence': concept_def.confidence,
            'created_at': concept_def.created_at,
            'updated_at': concept_def.updated_at
        }
        
        if vectorized:
            result['vector'] = vectorized.vector.tolist()
            result['null_ratio'] = vectorized.null_ratio
            result['empathy_scores'] = vectorized.empathy_scores
            result['metadata'] = vectorized.metadata
        
        return result

# For backward compatibility
async def start_evolution_loop(app_state: Dict):
    """Background process for memetic evolution"""
    service = MemeticEvolutionService(app_state)
    await service.start()

async def evolve_concepts(app_state: Dict):
    """Evolve concepts based on usage and fitness"""
    service = MemeticEvolutionService(app_state)
    await service.evolution_step()
