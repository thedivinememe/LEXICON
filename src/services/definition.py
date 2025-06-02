from typing import List, Dict, Optional
import uuid
from datetime import datetime
import torch

from src.core.types import ConceptDefinition, VectorizedObject
from src.core.x_shaped_hole import XShapedHoleEngine
from src.core.reducer import PrimitiveReducer

class DefinitionService:
    """Service for creating and managing concept definitions"""
    
    def __init__(self, app_state: Dict):
        self.db = app_state['db']
        self.cache = app_state['cache']
        self.vectorizer = app_state['vectorizer']
        self.vector_store = app_state['vector_store']
        self.x_hole_engine = XShapedHoleEngine()
        self.reducer = PrimitiveReducer()
        self.app_state = app_state
    
    async def define_concept(self, 
                           concept: str, 
                           negations: List[str],
                           user_id: Optional[str] = None) -> VectorizedObject:
        """Define a concept through negation and generate its vector"""
        
        # Check cache first
        cache_key = f"concept:{concept}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Create definition using X-shaped hole
        definition = self.x_hole_engine.define_through_negation(
            concept=concept,
            user_negations=negations
        )
        
        # Reduce to atomic pattern
        atomic_pattern = self.reducer.reduce_to_primitives(
            concept=concept,
            not_space=set(negations)
        )
        
        # Create concept definition
        concept_def = ConceptDefinition(
            id=str(uuid.uuid4()),
            name=concept,
            atomic_pattern=atomic_pattern,
            not_space=set(negations),
            confidence=definition.confidence,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Generate vectorized object
        with torch.no_grad():
            vectorized = self.vectorizer(concept_def)
        
        # Store in database
        await self.db.concepts.insert_one(concept_def.__dict__)
        
        # Store vector in FAISS
        self.vector_store.add_vectors(
            vectors=[vectorized.vector],
            ids=[vectorized.concept_id]
        )
        
        # Cache result
        await self.cache.set(cache_key, vectorized, expire=3600)
        
        # Broadcast update if real-time enabled
        if self.app_state.get('websocket_manager'):
            await self.app_state['websocket_manager'].broadcast({
                'type': 'concept_defined',
                'data': {
                    'concept': concept,
                    'vector': vectorized.vector.tolist(),
                    'null_ratio': vectorized.null_ratio
                }
            })
        
        return vectorized
    
    async def get_similar_concepts(self, 
                                 concept_id: str, 
                                 k: int = 10) -> List[Dict]:
        """Find similar concepts using vector similarity"""
        
        # Get concept vector
        concept = await self.db.concepts.find_one({"id": concept_id})
        if not concept:
            raise ValueError(f"Concept {concept_id} not found")
        
        # Search similar vectors
        distances, indices = self.vector_store.search(
            query_vector=concept['vector'],
            k=k
        )
        
        # Get concept details
        similar_concepts = []
        for dist, idx in zip(distances, indices):
            similar_id = self.vector_store.get_id(idx)
            similar_concept = await self.db.concepts.find_one({"id": similar_id})
            
            if similar_concept:
                similar_concepts.append({
                    'concept': similar_concept['name'],
                    'similarity': 1 - dist,  # Convert distance to similarity
                    'null_ratio': similar_concept.get('null_ratio', 0),
                    'shared_not_space': len(
                        set(concept['not_space']) & 
                        set(similar_concept['not_space'])
                    )
                })
        
        return similar_concepts
