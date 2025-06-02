"""
GraphQL schema for LEXICON.
"""

import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import List, Optional
import asyncio

from src.core.types import ConceptDefinition, VectorizedObject
from src.services.definition import DefinitionService
from src.api.dependencies import get_app_state

# GraphQL Types
@strawberry.type
class Concept:
    """GraphQL representation of a concept"""
    id: str
    name: str
    confidence: float
    null_ratio: Optional[float] = None
    vector_preview: Optional[List[float]] = None
    
    @strawberry.field
    async def similar_concepts(self, k: int = 10) -> List["SimilarConcept"]:
        """Get similar concepts"""
        app_state = await get_app_state()
        service = DefinitionService(app_state)
        
        similar = await service.get_similar_concepts(self.id, k)
        return [
            SimilarConcept(
                concept=concept["concept"],
                similarity=concept["similarity"],
                null_ratio=concept.get("null_ratio", 0),
                shared_not_space=concept.get("shared_not_space", 0)
            )
            for concept in similar
        ]

@strawberry.type
class SimilarConcept:
    """GraphQL representation of a similar concept"""
    concept: str
    similarity: float
    null_ratio: float
    shared_not_space: int

@strawberry.type
class EmpathyScores:
    """Empathy scores for a concept"""
    self_empathy: float
    other_empathy: float
    mutual_empathy: float

@strawberry.type
class ConceptDefinitionResult:
    """Result of defining a concept"""
    concept_id: str
    concept_name: str
    confidence: float
    null_ratio: float
    empathy_scores: Optional[EmpathyScores] = None
    vector_preview: Optional[List[float]] = None

@strawberry.type
class VectorVisualization:
    """Vector visualization data"""
    concept_id: str
    concept_name: str
    coordinates: List[float]
    null_ratio: float
    cluster: Optional[int] = None

# GraphQL Inputs
@strawberry.input
class DefineConceptInput:
    """Input for defining a concept"""
    concept: str
    negations: List[str]
    cultural_context: Optional[str] = "universal"

# GraphQL Queries
@strawberry.type
class Query:
    @strawberry.field
    async def concept(self, id: str) -> Optional[Concept]:
        """Get a concept by ID"""
        app_state = await get_app_state()
        db = app_state["db"]
        
        concept_data = await db.concepts.find_one({"id": id})
        if not concept_data:
            return None
        
        return Concept(
            id=concept_data["id"],
            name=concept_data["name"],
            confidence=concept_data["confidence"],
            null_ratio=concept_data.get("null_ratio", 0.0),
            vector_preview=None  # Vector is too large for GraphQL response
        )
    
    @strawberry.field
    async def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search for concepts by name"""
        app_state = await get_app_state()
        db = app_state["db"]
        
        # Simple text search
        concepts = await db.fetch(
            f"SELECT * FROM concepts WHERE name ILIKE $1 LIMIT $2",
            f"%{query}%",
            limit
        )
        
        return [
            Concept(
                id=concept["id"],
                name=concept["name"],
                confidence=concept["confidence"],
                null_ratio=concept.get("null_ratio", 0.0)
            )
            for concept in concepts
        ]
    
    @strawberry.field
    async def vector_visualization(
        self, 
        concept_ids: Optional[List[str]] = None,
        method: str = "tsne",
        dimensions: int = 3
    ) -> List[VectorVisualization]:
        """Get vector visualization data"""
        app_state = await get_app_state()
        
        # Import here to avoid circular imports
        from src.services.visualization import VisualizationService
        
        service = VisualizationService(app_state)
        
        # If no concepts specified, get top 50
        if not concept_ids:
            concept_ids = await service.get_top_concepts(50)
        
        viz_data = await service.create_visualization(
            concept_ids=concept_ids,
            method=method,
            dimensions=dimensions
        )
        
        return [
            VectorVisualization(
                concept_id=item["concept_id"],
                concept_name=item["concept_name"],
                coordinates=item["coordinates"],
                null_ratio=item["null_ratio"],
                cluster=item.get("cluster")
            )
            for item in viz_data["points"]
        ]

# GraphQL Mutations
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def define_concept(
        self, input: DefineConceptInput
    ) -> ConceptDefinitionResult:
        """Define a new concept through negation"""
        app_state = await get_app_state()
        service = DefinitionService(app_state)
        
        vectorized = await service.define_concept(
            concept=input.concept,
            negations=input.negations
        )
        
        # Convert empathy scores dict to EmpathyScores type
        empathy = None
        if vectorized.empathy_scores:
            empathy = EmpathyScores(
                self_empathy=vectorized.empathy_scores.get("self_empathy", 0.0),
                other_empathy=vectorized.empathy_scores.get("other_empathy", 0.0),
                mutual_empathy=vectorized.empathy_scores.get("mutual_empathy", 0.0)
            )
            
        return ConceptDefinitionResult(
            concept_id=vectorized.concept_id,
            concept_name=input.concept,
            confidence=vectorized.metadata.get("confidence", 0.0),
            null_ratio=vectorized.null_ratio,
            empathy_scores=empathy,
            vector_preview=vectorized.vector[:10].tolist() if vectorized.vector is not None else None
        )

# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create FastAPI GraphQL app
graphql_app = GraphQLRouter(schema)
