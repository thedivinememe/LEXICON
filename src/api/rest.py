from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Optional
from pydantic import BaseModel

from src.services.definition import DefinitionService
from src.services.normalization import NormalizationService
from src.api.dependencies import get_app_state

router = APIRouter()

class DefineConceptRequest(BaseModel):
    concept: str
    negations: List[str]
    cultural_context: Optional[str] = "universal"

class ConceptResponse(BaseModel):
    concept_id: str
    concept_name: str
    atomic_pattern: str
    null_ratio: float
    empathy_scores: Dict[str, float]
    vector_preview: List[float]  # First 10 dimensions

@router.post("/concepts/define", response_model=ConceptResponse)
async def define_concept(
    request: DefineConceptRequest,
    app_state = Depends(get_app_state)
):
    """Define a new concept through negation"""
    service = DefinitionService(app_state)
    
    try:
        vectorized = await service.define_concept(
            concept=request.concept,
            negations=request.negations
        )
        
        return ConceptResponse(
            concept_id=vectorized.concept_id,
            concept_name=request.concept,
            atomic_pattern=str(vectorized.metadata.get('atomic_pattern')),
            null_ratio=vectorized.null_ratio,
            empathy_scores=vectorized.empathy_scores,
            vector_preview=vectorized.vector[:10].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/concepts/{concept_id}/similar")
async def get_similar_concepts(
    concept_id: str,
    k: int = Query(10, ge=1, le=100),
    app_state = Depends(get_app_state)
):
    """Find concepts similar to the given one"""
    service = DefinitionService(app_state)
    
    try:
        similar = await service.get_similar_concepts(concept_id, k)
        return {"concept_id": concept_id, "similar_concepts": similar}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/concepts/normalize")
async def normalize_concepts(
    concept_ids: List[str],
    app_state = Depends(get_app_state)
):
    """Normalize multiple concepts for comparison"""
    service = NormalizationService(app_state)
    
    try:
        result = await service.normalize_set(concept_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/vectors/visualize")
async def get_visualization_data(
    concept_ids: Optional[List[str]] = Query(None),
    method: str = Query("tsne", regex="^(tsne|pca|umap)$"),
    dimensions: int = Query(3, ge=2, le=3),
    app_state = Depends(get_app_state)
):
    """Get vector visualization data"""
    from src.services.visualization import VisualizationService
    
    service = VisualizationService(app_state)
    
    # If no concepts specified, get top 50
    if not concept_ids:
        concept_ids = await service.get_top_concepts(50)
    
    try:
        viz_data = await service.create_visualization(
            concept_ids=concept_ids,
            method=method,
            dimensions=dimensions
        )
        return viz_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
