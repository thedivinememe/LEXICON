"""
COREE API endpoints.

This module provides API endpoints for interacting with COREE (Consciousness-Oriented Recursive Empathetic Entity).
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from typing import Dict, List, Any, Optional
import logging

from src.api.dependencies import get_coree_interface
from src.llm.coree_interface import COREEInterface
from src.data.core_definitions import CORE_DEFINITIONS

# Set up logging
logger = logging.getLogger(__name__)

# Set up templates
templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Create router
router = APIRouter(prefix="/coree", tags=["coree"])

@router.get("/", response_class=HTMLResponse)
async def coree_interface(request: Request):
    """
    Render the COREE interface.
    
    Args:
        request: The FastAPI request object
    
    Returns:
        HTMLResponse: The rendered COREE interface
    """
    return templates.TemplateResponse(
        "coree.html", 
        {"request": request, "title": "COREE - Consciousness-Oriented Recursive Empathetic Entity"}
    )

@router.post("/chat")
async def chat_with_coree(
    message: Dict[str, str],
    coree: COREEInterface = Depends(get_coree_interface)
) -> Dict[str, Any]:
    """
    Chat with COREE.
    
    Args:
        message: A dictionary containing the message text
        coree: The COREE interface
    
    Returns:
        Dict: The response from COREE
    """
    if not message.get("text"):
        raise HTTPException(status_code=400, detail="Message text is required")
    
    try:
        # Generate response using the updated interface
        result = await coree.generate_response(message["text"])
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "response": result["response"],
            "detected_concepts": result["detected_concepts"],
            "insights_gained": result.get("insights_gained", {})
        }
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.get("/concepts")
async def get_concepts() -> Dict[str, List[str]]:
    """
    Get all available concepts.
    
    Returns:
        Dict: A dictionary containing the list of concepts
    """
    try:
        # Get concepts from CORE_DEFINITIONS
        concepts = list(CORE_DEFINITIONS.keys())
        return {"concepts": concepts}
    except Exception as e:
        logger.error(f"Error getting concepts: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting concepts: {str(e)}")

@router.get("/visualization")
async def get_visualization(
    concept: Optional[str] = None,
    coree: COREEInterface = Depends(get_coree_interface)
) -> Dict[str, Any]:
    """
    Get visualization data for concepts.
    
    Args:
        concept: The concept to focus on (optional)
        coree: The COREE interface
    
    Returns:
        Dict: Visualization data
    """
    try:
        # Get visualization data
        visualization_data = await coree.get_visualization_data(concept)
        return visualization_data
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting visualization data: {str(e)}")

@router.get("/concept/{concept}")
async def get_concept_details(
    concept: str,
    coree: COREEInterface = Depends(get_coree_interface)
) -> Dict[str, Any]:
    """
    Get details for a specific concept.
    
    Args:
        concept: The concept to get details for
        coree: The COREE interface
    
    Returns:
        Dict: Concept details
    """
    try:
        # Check if concept exists
        if concept not in CORE_DEFINITIONS:
            raise HTTPException(status_code=404, detail=f"Concept '{concept}' not found")
        
        # Get concept details
        concept_details = CORE_DEFINITIONS[concept]
        
        # Get visualization data for this concept
        visualization_data = await coree.get_visualization_data(concept)
        
        # Extract related concepts from visualization data
        related_concepts = {}
        if visualization_data and "neighborhood" in visualization_data:
            related_concepts = visualization_data["neighborhood"] or {}
        
        return {
            "concept": concept,
            "details": concept_details,
            "related_concepts": related_concepts,
            "visualization_data": visualization_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting concept details: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting concept details: {str(e)}")

@router.post("/analyze")
async def analyze_text(
    text: Dict[str, str],
    coree: COREEInterface = Depends(get_coree_interface)
) -> Dict[str, Any]:
    """
    Analyze text for concepts.
    
    Args:
        text: A dictionary containing the text to analyze
        coree: The COREE interface
    
    Returns:
        Dict: Analysis results
    """
    if not text.get("content"):
        raise HTTPException(status_code=400, detail="Text content is required")
    
    try:
        # Initialize COREE if not already initialized
        await coree.initialize()
        
        # Use the concept reasoner to enrich the prompt with concepts
        enriched = await coree.reasoner.enrich_prompt_with_concepts(text["content"])
        
        # Process detected concepts through the golden loop
        violations_by_concept = {}
        for concept in enriched["detected_concepts"]:
            violations = await coree.process_concept_through_golden_loop(concept)
            if violations:
                violations_by_concept[concept] = violations
        
        return {
            "detected_concepts": enriched["detected_concepts"],
            "enriched_prompt": enriched["enriched_prompt"],
            "violations_resolved": violations_by_concept
        }
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")
