"""
API for LEXICON LLM Integration.

This module provides a FastAPI-based REST API for integrating LEXICON
core definitions with LLMs.
"""

import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import json
import os

from src.llm.definition_provider import DefinitionProvider
from src.llm.concept_reasoner import ConceptReasoner

logger = logging.getLogger(__name__)

app = FastAPI(title="LEXICON LLM API", description="API for integrating LEXICON concepts with LLMs")


# Models
class LLMRequest(BaseModel):
    """Request model for LLM integration."""
    prompt: str = Field(..., description="The prompt to send to the LLM")
    model: str = Field("gpt-4", description="The LLM model to use")
    temperature: float = Field(0.7, description="The temperature to use for generation")
    max_tokens: int = Field(1000, description="The maximum number of tokens to generate")
    
    
class ConceptQuery(BaseModel):
    """Query model for concept information."""
    concept: str = Field(..., description="The name of the concept to query")
    
    
class RelationQuery(BaseModel):
    """Query model for concept relationships."""
    concept1: str = Field(..., description="The first concept")
    concept2: str = Field(..., description="The second concept")
    max_depth: Optional[int] = Field(3, description="The maximum path depth to search")
    
    
class NeighborhoodQuery(BaseModel):
    """Query model for concept neighborhoods."""
    concept: str = Field(..., description="The concept to find the neighborhood for")
    depth: Optional[int] = Field(1, description="The depth of the neighborhood")
    
    
class SearchQuery(BaseModel):
    """Query model for searching concepts."""
    query: str = Field(..., description="The search query")
    

# Dependencies
async def get_reasoner():
    """Get a ConceptReasoner instance."""
    return await ConceptReasoner.create()


# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LEXICON LLM API",
        "version": "1.0.0",
        "description": "API for integrating LEXICON concepts with LLMs"
    }


@app.post("/enrich_prompt")
async def enrich_prompt(
    request: LLMRequest,
    reasoner: ConceptReasoner = Depends(get_reasoner)
):
    """
    Enrich a prompt with LEXICON definitions.
    
    This endpoint takes a prompt and enriches it with relevant LEXICON
    definitions for use with an LLM.
    """
    try:
        result = await reasoner.enrich_prompt_with_concepts(request.prompt)
        return result
    except Exception as e:
        logger.exception("Error enriching prompt")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/concept_info")
async def concept_info(
    query: ConceptQuery,
    reasoner: ConceptReasoner = Depends(get_reasoner)
):
    """
    Get information about a concept.
    
    This endpoint returns information about a specific LEXICON concept,
    including its definition and related concepts.
    """
    try:
        definition = DefinitionProvider.get_definition(query.concept)
        if not definition:
            raise HTTPException(status_code=404, detail=f"Concept '{query.concept}' not found")
            
        related = await reasoner.get_related_concepts(query.concept)
        if "error" in related:
            raise HTTPException(status_code=404, detail=related["error"])
            
        return {
            "concept": query.concept,
            "definition": definition,
            "related_concepts": related
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting concept info")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/concept_relationship")
async def concept_relationship(
    query: RelationQuery,
    reasoner: ConceptReasoner = Depends(get_reasoner)
):
    """
    Get the relationship between two concepts.
    
    This endpoint returns information about the relationship between
    two LEXICON concepts, including the distance between them and
    a path connecting them.
    """
    try:
        distance = await reasoner.get_concept_distance(query.concept1, query.concept2)
        if "error" in distance:
            raise HTTPException(status_code=404, detail=distance["error"])
            
        path = await reasoner.find_path(query.concept1, query.concept2, query.max_depth)
        if "error" in path:
            # Don't raise an exception for path errors, just include the error in the response
            pass
            
        return {
            "distance": distance,
            "path": path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting concept relationship")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/concept_neighborhood")
async def concept_neighborhood(
    query: NeighborhoodQuery,
    reasoner: ConceptReasoner = Depends(get_reasoner)
):
    """
    Get the neighborhood of a concept.
    
    This endpoint returns information about the neighborhood of a
    LEXICON concept, including related concepts within a certain depth.
    """
    try:
        neighborhood = await reasoner.get_concept_neighborhood(query.concept, query.depth)
        if "error" in neighborhood:
            raise HTTPException(status_code=404, detail=neighborhood["error"])
            
        return neighborhood
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting concept neighborhood")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/concept_vector")
async def concept_vector(
    query: ConceptQuery,
    reasoner: ConceptReasoner = Depends(get_reasoner)
):
    """
    Get the vector representation of a concept.
    
    This endpoint returns the vector representation of a LEXICON concept
    in both spherical and Cartesian coordinates.
    """
    try:
        vector = await reasoner.get_concept_vector(query.concept)
        if "error" in vector:
            raise HTTPException(status_code=404, detail=vector["error"])
            
        return vector
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting concept vector")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_concepts")
async def search_concepts(query: SearchQuery):
    """
    Search for concepts matching a query.
    
    This endpoint searches for LEXICON concepts matching a query string.
    """
    try:
        results = DefinitionProvider.search_definitions(query.query)
        return {
            "query": query.query,
            "results": results
        }
    except Exception as e:
        logger.exception("Error searching concepts")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_with_llm")
async def generate_with_llm(
    request: LLMRequest,
    reasoner: ConceptReasoner = Depends(get_reasoner)
):
    """
    Generate text with an LLM using LEXICON concepts.
    
    This endpoint enriches a prompt with LEXICON concepts and sends it
    to an LLM for generation.
    
    Note: This endpoint requires an OpenAI API key to be set in the
    OPENAI_API_KEY environment variable.
    """
    try:
        # Get the OpenAI API key from the environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key not found. Set the OPENAI_API_KEY environment variable."
            )
            
        # Enrich the prompt with LEXICON concepts
        enriched = await reasoner.enrich_prompt_with_concepts(request.prompt)
        
        # Call the OpenAI API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": request.model,
                    "messages": [
                        {"role": "system", "content": enriched["enriched_system_prompt"]},
                        {"role": "user", "content": request.prompt}
                    ],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                error_message = f"Error from OpenAI API: {response.text}"
                
                # Handle insufficient quota error specifically
                try:
                    error_data = response.json()
                    if "error" in error_data and error_data["error"].get("type") == "insufficient_quota":
                        error_message = (
                            "You've exceeded your OpenAI API quota. Please check your OpenAI account "
                            "at https://platform.openai.com/account/usage, ensure your billing information "
                            "is up to date, or consider using a different model (e.g., gpt-3.5-turbo instead of gpt-4)."
                        )
                except Exception:
                    pass  # If we can't parse the JSON, just use the generic error message
                
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_message
                )
                
            result = response.json()
            
            return {
                "prompt": request.prompt,
                "enriched_system_prompt": enriched["enriched_system_prompt"],
                "detected_concepts": enriched["detected_concepts"],
                "response": result["choices"][0]["message"]["content"],
                "model": request.model,
                "usage": result.get("usage", {})
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating with LLM")
        raise HTTPException(status_code=500, detail=str(e))
