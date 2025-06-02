"""
Export service for LEXICON.
Provides methods for exporting concepts to various formats.
"""

from typing import Dict, List, Any, Optional, Set
import json
import numpy as np
from datetime import datetime

class ExportService:
    """Service for exporting concepts to various formats"""
    
    def __init__(self, app_state: Dict[str, Any]):
        """Initialize the export service"""
        self.app_state = app_state
        self.db = app_state["db"]
        self.cache = app_state["cache"]
        self.vector_store = app_state["vector_store"]
    
    async def export_to_llm(self, concept_ids: List[str]) -> Dict[str, Any]:
        """
        Export concepts to a format suitable for LLM consumption.
        
        Args:
            concept_ids: List of concept IDs to export
        
        Returns:
            Dictionary with LLM-friendly concept definitions
        """
        # Check cache
        cache_key = f"export_llm:{'|'.join(sorted(concept_ids))}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get concept data
        concepts = []
        
        for concept_id in concept_ids:
            # Get concept from database
            concept = await self.db.concepts.find_one({"id": concept_id})
            if not concept:
                continue
            
            # Get not-space as list
            not_space = list(concept.get("not_space", []))
            
            # Format concept for LLM
            concepts.append({
                "concept": concept["name"],
                "definition": self._generate_definition(concept),
                "is_not": not_space,
                "confidence": concept["confidence"]
            })
        
        result = {
            "concepts": concepts,
            "exported_at": datetime.utcnow().isoformat(),
            "format_version": "1.0"
        }
        
        # Cache the result
        await self.cache.set(cache_key, result, expire=3600)  # 1 hour
        
        return result
    
    async def export_to_json(self, concept_ids: List[str], include_vectors: bool = False) -> str:
        """
        Export concepts to JSON format.
        
        Args:
            concept_ids: List of concept IDs to export
            include_vectors: Whether to include vector data
        
        Returns:
            JSON string with concept data
        """
        # Check cache
        cache_key = f"export_json:{include_vectors}:{'|'.join(sorted(concept_ids))}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get concept data
        concepts = []
        
        for concept_id in concept_ids:
            # Get concept from database
            concept = await self.db.concepts.find_one({"id": concept_id})
            if not concept:
                continue
            
            # Create export object
            export_concept = {
                "id": concept["id"],
                "name": concept["name"],
                "atomic_pattern": concept["atomic_pattern"],
                "not_space": list(concept.get("not_space", [])),
                "confidence": concept["confidence"],
                "created_at": concept["created_at"].isoformat() if isinstance(concept["created_at"], datetime) else concept["created_at"],
                "updated_at": concept["updated_at"].isoformat() if isinstance(concept["updated_at"], datetime) else concept["updated_at"]
            }
            
            # Include vector if requested
            if include_vectors:
                vector = self.vector_store.get_vector(concept_id)
                if vector is not None:
                    export_concept["vector"] = vector.tolist()
            
            concepts.append(export_concept)
        
        result = {
            "concepts": concepts,
            "exported_at": datetime.utcnow().isoformat(),
            "format_version": "1.0"
        }
        
        # Convert to JSON
        json_str = json.dumps(result, indent=2)
        
        # Cache the result
        await self.cache.set(cache_key, json_str, expire=3600)  # 1 hour
        
        return json_str
    
    async def export_to_csv(self, concept_ids: List[str]) -> str:
        """
        Export concepts to CSV format.
        
        Args:
            concept_ids: List of concept IDs to export
        
        Returns:
            CSV string with concept data
        """
        # Check cache
        cache_key = f"export_csv:{'|'.join(sorted(concept_ids))}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # CSV header
        csv_lines = ["id,name,confidence,not_space"]
        
        for concept_id in concept_ids:
            # Get concept from database
            concept = await self.db.concepts.find_one({"id": concept_id})
            if not concept:
                continue
            
            # Format not-space as comma-separated list
            not_space_str = "|".join(concept.get("not_space", []))
            
            # Add CSV line
            csv_lines.append(f"{concept['id']},{concept['name']},{concept['confidence']},{not_space_str}")
        
        # Join lines
        csv_str = "\n".join(csv_lines)
        
        # Cache the result
        await self.cache.set(cache_key, csv_str, expire=3600)  # 1 hour
        
        return csv_str
    
    def _generate_definition(self, concept: Dict[str, Any]) -> str:
        """
        Generate a natural language definition for a concept.
        
        Args:
            concept: Concept data
        
        Returns:
            Natural language definition
        """
        name = concept["name"]
        not_space = list(concept.get("not_space", []))
        
        # Basic definition template
        definition = f"{name} is a concept that"
        
        # Add not-space information
        if not_space:
            if len(not_space) == 1:
                definition += f" is distinct from {not_space[0]}"
            elif len(not_space) == 2:
                definition += f" is distinct from both {not_space[0]} and {not_space[1]}"
            else:
                not_list = ", ".join(not_space[:-1]) + f", and {not_space[-1]}"
                definition += f" is distinct from {not_list}"
        
        # Add confidence information
        confidence = concept["confidence"]
        if confidence > 0.9:
            definition += ". This definition has high confidence."
        elif confidence > 0.7:
            definition += ". This definition has moderate confidence."
        else:
            definition += ". This definition has low confidence."
        
        return definition
