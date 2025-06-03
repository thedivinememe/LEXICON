"""
Definition Provider for LEXICON LLM Integration.

This module provides access to LEXICON core definitions for LLM integration.
"""

from src.data.core_definitions import CORE_DEFINITIONS
import json


class DefinitionProvider:
    """
    Provider for accessing LEXICON core definitions.
    
    This class provides methods for retrieving and searching LEXICON
    core definitions for use in LLM integration.
    """
    
    @staticmethod
    def get_definition(concept):
        """
        Get a specific concept definition.
        
        Args:
            concept (str): The name of the concept to retrieve.
            
        Returns:
            dict: The concept definition, or an empty dict if not found.
        """
        return CORE_DEFINITIONS.get(concept, {})
    
    @staticmethod
    def get_all_definitions():
        """
        Get all concept definitions.
        
        Returns:
            dict: All concept definitions.
        """
        return CORE_DEFINITIONS
    
    @staticmethod
    def get_definition_json(concept):
        """
        Get a specific concept definition as JSON.
        
        Args:
            concept (str): The name of the concept to retrieve.
            
        Returns:
            str: The concept definition as a JSON string.
        """
        return json.dumps(CORE_DEFINITIONS.get(concept, {}), indent=2)
    
    @staticmethod
    def search_definitions(query):
        """
        Search for concepts matching a query.
        
        Args:
            query (str): The search query.
            
        Returns:
            dict: A dictionary of matching concepts and their definitions.
        """
        results = {}
        query = query.lower()
        
        for concept, definition in CORE_DEFINITIONS.items():
            # Search in concept name
            if query in concept.lower():
                results[concept] = definition
                continue
                
            # Search in atomic pattern
            if "atomic_pattern" in definition and query in definition["atomic_pattern"].lower():
                results[concept] = definition
                continue
                
            # Search in not_space
            if "not_space" in definition:
                for term in definition["not_space"]:
                    if query in term.lower():
                        results[concept] = definition
                        break
                if concept in results:
                    continue
                
            # Search in relationships
            for rel_type in ["and_relationships", "or_relationships", "not_relationships"]:
                if rel_type in definition:
                    for rel in definition[rel_type]:
                        if query in rel[0].lower():
                            results[concept] = definition
                            break
                    if concept in results:
                        break
                        
            # Search in vector_properties
            if "vector_properties" in definition and isinstance(definition["vector_properties"], str):
                if query in definition["vector_properties"].lower():
                    results[concept] = definition
                    continue
                    
            # Search in spherical_properties
            if "spherical_properties" in definition and isinstance(definition["spherical_properties"], dict):
                for key, value in definition["spherical_properties"].items():
                    if isinstance(value, str) and query in value.lower():
                        results[concept] = definition
                        break
        
        return results
    
    @staticmethod
    def format_definition_for_prompt(concept):
        """
        Format a concept definition for inclusion in an LLM prompt.
        
        Args:
            concept (str): The name of the concept to format.
            
        Returns:
            str: A formatted string representation of the concept.
        """
        definition = CORE_DEFINITIONS.get(concept, {})
        if not definition:
            return f"Concept '{concept}' not found."
            
        formatted = f"Concept: {concept}\n"
        
        if "atomic_pattern" in definition:
            formatted += f"Atomic Pattern: {definition['atomic_pattern']}\n"
            
        if "not_space" in definition:
            formatted += f"Not Space: {', '.join(definition['not_space'])}\n"
            
        for rel_type, rel_name in [
            ("and_relationships", "AND Relationships"),
            ("or_relationships", "OR Relationships"),
            ("not_relationships", "NOT Relationships")
        ]:
            if rel_type in definition:
                rel_str = ", ".join([f"{rel[0]} ({rel[1]})" for rel in definition[rel_type]])
                formatted += f"{rel_name}: {rel_str}\n"
                
        if "vector_properties" in definition:
            formatted += f"Vector Properties: {definition['vector_properties']}\n"
            
        if "spherical_properties" in definition:
            formatted += "Spherical Properties:\n"
            for key, value in definition["spherical_properties"].items():
                formatted += f"  - {key}: {value}\n"
                
        return formatted
