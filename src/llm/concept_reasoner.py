"""
Concept Reasoner for LEXICON LLM Integration.

This module provides reasoning capabilities for LEXICON concepts.
"""

import logging
from src.data.core_definitions import CORE_DEFINITIONS
from src.core.spherical_universe import BlochSphereUniverse
from src.examples.spherical_integration_example import initialize_universe

logger = logging.getLogger(__name__)


class ConceptReasoner:
    """
    Reasoner for LEXICON concepts.
    
    This class provides methods for reasoning about relationships
    between LEXICON concepts, including finding related concepts,
    calculating distances, and finding paths between concepts.
    """
    
    def __init__(self):
        """
        Initialize the ConceptReasoner.
        
        Note: This does not initialize the universe. Call initialize() to do that.
        """
        self.universe = None
        
    @classmethod
    async def create(cls):
        """
        Create and initialize a ConceptReasoner.
        
        This is a factory method that creates a ConceptReasoner and initializes
        the spherical universe with all concepts and their relationships.
        
        Returns:
            ConceptReasoner: An initialized ConceptReasoner.
        """
        logger.info("Creating ConceptReasoner with spherical universe")
        reasoner = cls()
        await reasoner.initialize()
        return reasoner
        
    async def initialize(self):
        """
        Initialize the spherical universe with all concepts and their relationships.
        """
        logger.info("Initializing ConceptReasoner with spherical universe")
        self.universe, _, _, _, _ = await initialize_universe()
        
    async def get_related_concepts(self, concept):
        """
        Get concepts related to the given concept.
        
        Args:
            concept (str): The concept to find related concepts for.
            
        Returns:
            dict: A dictionary of related concepts grouped by relationship type.
        """
        if concept not in self.universe.concepts:
            return {"error": f"Concept '{concept}' not found"}
            
        related = self.universe.get_related_concepts(concept)
        result = {}
        
        for related_concept in related:
            rel_type = self.universe.get_relationship_type(concept, related_concept)
            if rel_type not in result:
                result[rel_type] = []
            result[rel_type].append(related_concept)
            
        return result
        
    async def get_concept_distance(self, concept1, concept2):
        """
        Get the distance between two concepts.
        
        Args:
            concept1 (str): The first concept.
            concept2 (str): The second concept.
            
        Returns:
            dict: A dictionary containing the distance and position information.
        """
        if concept1 not in self.universe.concepts:
            return {"error": f"Concept '{concept1}' not found"}
        if concept2 not in self.universe.concepts:
            return {"error": f"Concept '{concept2}' not found"}
            
        pos1 = self.universe.get_concept_position(concept1)
        pos2 = self.universe.get_concept_position(concept2)
        
        # Calculate Euclidean distance
        cart1 = pos1.to_cartesian()
        cart2 = pos2.to_cartesian()
        distance = ((cart1[0] - cart2[0])**2 + 
                    (cart1[1] - cart2[1])**2 + 
                    (cart1[2] - cart2[2])**2)**0.5
                    
        return {
            "distance": distance,
            "concept1": concept1,
            "concept2": concept2,
            "position1": {"r": pos1.r, "theta": pos1.theta, "phi": pos1.phi},
            "position2": {"r": pos2.r, "theta": pos2.theta, "phi": pos2.phi}
        }
        
    async def find_path(self, concept1, concept2, max_depth=3):
        """
        Find a path between two concepts.
        
        Args:
            concept1 (str): The first concept.
            concept2 (str): The second concept.
            max_depth (int): The maximum path depth to search.
            
        Returns:
            dict: A dictionary containing the path or an error message.
        """
        if concept1 not in self.universe.concepts:
            return {"error": f"Concept '{concept1}' not found"}
        if concept2 not in self.universe.concepts:
            return {"error": f"Concept '{concept2}' not found"}
            
        # Simple BFS to find path
        visited = {concept1}
        queue = [(concept1, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == concept2:
                return {"path": path + [current]}
                
            if len(path) >= max_depth:
                continue
                
            related = self.universe.get_related_concepts(current)
            for next_concept in related:
                if next_concept not in visited:
                    visited.add(next_concept)
                    queue.append((next_concept, path + [current]))
                    
        return {"error": f"No path found between '{concept1}' and '{concept2}' within depth {max_depth}"}
    
    async def get_concept_neighborhood(self, concept, depth=1):
        """
        Get the neighborhood of a concept.
        
        Args:
            concept (str): The concept to find the neighborhood for.
            depth (int): The depth of the neighborhood.
            
        Returns:
            dict: A dictionary containing the neighborhood concepts and their relationships.
        """
        if concept not in self.universe.concepts:
            return {"error": f"Concept '{concept}' not found"}
            
        # BFS to find neighborhood
        visited = {concept}
        queue = [(concept, 0)]
        neighborhood = {}
        
        while queue:
            current, current_depth = queue.pop(0)
            
            if current_depth > depth:
                continue
                
            if current != concept:  # Don't include the center concept itself
                rel_type = self.universe.get_relationship_type(concept, current)
                if rel_type not in neighborhood:
                    neighborhood[rel_type] = []
                neighborhood[rel_type].append(current)
                
            if current_depth < depth:
                related = self.universe.get_related_concepts(current)
                for next_concept in related:
                    if next_concept not in visited:
                        visited.add(next_concept)
                        queue.append((next_concept, current_depth + 1))
                    
        return {
            "concept": concept,
            "depth": depth,
            "neighborhood": neighborhood
        }
    
    async def get_concept_vector(self, concept):
        """
        Get the vector representation of a concept.
        
        Args:
            concept (str): The concept to get the vector for.
            
        Returns:
            dict: A dictionary containing the vector representation.
        """
        if concept not in self.universe.concepts:
            return {"error": f"Concept '{concept}' not found"}
            
        pos = self.universe.get_concept_position(concept)
        cart = pos.to_cartesian()
        
        return {
            "concept": concept,
            "spherical": {"r": pos.r, "theta": pos.theta, "phi": pos.phi},
            "cartesian": {"x": cart[0], "y": cart[1], "z": cart[2]}
        }
    
    async def enrich_prompt_with_concepts(self, prompt, max_concepts=5):
        """
        Enrich a prompt with relevant LEXICON concepts.
        
        Args:
            prompt (str): The prompt to enrich.
            max_concepts (int): The maximum number of concepts to include.
            
        Returns:
            dict: A dictionary containing the enriched prompt and detected concepts.
        """
        # Extract potential concepts from the prompt
        words = set(prompt.lower().split())
        detected_concepts = []
        
        # First, check for exact matches
        for concept in CORE_DEFINITIONS:
            if concept.lower() in words:
                detected_concepts.append(concept)
                
        # If we don't have enough concepts, search for partial matches
        if len(detected_concepts) < max_concepts:
            for word in words:
                if len(word) < 3:  # Skip very short words
                    continue
                    
                for concept in CORE_DEFINITIONS:
                    if word in concept.lower() and concept not in detected_concepts:
                        detected_concepts.append(concept)
                        if len(detected_concepts) >= max_concepts:
                            break
                            
                if len(detected_concepts) >= max_concepts:
                    break
        
        # Create an enriched system prompt
        system_prompt = """
You are an AI assistant that understands the LEXICON system of concepts.
The following concepts from LEXICON are relevant to this conversation:

"""
        
        for concept in detected_concepts:
            definition = CORE_DEFINITIONS.get(concept, {})
            
            system_prompt += f"\n## {concept}\n"
            
            if "atomic_pattern" in definition:
                system_prompt += f"Atomic Pattern: {definition['atomic_pattern']}\n"
                
            if "not_space" in definition:
                system_prompt += f"Not Space: {', '.join(definition['not_space'])}\n"
                
            for rel_type, rel_name in [
                ("and_relationships", "AND Relationships"),
                ("or_relationships", "OR Relationships"),
                ("not_relationships", "NOT Relationships")
            ]:
                if rel_type in definition:
                    rel_str = ", ".join([f"{rel[0]} ({rel[1]})" for rel in definition[rel_type]])
                    system_prompt += f"{rel_name}: {rel_str}\n"
                    
            if "vector_properties" in definition:
                system_prompt += f"Vector Properties: {definition['vector_properties']}\n"
                
            if "spherical_properties" in definition and isinstance(definition["spherical_properties"], dict):
                system_prompt += "Spherical Properties:\n"
                for key, value in definition["spherical_properties"].items():
                    system_prompt += f"  - {key}: {value}\n"
                    
            system_prompt += "\n"
        
        return {
            "original_prompt": prompt,
            "enriched_system_prompt": system_prompt,
            "detected_concepts": detected_concepts
        }
