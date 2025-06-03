#!/usr/bin/env python
"""
LEXICON LLM Integration Example.

This script demonstrates how to use the LEXICON LLM integration in a Python script.
"""

import asyncio
import os
import sys
from pathlib import Path
import httpx
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm.definition_provider import DefinitionProvider
from src.llm.concept_reasoner import ConceptReasoner


async def example_1_basic_concept_info():
    """Example 1: Get basic information about a concept."""
    print("\n=== Example 1: Basic Concept Info ===\n")
    
    # Get a concept definition
    concept = "existence"
    definition = DefinitionProvider.get_definition(concept)
    
    print(f"Concept: {concept}")
    print(f"Atomic Pattern: {definition.get('atomic_pattern', 'N/A')}")
    
    if "not_space" in definition:
        print(f"Not Space: {', '.join(definition['not_space'])}")
        
    # Print relationships
    for rel_type, rel_name in [
        ("and_relationships", "AND Relationships"),
        ("or_relationships", "OR Relationships"),
        ("not_relationships", "NOT Relationships")
    ]:
        if rel_type in definition:
            rel_str = ", ".join([f"{rel[0]} ({rel[1]})" for rel in definition[rel_type]])
            print(f"{rel_name}: {rel_str}")


async def example_2_concept_relationships():
    """Example 2: Explore relationships between concepts."""
    print("\n=== Example 2: Concept Relationships ===\n")
    
    reasoner = await ConceptReasoner.create()
    
    # Get related concepts
    concept = "existence"
    related = await reasoner.get_related_concepts(concept)
    
    print(f"Concepts related to '{concept}':")
    for rel_type, concepts in related.items():
        print(f"  {rel_type}: {', '.join(concepts)}")
        
    # Find a path between concepts
    concept1 = "existence"
    concept2 = "knowledge"
    path = await reasoner.find_path(concept1, concept2)
    
    if "path" in path:
        print(f"\nPath from '{concept1}' to '{concept2}':")
        print(" -> ".join(path["path"]))
    else:
        print(f"\nNo path found from '{concept1}' to '{concept2}'")
        
    # Get concept distance
    distance = await reasoner.get_concept_distance(concept1, concept2)
    print(f"\nDistance between '{concept1}' and '{concept2}': {distance['distance']:.4f}")


async def example_3_enrich_prompt():
    """Example 3: Enrich a prompt with LEXICON concepts."""
    print("\n=== Example 3: Enrich Prompt ===\n")
    
    reasoner = await ConceptReasoner.create()
    
    # Create a prompt
    prompt = "Explain how existence relates to pattern and relationship in the context of consciousness."
    
    # Enrich the prompt
    enriched = await reasoner.enrich_prompt_with_concepts(prompt)
    
    print("Original prompt:")
    print(prompt)
    
    print("\nDetected concepts:")
    for concept in enriched["detected_concepts"]:
        print(f"- {concept}")
        
    print("\nEnriched system prompt:")
    print(enriched["enriched_system_prompt"])


async def example_4_generate_with_llm():
    """Example 4: Generate text with an LLM using LEXICON concepts."""
    print("\n=== Example 4: Generate with LLM ===\n")
    
    # Get the OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        print("Skipping this example.")
        return
        
    reasoner = await ConceptReasoner.create()
    
    # Create a prompt
    prompt = "Explain how existence relates to pattern and relationship in the context of consciousness."
    
    # Enrich the prompt
    enriched = await reasoner.enrich_prompt_with_concepts(prompt)
    
    print("Original prompt:")
    print(prompt)
    
    print("\nDetected concepts:")
    for concept in enriched["detected_concepts"]:
        print(f"- {concept}")
    
    print("\nEnriched system prompt:")
    print(enriched["enriched_system_prompt"])
    
    # Check if we should use a mock response (for demonstration purposes)
    use_mock = os.environ.get("USE_MOCK_LLM", "false").lower() in ("true", "1", "yes")
    
    if use_mock:
        print("\nUsing mock LLM response for demonstration purposes.")
        print("\nLLM Response (MOCK):")
        mock_response = """
In the LEXICON framework, existence, pattern, and relationship form a fundamental triad that underpins consciousness.

Existence (atomic pattern: 1) serves as the foundational concept - the very basis of being. It represents the origin point in vector space, with a radial growth pattern. Within consciousness, existence manifests as the fundamental awareness that "something is" rather than nothing.

Pattern (atomic pattern: Repeating(Structure)) provides the structured arrangement that allows consciousness to form. Consciousness requires recognizable patterns to distinguish signal from noise. The fractal growth pattern of "pattern" enables consciousness to recognize similarities across different scales and contexts.

Relationship (atomic pattern: Connection(A, B)) creates the network of connections that bind patterns of existence together. Consciousness emerges from these relationships - the connections between patterns of neural activity. The network growth pattern of "relationship" enables the complex web of associations that characterize conscious thought.

In consciousness, these three concepts operate together:
1. Existence provides the fundamental substrate
2. Pattern provides the structured arrangements
3. Relationship provides the connections between patterns

Consciousness itself (which has AND relationships with I/self and knowledge) emerges as a meta-pattern of relationships between patterns of existence. It requires all three concepts working in harmony - existence as the foundation, patterns that can be recognized, and relationships that connect these patterns into a coherent whole.

This triad forms what might be called the "existential grammar" of consciousness - existence as the nouns, patterns as the adjectives, and relationships as the verbs that together construct the sentences of conscious experience.
"""
        print(mock_response)
        print("\nToken usage (MOCK): 350 tokens")
        return
    
    print("\nTo generate a response with an LLM, you need a valid OpenAI API key with sufficient quota.")
    print("You can:")
    print("1. Check your OpenAI account at https://platform.openai.com/account/usage")
    print("2. Ensure your billing information is up to date")
    print("3. Consider using a different model (e.g., gpt-3.5-turbo instead of gpt-4)")
    print("4. If you're using a free tier, be aware of usage limits")
    print("5. Set USE_MOCK_LLM=true as an environment variable to see a mock response")
    
    print("\nGenerating response...")
    
    # Call the OpenAI API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",  # Consider using a less expensive model
                    "messages": [
                        {"role": "system", "content": enriched["enriched_system_prompt"]},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"Error from OpenAI API: {response.status_code}")
                error_data = response.json() if response.text else {"error": {"type": "unknown"}}
                print(response.text)
                
                # Handle insufficient quota error specifically
                if "error" in error_data and error_data["error"].get("type") == "insufficient_quota":
                    print("\nYou've exceeded your OpenAI API quota. You can:")
                    print("1. Check your OpenAI account at https://platform.openai.com/account/usage")
                    print("2. Ensure your billing information is up to date")
                    print("3. Consider using a different model (e.g., gpt-3.5-turbo instead of gpt-4)")
                    print("4. If you're using a free tier, be aware of usage limits")
                    print("5. Set USE_MOCK_LLM=true as an environment variable to see a mock response")
                
                return
                
            result = response.json()
            
            print("\nLLM Response:")
            print(result["choices"][0]["message"]["content"])
            
            if "usage" in result:
                print(f"\nToken usage: {result['usage']['total_tokens']} tokens")
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            print("\nTip: Set USE_MOCK_LLM=true as an environment variable to see a mock response")


async def main():
    """Main entry point for the script."""
    print("LEXICON LLM Integration Example")
    print("===============================")
    
    await example_1_basic_concept_info()
    await example_2_concept_relationships()
    await example_3_enrich_prompt()
    await example_4_generate_with_llm()


if __name__ == "__main__":
    asyncio.run(main())
