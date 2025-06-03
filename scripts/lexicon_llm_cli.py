#!/usr/bin/env python
"""
LEXICON LLM CLI.

This script provides a command-line interface for interacting with
LEXICON concepts and integrating them with LLMs.
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.definition_provider import DefinitionProvider
from src.llm.concept_reasoner import ConceptReasoner


async def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="LEXICON LLM CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Define concept command
    concept_parser = subparsers.add_parser("concept", help="Get information about a concept")
    concept_parser.add_argument("name", help="Name of the concept")
    
    # Define related command
    related_parser = subparsers.add_parser("related", help="Get related concepts")
    related_parser.add_argument("name", help="Name of the concept")
    
    # Define path command
    path_parser = subparsers.add_parser("path", help="Find path between concepts")
    path_parser.add_argument("concept1", help="First concept")
    path_parser.add_argument("concept2", help="Second concept")
    path_parser.add_argument("--max-depth", type=int, default=3, help="Maximum path depth")
    
    # Define neighborhood command
    neighborhood_parser = subparsers.add_parser("neighborhood", help="Get concept neighborhood")
    neighborhood_parser.add_argument("name", help="Name of the concept")
    neighborhood_parser.add_argument("--depth", type=int, default=1, help="Neighborhood depth")
    
    # Define search command
    search_parser = subparsers.add_parser("search", help="Search for concepts")
    search_parser.add_argument("query", help="Search query")
    
    # Define prompt command
    prompt_parser = subparsers.add_parser("prompt", help="Generate an LLM prompt with LEXICON concepts")
    prompt_parser.add_argument("text", help="Prompt text")
    prompt_parser.add_argument("--max-concepts", type=int, default=5, help="Maximum number of concepts to include")
    
    # Define generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text with an LLM using LEXICON concepts")
    generate_parser.add_argument("text", help="Prompt text")
    generate_parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    generate_parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens to generate")
    generate_parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    
    # Define list command
    list_parser = subparsers.add_parser("list", help="List all available concepts")
    
    args = parser.parse_args()
    
    # Initialize reasoner
    reasoner = await ConceptReasoner.create()
    
    if args.command == "concept":
        definition = DefinitionProvider.get_definition(args.name)
        if not definition:
            print(f"Concept '{args.name}' not found")
            return
            
        print(json.dumps(definition, indent=2))
        
    elif args.command == "related":
        related = await reasoner.get_related_concepts(args.name)
        if "error" in related:
            print(related["error"])
            return
            
        print(json.dumps(related, indent=2))
        
    elif args.command == "path":
        path = await reasoner.find_path(args.concept1, args.concept2, args.max_depth)
        if "error" in path:
            print(path["error"])
            return
            
        print(f"Path from '{args.concept1}' to '{args.concept2}':")
        print(" -> ".join(path["path"]))
        
    elif args.command == "neighborhood":
        neighborhood = await reasoner.get_concept_neighborhood(args.name, args.depth)
        if "error" in neighborhood:
            print(neighborhood["error"])
            return
            
        print(json.dumps(neighborhood, indent=2))
        
    elif args.command == "search":
        results = DefinitionProvider.search_definitions(args.query)
        if not results:
            print(f"No concepts found matching '{args.query}'")
            return
            
        print(f"Found {len(results)} concepts matching '{args.query}':")
        for concept in results:
            print(f"- {concept}")
            
    elif args.command == "prompt":
        enriched = await reasoner.enrich_prompt_with_concepts(args.text, args.max_concepts)
        
        print("Detected concepts:")
        for concept in enriched["detected_concepts"]:
            print(f"- {concept}")
            
        print("\nEnriched system prompt:")
        print(enriched["enriched_system_prompt"])
        
    elif args.command == "generate":
        # Get the OpenAI API key
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key not provided. Use --api-key or set the OPENAI_API_KEY environment variable.")
            return
            
        # Enrich the prompt
        enriched = await reasoner.enrich_prompt_with_concepts(args.text)
        
        print("Detected concepts:")
        for concept in enriched["detected_concepts"]:
            print(f"- {concept}")
            
        print("\nEnriched system prompt:")
        print(enriched["enriched_system_prompt"])
        
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
                        "model": args.model if args.model else "gpt-3.5-turbo",  # Default to a less expensive model
                        "messages": [
                            {"role": "system", "content": enriched["enriched_system_prompt"]},
                            {"role": "user", "content": args.text}
                        ],
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    print(f"Error from OpenAI API: {response.status_code}")
                    error_data = response.json()
                    print(response.text)
                    
                    # Handle insufficient quota error specifically
                    if "error" in error_data and error_data["error"].get("type") == "insufficient_quota":
                        print("\nYou've exceeded your OpenAI API quota. You can:")
                        print("1. Check your OpenAI account at https://platform.openai.com/account/usage")
                        print("2. Ensure your billing information is up to date")
                        print("3. Consider using a different model (e.g., gpt-3.5-turbo instead of gpt-4)")
                        print("4. If you're using a free tier, be aware of usage limits")
                    
                    return
                
                result = response.json()
                
                print("\nLLM Response:")
                print(result["choices"][0]["message"]["content"])
                
                if "usage" in result:
                    print(f"\nToken usage: {result['usage']['total_tokens']} tokens")
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                
    elif args.command == "list":
        concepts = list(DefinitionProvider.get_all_definitions().keys())
        concepts.sort()
        
        print(f"Available concepts ({len(concepts)}):")
        for concept in concepts:
            print(f"- {concept}")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
