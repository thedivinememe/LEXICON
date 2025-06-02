#!/usr/bin/env python
"""
Database initialization script for LEXICON.
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.storage.database import Database

# SQL statements for table creation
CREATE_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS concepts (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        atomic_pattern JSONB NOT NULL,
        not_space JSONB NOT NULL,
        confidence FLOAT NOT NULL,
        null_ratio FLOAT,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts (name)
    """,
    """
    CREATE TABLE IF NOT EXISTS concept_access (
        id SERIAL PRIMARY KEY,
        concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
        user_id TEXT,
        accessed_at TIMESTAMP WITH TIME ZONE NOT NULL,
        context TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_concept_access_concept_id ON concept_access (concept_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_concept_access_accessed_at ON concept_access (accessed_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        email TEXT UNIQUE,
        password_hash TEXT NOT NULL,
        is_admin BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        last_login TIMESTAMP WITH TIME ZONE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memetic_evolution (
        id SERIAL PRIMARY KEY,
        concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
        generation INTEGER NOT NULL,
        fitness_score FLOAT NOT NULL,
        mutation_history JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memetic_evolution_concept_id ON memetic_evolution (concept_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS cultural_variants (
        id SERIAL PRIMARY KEY,
        concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
        culture TEXT NOT NULL,
        vector BYTEA,
        metadata JSONB,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_cultural_variants_concept_culture ON cultural_variants (concept_id, culture)
    """
]

# Sample data for testing
SAMPLE_DATA = [
    """
    INSERT INTO concepts (id, name, atomic_pattern, not_space, confidence, null_ratio, created_at, updated_at)
    VALUES (
        'c1b94f1d-f06e-4cb9-acf3-02ad6f4f757e',
        'Tree',
        '{"pattern": ["1", "&&"]}',
        '["not-tree", "rock", "building"]',
        0.95,
        0.05,
        NOW(),
        NOW()
    ) ON CONFLICT (id) DO NOTHING
    """,
    """
    INSERT INTO concepts (id, name, atomic_pattern, not_space, confidence, null_ratio, created_at, updated_at)
    VALUES (
        '7f8d9e0a-b1c2-4d3e-9f0a-1b2c3d4e5f6a',
        'Water',
        '{"pattern": ["1", "||"]}',
        '["fire", "earth", "air"]',
        0.92,
        0.08,
        NOW(),
        NOW()
    ) ON CONFLICT (id) DO NOTHING
    """,
    """
    INSERT INTO concepts (id, name, atomic_pattern, not_space, confidence, null_ratio, created_at, updated_at)
    VALUES (
        'a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d',
        'Love',
        '{"pattern": ["1", "&&", "||"]}',
        '["hate", "indifference", "apathy"]',
        0.85,
        0.15,
        NOW(),
        NOW()
    ) ON CONFLICT (id) DO NOTHING
    """
]

async def init_database(drop_existing=False, add_sample_data=False):
    """Initialize the database"""
    print(f"Connecting to database: {settings.database_url}")
    
    # Create database connection
    db = Database(settings.database_url)
    await db.connect()
    
    try:
        # Drop existing tables if requested
        if drop_existing:
            print("Dropping existing tables...")
            await db.execute("DROP TABLE IF EXISTS cultural_variants CASCADE")
            await db.execute("DROP TABLE IF EXISTS memetic_evolution CASCADE")
            await db.execute("DROP TABLE IF EXISTS concept_access CASCADE")
            await db.execute("DROP TABLE IF EXISTS concepts CASCADE")
            await db.execute("DROP TABLE IF EXISTS users CASCADE")
        
        # Create tables
        print("Creating tables...")
        for statement in CREATE_TABLES:
            await db.execute(statement)
        
        # Add sample data if requested
        if add_sample_data:
            print("Adding sample data...")
            for statement in SAMPLE_DATA:
                await db.execute(statement)
        
        print("Database initialization complete!")
    
    finally:
        # Close database connection
        await db.disconnect()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Initialize LEXICON database")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables")
    parser.add_argument("--sample", action="store_true", help="Add sample data")
    
    args = parser.parse_args()
    
    # Run database initialization
    asyncio.run(init_database(drop_existing=args.drop, add_sample_data=args.sample))

if __name__ == "__main__":
    main()
