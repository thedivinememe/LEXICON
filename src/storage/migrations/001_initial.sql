-- Initial database schema for LEXICON
-- Creates tables for concepts, evolution history, and related data

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Concepts table
CREATE TABLE concepts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    vector FLOAT8[768] NOT NULL,
    null_ratio FLOAT NOT NULL,
    empathy_scores JSONB,
    negations TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Evolution history table
CREATE TABLE evolution_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    concept_id UUID REFERENCES concepts(id),
    generation INTEGER,
    fitness_score FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Concept access tracking
CREATE TABLE concept_access (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    concept_id UUID REFERENCES concepts(id),
    access_type VARCHAR(50) NOT NULL,
    accessed_at TIMESTAMP DEFAULT NOW(),
    user_id VARCHAR(255) NULL,
    context JSONB
);

-- Concept relationships
CREATE TABLE concept_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_concept_id UUID REFERENCES concepts(id),
    target_concept_id UUID REFERENCES concepts(id),
    relationship_type VARCHAR(50) NOT NULL,
    empathy_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_concept_id, target_concept_id, relationship_type)
);

-- Cultural variants
CREATE TABLE cultural_variants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    concept_id UUID REFERENCES concepts(id),
    cultural_context VARCHAR(100) NOT NULL,
    vector FLOAT8[768] NOT NULL,
    null_ratio FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(concept_id, cultural_context)
);

-- Memetic states
CREATE TABLE memetic_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    concept_id UUID REFERENCES concepts(id),
    generation INTEGER NOT NULL,
    fitness_score FLOAT NOT NULL,
    replication_count INTEGER DEFAULT 0,
    mutation_history JSONB,
    cultural_adaptations JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(concept_id, generation)
);

-- Indexes for performance
CREATE INDEX idx_concepts_name ON concepts(name);
CREATE INDEX idx_concepts_null_ratio ON concepts(null_ratio);
CREATE INDEX idx_evolution_history_concept_id ON evolution_history(concept_id);
CREATE INDEX idx_concept_access_concept_id ON concept_access(concept_id);
CREATE INDEX idx_concept_access_accessed_at ON concept_access(accessed_at);
CREATE INDEX idx_concept_relationships_source ON concept_relationships(source_concept_id);
CREATE INDEX idx_concept_relationships_target ON concept_relationships(target_concept_id);
CREATE INDEX idx_cultural_variants_concept_id ON cultural_variants(concept_id);
CREATE INDEX idx_memetic_states_concept_id ON memetic_states(concept_id);
CREATE INDEX idx_memetic_states_fitness ON memetic_states(fitness_score);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
CREATE TRIGGER update_concepts_updated_at
BEFORE UPDATE ON concepts
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Views for analytics
CREATE VIEW concept_evolution_metrics AS
SELECT 
    c.id,
    c.name,
    c.null_ratio,
    COUNT(e.id) AS evolution_count,
    MAX(e.generation) AS max_generation,
    AVG(e.fitness_score) AS avg_fitness,
    MAX(e.fitness_score) AS max_fitness
FROM 
    concepts c
LEFT JOIN 
    evolution_history e ON c.id = e.concept_id
GROUP BY 
    c.id, c.name, c.null_ratio;

CREATE VIEW concept_popularity AS
SELECT 
    c.id,
    c.name,
    COUNT(a.id) AS access_count,
    MAX(a.accessed_at) AS last_accessed
FROM 
    concepts c
LEFT JOIN 
    concept_access a ON c.id = a.concept_id
GROUP BY 
    c.id, c.name
ORDER BY 
    access_count DESC;

-- Initial system concepts (optional)
INSERT INTO concepts (name, vector, null_ratio, empathy_scores, negations)
VALUES (
    'existence',
    array_fill(0::float8, ARRAY[768]),
    0.1,
    '{"self_empathy": 1.0, "other_empathy": 0.5}',
    ARRAY['non-existence', 'void', 'nothingness']
);
