-- Migration: 003_spherical.sql
-- Adds tables for the spherical universe system

-- Create table for spherical concepts
CREATE TABLE IF NOT EXISTS spherical_concepts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    r FLOAT NOT NULL,
    theta FLOAT NOT NULL,
    phi FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on concept name
CREATE INDEX IF NOT EXISTS idx_spherical_concepts_name ON spherical_concepts(name);

-- Create table for spherical relationships
CREATE TABLE IF NOT EXISTS spherical_relationships (
    id SERIAL PRIMARY KEY,
    concept1_id INTEGER NOT NULL REFERENCES spherical_concepts(id) ON DELETE CASCADE,
    concept2_id INTEGER NOT NULL REFERENCES spherical_concepts(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(concept1_id, concept2_id)
);

-- Create indexes on relationship columns
CREATE INDEX IF NOT EXISTS idx_spherical_relationships_concept1 ON spherical_relationships(concept1_id);
CREATE INDEX IF NOT EXISTS idx_spherical_relationships_concept2 ON spherical_relationships(concept2_id);
CREATE INDEX IF NOT EXISTS idx_spherical_relationships_type ON spherical_relationships(relationship_type);

-- Create table for type hierarchies
CREATE TABLE IF NOT EXISTS spherical_type_hierarchies (
    id SERIAL PRIMARY KEY,
    concept_id INTEGER NOT NULL REFERENCES spherical_concepts(id) ON DELETE CASCADE,
    bottom_type VARCHAR(255) NOT NULL,
    top_type VARCHAR(255) NOT NULL,
    unified_type VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(concept_id)
);

-- Create index on type hierarchy concept
CREATE INDEX IF NOT EXISTS idx_spherical_type_hierarchies_concept ON spherical_type_hierarchies(concept_id);

-- Create table for middle types in type hierarchies
CREATE TABLE IF NOT EXISTS spherical_middle_types (
    id SERIAL PRIMARY KEY,
    hierarchy_id INTEGER NOT NULL REFERENCES spherical_type_hierarchies(id) ON DELETE CASCADE,
    type_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hierarchy_id, type_name)
);

-- Create index on middle type hierarchy
CREATE INDEX IF NOT EXISTS idx_spherical_middle_types_hierarchy ON spherical_middle_types(hierarchy_id);

-- Create table for subtype relationships in type hierarchies
CREATE TABLE IF NOT EXISTS spherical_subtype_relationships (
    id SERIAL PRIMARY KEY,
    hierarchy_id INTEGER NOT NULL REFERENCES spherical_type_hierarchies(id) ON DELETE CASCADE,
    subtype VARCHAR(255) NOT NULL,
    supertype VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hierarchy_id, subtype, supertype)
);

-- Create indexes on subtype relationship columns
CREATE INDEX IF NOT EXISTS idx_spherical_subtype_relationships_hierarchy ON spherical_subtype_relationships(hierarchy_id);
CREATE INDEX IF NOT EXISTS idx_spherical_subtype_relationships_subtype ON spherical_subtype_relationships(subtype);
CREATE INDEX IF NOT EXISTS idx_spherical_subtype_relationships_supertype ON spherical_subtype_relationships(supertype);

-- Create table for type boundaries
CREATE TABLE IF NOT EXISTS spherical_type_boundaries (
    id SERIAL PRIMARY KEY,
    hierarchy_id INTEGER NOT NULL REFERENCES spherical_type_hierarchies(id) ON DELETE CASCADE,
    type_name VARCHAR(255) NOT NULL,
    r FLOAT NOT NULL,
    theta FLOAT NOT NULL,
    phi FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hierarchy_id, type_name)
);

-- Create indexes on type boundary columns
CREATE INDEX IF NOT EXISTS idx_spherical_type_boundaries_hierarchy ON spherical_type_boundaries(hierarchy_id);
CREATE INDEX IF NOT EXISTS idx_spherical_type_boundaries_type ON spherical_type_boundaries(type_name);

-- Create table for visualizations
CREATE TABLE IF NOT EXISTS spherical_visualizations (
    id SERIAL PRIMARY KEY,
    visualization_type VARCHAR(50) NOT NULL,
    concept_id INTEGER REFERENCES spherical_concepts(id) ON DELETE CASCADE,
    file_path VARCHAR(255) NOT NULL,
    url VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes on visualization columns
CREATE INDEX IF NOT EXISTS idx_spherical_visualizations_type ON spherical_visualizations(visualization_type);
CREATE INDEX IF NOT EXISTS idx_spherical_visualizations_concept ON spherical_visualizations(concept_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to update updated_at timestamp
CREATE TRIGGER update_spherical_concepts_updated_at
BEFORE UPDATE ON spherical_concepts
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_spherical_relationships_updated_at
BEFORE UPDATE ON spherical_relationships
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_spherical_type_hierarchies_updated_at
BEFORE UPDATE ON spherical_type_hierarchies
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_spherical_middle_types_updated_at
BEFORE UPDATE ON spherical_middle_types
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_spherical_subtype_relationships_updated_at
BEFORE UPDATE ON spherical_subtype_relationships
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_spherical_type_boundaries_updated_at
BEFORE UPDATE ON spherical_type_boundaries
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_spherical_visualizations_updated_at
BEFORE UPDATE ON spherical_visualizations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Create view for concept relationships
CREATE OR REPLACE VIEW spherical_concept_relationships AS
SELECT
    c1.name AS concept1_name,
    c2.name AS concept2_name,
    r.relationship_type
FROM
    spherical_relationships r
    JOIN spherical_concepts c1 ON r.concept1_id = c1.id
    JOIN spherical_concepts c2 ON r.concept2_id = c2.id;

-- Create view for type hierarchy details
CREATE OR REPLACE VIEW spherical_type_hierarchy_details AS
SELECT
    c.name AS concept_name,
    h.bottom_type,
    h.top_type,
    h.unified_type,
    array_agg(DISTINCT m.type_name) AS middle_types
FROM
    spherical_type_hierarchies h
    JOIN spherical_concepts c ON h.concept_id = c.id
    LEFT JOIN spherical_middle_types m ON h.id = m.hierarchy_id
GROUP BY
    c.name, h.bottom_type, h.top_type, h.unified_type;

-- Create view for type boundaries
CREATE OR REPLACE VIEW spherical_type_boundary_details AS
SELECT
    c.name AS concept_name,
    b.type_name,
    b.r,
    b.theta,
    b.phi
FROM
    spherical_type_boundaries b
    JOIN spherical_type_hierarchies h ON b.hierarchy_id = h.id
    JOIN spherical_concepts c ON h.concept_id = c.id;

-- Create view for visualization details
CREATE OR REPLACE VIEW spherical_visualization_details AS
SELECT
    v.id,
    v.visualization_type,
    c.name AS concept_name,
    v.file_path,
    v.url,
    v.created_at
FROM
    spherical_visualizations v
    LEFT JOIN spherical_concepts c ON v.concept_id = c.id;
