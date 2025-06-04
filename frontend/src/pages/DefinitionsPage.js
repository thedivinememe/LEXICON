import React, { useEffect, useState } from 'react';

import { coreDefinitionsApi } from '../services/api';

const DefinitionsPage = () => {
  const [definitions, setDefinitions] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDefinition, setSelectedDefinition] = useState(null);

  useEffect(() => {
    const fetchDefinitions = async () => {
      try {
        setLoading(true);
        const data = await coreDefinitionsApi.getAll();
        setDefinitions(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load definitions. Please try again later.');
        setLoading(false);
        console.error('Error fetching definitions:', err);
      }
    };

    fetchDefinitions();
  }, []);

  const handleDefinitionClick = (key) => {
    setSelectedDefinition(key);
  };

  const renderDefinitionDetails = (def) => {
    if (!def) return null;

    return (
      <div style={{ 
        backgroundColor: '#fff', 
        padding: '20px', 
        borderRadius: '8px', 
        boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
        marginTop: '20px'
      }}>
        <h3>{selectedDefinition}</h3>
        
        {def.atomic_pattern && (
          <div>
            <strong>Atomic Pattern:</strong> {def.atomic_pattern}
          </div>
        )}
        
        {def.not_space && def.not_space.length > 0 && (
          <div>
            <strong>Not Space:</strong> {def.not_space.join(', ')}
          </div>
        )}
        
        {def.and_relationships && def.and_relationships.length > 0 && (
          <div>
            <strong>AND Relationships:</strong> {def.and_relationships.map(r => `${r[0]} (${r[1]})`).join(', ')}
          </div>
        )}
        
        {def.or_relationships && def.or_relationships.length > 0 && (
          <div>
            <strong>OR Relationships:</strong> {def.or_relationships.map(r => `${r[0]} (${r[1]})`).join(', ')}
          </div>
        )}
        
        {def.not_relationships && def.not_relationships.length > 0 && (
          <div>
            <strong>NOT Relationships:</strong> {def.not_relationships.map(r => `${r[0]} (${r[1]})`).join(', ')}
          </div>
        )}
        
        {def.vector_properties && (
          <div>
            <strong>Vector Properties:</strong> {def.vector_properties}
          </div>
        )}
        
        {def.spherical_properties && (
          <div>
            <strong>Spherical Properties:</strong>
            <ul>
              {Object.entries(def.spherical_properties).map(([key, value]) => (
                <li key={key}>{key}: {value}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return <div className="container">Loading definitions...</div>;
  }

  if (error) {
    return <div className="container" style={{ color: 'red' }}>{error}</div>;
  }

  return (
    <div className="container">
      <h2 style={{ marginBottom: '20px' }}>Core Definitions</h2>
      
      <div style={{ display: 'flex', gap: '20px' }}>
        <div style={{ width: '30%' }}>
          <div style={{ 
            backgroundColor: '#fff', 
            padding: '20px', 
            borderRadius: '8px', 
            boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
          }}>
            <h3>Definitions</h3>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {Object.keys(definitions).map(key => (
                <li 
                  key={key} 
                  onClick={() => handleDefinitionClick(key)}
                  style={{ 
                    padding: '10px', 
                    cursor: 'pointer',
                    backgroundColor: selectedDefinition === key ? '#f0f0f0' : 'transparent',
                    borderRadius: '4px',
                    marginBottom: '5px'
                  }}
                >
                  {key}
                </li>
              ))}
            </ul>
          </div>
        </div>
        
        <div style={{ width: '70%' }}>
          {selectedDefinition ? 
            renderDefinitionDetails(definitions[selectedDefinition]) : 
            <div style={{ textAlign: 'center', marginTop: '50px' }}>
              <p>Select a definition from the list to view details</p>
            </div>
          }
        </div>
      </div>
    </div>
  );
};

export default DefinitionsPage;
