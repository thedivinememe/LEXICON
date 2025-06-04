import { Link } from 'react-router-dom';
import React from 'react';

const HomePage = () => {
  return (
    <div className="container">
      <section style={{ marginTop: '40px', textAlign: 'center' }}>
        <h2>Welcome to LEXICON</h2>
        <p>
          LEXICON is a Memetic Atomic Dictionary with Vectorized Objects, designed to explore
          and understand the relationships between fundamental concepts.
        </p>
        
        <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '40px' }}>
          <div style={{ 
            padding: '20px', 
            backgroundColor: '#fff', 
            borderRadius: '8px', 
            boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
            width: '300px'
          }}>
            <h3>Core Definitions</h3>
            <p>Explore the atomic definitions that form the foundation of our conceptual framework.</p>
            <Link to="/definitions" style={{ 
              display: 'inline-block', 
              padding: '10px 20px', 
              backgroundColor: '#282c34', 
              color: 'white', 
              textDecoration: 'none',
              borderRadius: '4px',
              marginTop: '10px'
            }}>
              View Definitions
            </Link>
          </div>
          
          <div style={{ 
            padding: '20px', 
            backgroundColor: '#fff', 
            borderRadius: '8px', 
            boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
            width: '300px'
          }}>
            <h3>Spherical Visualization</h3>
            <p>Visualize concepts and their relationships in a multidimensional spherical space.</p>
            <Link to="/visualization" style={{ 
              display: 'inline-block', 
              padding: '10px 20px', 
              backgroundColor: '#282c34', 
              color: 'white', 
              textDecoration: 'none',
              borderRadius: '4px',
              marginTop: '10px'
            }}>
              Explore Visualization
            </Link>
          </div>
        </div>
      </section>
      
      <section style={{ marginTop: '60px', textAlign: 'center' }}>
        <h2>About the Project</h2>
        <p>
          LEXICON is an open-source project that aims to create a comprehensive framework for 
          understanding and relating fundamental concepts across disciplines.
        </p>
        <Link to="/about" style={{ 
          display: 'inline-block', 
          padding: '10px 20px', 
          backgroundColor: '#282c34', 
          color: 'white', 
          textDecoration: 'none',
          borderRadius: '4px',
          marginTop: '10px'
        }}>
          Learn More
        </Link>
      </section>
    </div>
  );
};

export default HomePage;
