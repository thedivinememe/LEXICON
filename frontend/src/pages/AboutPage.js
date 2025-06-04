import React from 'react';

const AboutPage = () => {
  return (
    <div className="container">
      <h2>About LEXICON</h2>
      
      <section style={{ marginTop: '20px' }}>
        <h3>Project Overview</h3>
        <p>
          LEXICON (Memetic Atomic Dictionary with Vectorized Objects) is an open-source project 
          that aims to create a comprehensive framework for understanding and relating fundamental 
          concepts across disciplines. By representing concepts as vectorized objects in a 
          multidimensional space, LEXICON enables the exploration of relationships between ideas 
          in a mathematically rigorous way.
        </p>
      </section>
      
      <section style={{ marginTop: '20px' }}>
        <h3>Core Concepts</h3>
        <p>
          At the heart of LEXICON are the core definitions - atomic concepts that form the 
          foundation of our conceptual framework. These include fundamental ideas like existence, 
          pattern, relationship, consciousness, and more. Each concept is defined not just by 
          description, but by its relationships to other concepts and its position in the 
          conceptual vector space.
        </p>
      </section>
      
      <section style={{ marginTop: '20px' }}>
        <h3>Spherical Representation</h3>
        <p>
          LEXICON uses a spherical representation system to visualize concepts and their 
          relationships. This approach allows for intuitive understanding of conceptual 
          distances, clusters, and hierarchies. The spherical model provides a unique way 
          to explore how concepts relate to each other in multiple dimensions.
        </p>
      </section>
      
      <section style={{ marginTop: '20px' }}>
        <h3>Technology Stack</h3>
        <ul>
          <li><strong>Backend:</strong> Python with FastAPI, PostgreSQL, Redis</li>
          <li><strong>Neural/ML:</strong> PyTorch, Transformers, FAISS for vector search</li>
          <li><strong>Frontend:</strong> React, Axios</li>
          <li><strong>Visualization:</strong> Plotly, D3.js</li>
          <li><strong>Deployment:</strong> Docker, Heroku, GitHub Pages</li>
        </ul>
      </section>
      
      <section style={{ marginTop: '20px' }}>
        <h3>Contributing</h3>
        <p>
          LEXICON is an open-source project and welcomes contributions from the community. 
          Whether you're interested in philosophy, linguistics, computer science, or data 
          visualization, there are many ways to get involved.
        </p>
        <p>
          Visit our <a href="https://github.com/yourusername/lexicon" style={{ color: '#282c34' }}>GitHub repository</a> to 
          learn more about contributing to the project.
        </p>
      </section>
      
      <section style={{ marginTop: '20px', marginBottom: '40px' }}>
        <h3>Contact</h3>
        <p>
          For questions, suggestions, or collaboration opportunities, please reach out to us at:
        </p>
        <p>
          <a href="mailto:contact@lexicon-project.org" style={{ color: '#282c34' }}>contact@lexicon-project.org</a>
        </p>
      </section>
    </div>
  );
};

export default AboutPage;
