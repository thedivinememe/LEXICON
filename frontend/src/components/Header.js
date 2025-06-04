import { Link } from 'react-router-dom';
import React from 'react';

const Header = () => {
  return (
    <header className="header">
      <div className="container">
        <h1>LEXICON</h1>
        <p>Memetic Atomic Dictionary with Vectorized Objects</p>
        <nav>
          <ul style={{ listStyle: 'none', display: 'flex', justifyContent: 'center', gap: '20px' }}>
            <li><Link to="/" style={{ color: 'white', textDecoration: 'none' }}>Home</Link></li>
            <li><Link to="/definitions" style={{ color: 'white', textDecoration: 'none' }}>Definitions</Link></li>
            <li><Link to="/visualization" style={{ color: 'white', textDecoration: 'none' }}>Visualization</Link></li>
            <li><Link to="/about" style={{ color: 'white', textDecoration: 'none' }}>About</Link></li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
