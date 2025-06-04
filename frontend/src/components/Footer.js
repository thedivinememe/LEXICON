import React from 'react';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <p>&copy; {new Date().getFullYear()} LEXICON Project</p>
        <p>
          <a href="https://github.com/yourusername/lexicon" style={{ color: 'white', marginRight: '10px' }}>GitHub</a>
          <a href="/docs" style={{ color: 'white', marginRight: '10px' }}>Documentation</a>
          <a href="/api/docs" style={{ color: 'white' }}>API</a>
        </p>
      </div>
    </footer>
  );
};

export default Footer;
