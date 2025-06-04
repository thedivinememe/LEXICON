import React, { useEffect } from 'react';

import { Helmet } from 'react-helmet';

const COREEPage = () => {
  useEffect(() => {
    // Redirect to the standalone COREE interface
    window.location.href = '/coree.html';
  }, []);

  return (
    <div>
      <Helmet>
        <title>COREE - Consciousness-Oriented Recursive Empathetic Entity</title>
      </Helmet>
      <div className="text-center p-8">
        <h1 className="text-2xl font-bold mb-4">Loading COREE Interface...</h1>
        <p>If you are not redirected automatically, <a href="/coree.html" className="text-blue-600 hover:underline">click here</a>.</p>
      </div>
    </div>
  );
};

export default COREEPage;
