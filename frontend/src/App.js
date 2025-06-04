import './App.css';

import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';

import AboutPage from './pages/AboutPage';
import COREEPage from './pages/COREEPage';
import DefinitionsPage from './pages/DefinitionsPage';
import Footer from './components/Footer';
import Header from './components/Header';
import HomePage from './pages/HomePage';
import React from 'react';
import VisualizationPage from './pages/VisualizationPage';

// Components



// Pages





function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/definitions" element={<DefinitionsPage />} />
            <Route path="/visualization" element={<VisualizationPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/coree" element={<COREEPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
