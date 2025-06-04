import axios from 'axios';

// Get the API URL from environment variables or use a default
// In production, this will be set to the Heroku backend URL via REACT_APP_API_URL
// This is configured in the GitHub Actions workflow
const API_URL = process.env.REACT_APP_API_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'https://lexicon-api.herokuapp.com/api/v1' // Fallback production URL if not set
    : 'http://localhost:8000/api/v1');

console.log('Using API URL:', API_URL);

// Create an axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions for core definitions
export const coreDefinitionsApi = {
  // Get all core definitions
  getAll: async () => {
    try {
      const response = await api.get('/core-definitions');
      return response.data;
    } catch (error) {
      console.error('Error fetching core definitions:', error);
      throw error;
    }
  },
  
  // Get a specific core definition by name
  getByName: async (name) => {
    try {
      const response = await api.get(`/core-definitions/${name}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching core definition ${name}:`, error);
      throw error;
    }
  },
  
  // Create a new core definition
  create: async (definition) => {
    try {
      const response = await api.post('/core-definitions', definition);
      return response.data;
    } catch (error) {
      console.error('Error creating core definition:', error);
      throw error;
    }
  },
  
  // Update an existing core definition
  update: async (name, definition) => {
    try {
      const response = await api.put(`/core-definitions/${name}`, definition);
      return response.data;
    } catch (error) {
      console.error(`Error updating core definition ${name}:`, error);
      throw error;
    }
  },
  
  // Delete a core definition
  delete: async (name) => {
    try {
      const response = await api.delete(`/core-definitions/${name}`);
      return response.data;
    } catch (error) {
      console.error(`Error deleting core definition ${name}:`, error);
      throw error;
    }
  }
};

// API functions for spherical visualization
export const sphericalApi = {
  // Get spherical representation of concepts
  getSphericalData: async () => {
    try {
      const response = await api.get('/spherical/concepts');
      return response.data;
    } catch (error) {
      console.error('Error fetching spherical data:', error);
      throw error;
    }
  },
  
  // Get relationships between concepts in spherical space
  getRelationships: async () => {
    try {
      const response = await api.get('/spherical/relationships');
      return response.data;
    } catch (error) {
      console.error('Error fetching spherical relationships:', error);
      throw error;
    }
  }
};

export default api;
