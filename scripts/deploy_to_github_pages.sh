#!/bin/bash

echo "===== LEXICON GitHub Pages Deployment Script ====="
echo "This script will build and deploy the frontend to GitHub Pages"

# Navigate to the frontend directory
echo "Navigating to frontend directory..."
cd frontend

# Set the API URL in the environment
echo "Setting API URL..."
read -p "Enter your Heroku app name (e.g., lexicon-api): " HEROKU_APP_NAME
echo "REACT_APP_API_URL=https://$HEROKU_APP_NAME.herokuapp.com/api/v1" > .env.local
echo "API URL set to: https://$HEROKU_APP_NAME.herokuapp.com/api/v1"

# Install dependencies
echo "Installing dependencies..."
npm install

# Build the frontend
echo "Building the frontend..."
npm run build

# Replace the API_URL placeholder in coree.html
echo "Updating COREE interface with API URL..."
sed -i "s|{{REACT_APP_API_URL}}|https://$HEROKU_APP_NAME.herokuapp.com/api/v1|g" build/coree.html

# Install gh-pages if not already installed
echo "Installing gh-pages..."
npm install -g gh-pages

# Deploy to GitHub Pages
echo "Deploying to GitHub Pages..."
npx gh-pages -d build

echo "===== Deployment Complete ====="
echo "Your frontend should now be deployed to GitHub Pages."
echo ""
echo "To view your deployed frontend, visit:"
echo "https://your-github-username.github.io/LEXICON/"
echo ""
echo "Note: It may take a few minutes for the changes to propagate."
echo ""
echo "If you haven't set up GitHub Pages yet, you need to:"
echo "1. Go to your GitHub repository"
echo "2. Click on \"Settings\" > \"Pages\""
echo "3. Under \"Source\", select \"gh-pages\" branch"
echo "4. Click \"Save\""

# Return to the original directory
cd ..
