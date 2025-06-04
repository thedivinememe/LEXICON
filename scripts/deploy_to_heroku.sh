#!/bin/bash

echo "===== LEXICON Heroku Deployment Script ====="
echo "This script will create a temporary directory with only the necessary files for Heroku deployment"

# Create a temporary directory for Heroku deployment
echo "Creating temporary deployment directory..."
mkdir -p ../lexicon-heroku-deploy
cd ../lexicon-heroku-deploy

# Initialize a new Git repository
echo "Initializing Git repository..."
git init

# Copy only the necessary backend files
echo "Copying backend files..."
mkdir -p scripts
cp -r ../LEXICON/src .
cp -r ../LEXICON/templates .
cp ../LEXICON/scripts/init_db.py scripts/
cp ../LEXICON/app.py .
cp ../LEXICON/Procfile .
cp ../LEXICON/runtime.txt .
cp ../LEXICON/requirements.txt .
cp ../LEXICON/.env.example .env

# Create the .slugignore file
echo "Creating .slugignore file..."
cat > .slugignore << EOL
# Directories to exclude from Heroku deployment
visualizations/
models/
data/
tests/
docs/
docker/
frontend/
.github/
.git/
.vscode/
__pycache__/
*.egg-info/

# Large files
*.pkl
*.h5
*.model
*.bin
*.vec
*.npy
*.npz
*.zip
*.tar.gz
*.csv
*.tsv
*.json
*.html
*.png
*.jpg
*.jpeg
*.gif
*.svg
*.pdf

# Development files
*.ipynb
*.pyc
.coverage
htmlcov/
.pytest_cache/
.tox/
.hypothesis/
.mypy_cache/
.dmypy.json
dmypy.json
.pyre/

# Node.js files
node_modules/
EOL

# Add and commit the files
echo "Adding files to Git..."
git add .
git commit -m "Initial Heroku deployment"

# Connect to your Heroku app
echo "Connecting to Heroku app..."
read -p "Enter your Heroku app name (e.g., lexicon-api): " HEROKU_APP_NAME
heroku git:remote -a $HEROKU_APP_NAME

# Push to Heroku
echo "Pushing to Heroku..."
git push heroku main

# Run database migrations
echo "Running database migrations..."
heroku run python -m scripts.init_db --app $HEROKU_APP_NAME

echo "===== Deployment Complete ====="
echo "Your backend should now be deployed to Heroku at https://$HEROKU_APP_NAME.herokuapp.com"
echo ""
echo "To verify the deployment, visit:"
echo "https://$HEROKU_APP_NAME.herokuapp.com/health"
echo ""
echo "To view the logs, run:"
echo "heroku logs --tail --app $HEROKU_APP_NAME"
echo ""
echo "To set environment variables, run:"
echo "heroku config:set VARIABLE_NAME=value --app $HEROKU_APP_NAME"
echo ""
echo "Remember to deploy your frontend to GitHub Pages separately."

# Return to the original directory
cd ../LEXICON
