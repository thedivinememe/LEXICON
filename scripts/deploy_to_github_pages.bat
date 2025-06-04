@echo off
echo ===== LEXICON GitHub Pages Deployment Script =====
echo This script will build and deploy the frontend to GitHub Pages

REM Navigate to the frontend directory
echo Navigating to frontend directory...
cd frontend

REM Set the API URL in the environment
echo Setting API URL...
set /p HEROKU_APP_NAME=Enter your Heroku app name (e.g., lexicon-api): 
echo REACT_APP_API_URL=https://%HEROKU_APP_NAME%.herokuapp.com/api/v1 > .env.local
echo API URL set to: https://%HEROKU_APP_NAME%.herokuapp.com/api/v1

REM Install dependencies
echo Installing dependencies...
call npm install

REM Build the frontend
echo Building the frontend...
call npm run build

REM Install gh-pages if not already installed
echo Installing gh-pages...
call npm install -g gh-pages

REM Deploy to GitHub Pages
echo Deploying to GitHub Pages...
call npx gh-pages -d build

echo ===== Deployment Complete =====
echo Your frontend should now be deployed to GitHub Pages.
echo.
echo To view your deployed frontend, visit:
echo https://your-github-username.github.io/LEXICON/
echo.
echo Note: It may take a few minutes for the changes to propagate.
echo.
echo If you haven't set up GitHub Pages yet, you need to:
echo 1. Go to your GitHub repository
echo 2. Click on "Settings" > "Pages"
echo 3. Under "Source", select "gh-pages" branch
echo 4. Click "Save"

REM Return to the original directory
cd ..
