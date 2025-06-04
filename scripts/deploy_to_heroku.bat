@echo off
echo ===== LEXICON Heroku Deployment Script =====
echo This script will create a temporary directory with only the necessary files for Heroku deployment

REM Create a temporary directory for Heroku deployment
echo Creating temporary deployment directory...
mkdir ..\lexicon-heroku-deploy
cd ..\lexicon-heroku-deploy

REM Initialize a new Git repository
echo Initializing Git repository...
git init

REM Copy only the necessary backend files
echo Copying backend files...
xcopy /E /I /Y ..\LEXICON\src src
xcopy /E /I /Y ..\LEXICON\templates templates
xcopy /E /I /Y ..\LEXICON\scripts\init_db.py scripts\
copy ..\LEXICON\app.py .
copy ..\LEXICON\Procfile .
copy ..\LEXICON\runtime.txt .
copy ..\LEXICON\requirements.txt .
copy ..\LEXICON\.env.example .env

REM Create the .slugignore file
echo Creating .slugignore file...
echo # Directories to exclude from Heroku deployment > .slugignore
echo visualizations/ >> .slugignore
echo models/ >> .slugignore
echo data/ >> .slugignore
echo tests/ >> .slugignore
echo docs/ >> .slugignore
echo docker/ >> .slugignore
echo frontend/ >> .slugignore
echo .github/ >> .slugignore
echo .git/ >> .slugignore
echo .vscode/ >> .slugignore
echo __pycache__/ >> .slugignore
echo *.egg-info/ >> .slugignore
echo. >> .slugignore
echo # Large files >> .slugignore
echo *.pkl >> .slugignore
echo *.h5 >> .slugignore
echo *.model >> .slugignore
echo *.bin >> .slugignore
echo *.vec >> .slugignore
echo *.npy >> .slugignore
echo *.npz >> .slugignore
echo *.zip >> .slugignore
echo *.tar.gz >> .slugignore
echo *.csv >> .slugignore
echo *.tsv >> .slugignore
echo *.json >> .slugignore
echo *.html >> .slugignore
echo *.png >> .slugignore
echo *.jpg >> .slugignore
echo *.jpeg >> .slugignore
echo *.gif >> .slugignore
echo *.svg >> .slugignore
echo *.pdf >> .slugignore
echo. >> .slugignore
echo # Development files >> .slugignore
echo *.ipynb >> .slugignore
echo *.pyc >> .slugignore
echo .coverage >> .slugignore
echo htmlcov/ >> .slugignore
echo .pytest_cache/ >> .slugignore
echo .tox/ >> .slugignore
echo .hypothesis/ >> .slugignore
echo .mypy_cache/ >> .slugignore
echo .dmypy.json >> .slugignore
echo dmypy.json >> .slugignore
echo .pyre/ >> .slugignore
echo. >> .slugignore
echo # Node.js files >> .slugignore
echo node_modules/ >> .slugignore

REM Add and commit the files
echo Adding files to Git...
git add .
git commit -m "Initial Heroku deployment"

REM Connect to your Heroku app
echo Connecting to Heroku app...
set /p HEROKU_APP_NAME=Enter your Heroku app name (e.g., lexicon-api): 
heroku git:remote -a %HEROKU_APP_NAME%

REM Push to Heroku
echo Pushing to Heroku...
git push heroku main

REM Run database migrations
echo Running database migrations...
heroku run python -m scripts.init_db --app %HEROKU_APP_NAME%

echo ===== Deployment Complete =====
echo Your backend should now be deployed to Heroku at https://%HEROKU_APP_NAME%.herokuapp.com
echo.
echo To verify the deployment, visit:
echo https://%HEROKU_APP_NAME%.herokuapp.com/health
echo.
echo To view the logs, run:
echo heroku logs --tail --app %HEROKU_APP_NAME%
echo.
echo To set environment variables, run:
echo heroku config:set VARIABLE_NAME=value --app %HEROKU_APP_NAME%
echo.
echo Remember to deploy your frontend to GitHub Pages separately.

REM Return to the original directory
cd ..\LEXICON
