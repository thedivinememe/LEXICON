# LEXICON Deployment Guide

This document provides detailed instructions for deploying the LEXICON application to Heroku (backend) and GitHub Pages (frontend).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setting Up GitHub Repository Secrets](#setting-up-github-repository-secrets)
3. [Backend Deployment (Heroku)](#backend-deployment-heroku)
4. [Frontend Deployment (GitHub Pages)](#frontend-deployment-github-pages)
5. [Environment Variables](#environment-variables)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

Before you begin, make sure you have:

- A GitHub account with GitHub Actions enabled
- A Heroku account
- The Heroku CLI installed locally
- Git installed locally
- Node.js and npm installed (for frontend development)
- Python 3.10+ installed (for backend development)

## Setting Up GitHub Repository Secrets

To enable automated deployments via GitHub Actions, you need to set up the following secrets in your GitHub repository:

1. Go to your GitHub repository
2. Click on "Settings" > "Secrets and variables" > "Actions"
3. Click on "New repository secret"
4. Add the following secrets:

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `HEROKU_API_KEY` | Your Heroku API key | From [Heroku Account Settings](https://dashboard.heroku.com/account) under "API Key" |
| `HEROKU_APP_NAME` | Your Heroku application name | The name of your Heroku app (e.g., "lexicon-api") |
| `HEROKU_EMAIL` | Your Heroku account email | The email address associated with your Heroku account |
| `GITHUB_ORG` | (Optional) Your GitHub username or organization | Your GitHub username or organization name |
| `GITHUB_REPO` | (Optional) Your GitHub repository name | The name of your GitHub repository |
| `API_URL` | (Optional) The full URL to your backend API | `https://your-heroku-app-name.herokuapp.com/api/v1` |

## Required Heroku Information

To successfully deploy the LEXICON application to Heroku, you'll need the following information and resources:

1. **Heroku Account**: You need a Heroku account. If you don't have one, sign up at [heroku.com](https://heroku.com).

2. **Heroku API Key**: This is required for GitHub Actions to deploy to Heroku on your behalf.
   - Go to [Heroku Account Settings](https://dashboard.heroku.com/account)
   - Scroll down to the "API Key" section
   - Click "Reveal" to see your API key or "Regenerate API Key" if you need a new one
   - Use this value for the `HEROKU_API_KEY` secret in GitHub

3. **Heroku Application**: You need to create a Heroku application or use an existing one.
   - The application name must be unique across all of Heroku
   - This name will be used for the `HEROKU_APP_NAME` secret in GitHub
   - Your application will be accessible at `https://your-app-name.herokuapp.com`

4. **Heroku Add-ons**: The LEXICON application requires the following Heroku add-ons:
   - **Heroku Postgres**: For the database (can use the free hobby-dev tier for testing)
   - **Heroku Redis** (optional but recommended): For caching (can use the free hobby-dev tier for testing)

5. **Heroku Config Variables**: The following environment variables should be set in your Heroku application:
   - `SECRET_KEY`: A secure random string for JWT signing
   - `ENVIRONMENT`: Set to "production"
   - `GITHUB_ORG`: Your GitHub username or organization
   - `GITHUB_REPO`: Your GitHub repository name
   - `GITHUB_PAGES_URL`: The URL to your GitHub Pages site (e.g., `https://your-github-username.github.io/your-repo-name`)
   - `OPENAI_API_KEY` (optional): If you want to use the COREE interface with OpenAI integration

You can set these variables either:
- Through the Heroku Dashboard: Go to your app > Settings > Config Vars
- Using the Heroku CLI: `heroku config:set VARIABLE_NAME=value --app your-app-name`
- Automatically via GitHub Actions (as configured in the deployment workflow)

## Backend Deployment (Heroku)

### Automated Deployment

The backend is automatically deployed to Heroku when changes are pushed to the `main` branch (excluding frontend files). The deployment workflow:

1. Checks out the code
2. Sets up Python
3. Installs dependencies
4. Checks for secrets and sensitive information
5. Runs tests
6. Deploys to Heroku
7. Runs database migrations
8. Sets environment variables

The security check step uses the `check_for_secrets.py` script to scan the codebase for potential API keys, passwords, or other sensitive information that should not be committed to the repository. If any potential secrets are found, the deployment will fail, preventing accidental exposure of sensitive credentials.

### Using the Deployment Scripts

The LEXICON project includes deployment scripts that simplify the deployment process by creating a temporary directory with only the necessary files for Heroku deployment. This helps avoid the 500MB slug size limit.

#### For Windows Users:

```bash
# Run the deployment script
scripts\deploy_to_heroku.bat
```

#### For Linux/macOS Users:

```bash
# Make the script executable
chmod +x scripts/deploy_to_heroku.sh

# Run the deployment script
./scripts/deploy_to_heroku.sh
```

The script will:
1. Create a temporary directory
2. Copy only the necessary backend files
3. Create a `.slugignore` file to exclude large files
4. Initialize a Git repository
5. Connect to your Heroku app
6. Deploy to Heroku
7. Run database migrations

### Manual Deployment

If you prefer to deploy manually without using the scripts:

```bash
# Login to Heroku
heroku login

# Create a new Heroku app if you don't have one
heroku create your-app-name

# Add PostgreSQL addon
heroku addons:create heroku-postgresql

# Add Redis addon (optional but recommended)
heroku addons:create heroku-redis

# Set environment variables
heroku config:set ENVIRONMENT=production
heroku config:set SECRET_KEY=your_secure_random_string
heroku config:set GITHUB_ORG=your-github-username
heroku config:set GITHUB_REPO=your-repo-name
heroku config:set GITHUB_PAGES_URL=https://your-github-username.github.io/your-repo-name

# Deploy
git push heroku main

# Run database migrations
heroku run python -m scripts.init_db
```

### Dealing with Slug Size Limits

If you encounter the "Compiled slug size is too large" error, you can:

1. Use the provided deployment scripts which handle this automatically
2. Create a `.slugignore` file in your project root to exclude unnecessary files
3. Use Git subtree to push only a subset of your repository:
   ```bash
   git subtree push --prefix src heroku main
   ```

### Verifying the Deployment

To verify that your backend is deployed correctly:

1. Visit `https://your-heroku-app-name.herokuapp.com/health`
2. You should see a JSON response with `"status": "healthy"`

## Frontend Deployment (GitHub Pages)

### Automated Deployment

The frontend is automatically deployed to GitHub Pages when changes are pushed to the `main` branch in the `frontend` directory. The deployment workflow:

1. Checks out the code
2. Sets up Node.js
3. Installs dependencies
4. Sets the API URL environment variable
5. Builds the React application
6. Updates the COREE interface with the correct API URL
7. Deploys to the `gh-pages` branch

The COREE interface (frontend/public/coree.html) is a standalone HTML file that communicates with the backend API. During deployment, the placeholder `{{REACT_APP_API_URL}}` in this file is replaced with the actual API URL to ensure it can connect to the backend.

### Using the Deployment Scripts

The LEXICON project includes deployment scripts that simplify the frontend deployment process.

#### For Windows Users:

```bash
# Run the deployment script
scripts\deploy_to_github_pages.bat
```

#### For Linux/macOS Users:

```bash
# Make the script executable
chmod +x scripts/deploy_to_github_pages.sh

# Run the deployment script
./scripts/deploy_to_github_pages.sh
```

The script will:
1. Navigate to the frontend directory
2. Prompt for your Heroku app name to set the API URL
3. Install dependencies
4. Build the React application
5. Deploy to GitHub Pages

### Manual Deployment

If you prefer to deploy manually without using the scripts:

```bash
# Set the API URL in .env.local
echo "REACT_APP_API_URL=https://your-heroku-app.herokuapp.com/api/v1" > frontend/.env.local

# Install dependencies
cd frontend
npm install

# Build the frontend
npm run build

# Install gh-pages if not already installed
npm install -g gh-pages

# Deploy to GitHub Pages
npx gh-pages -d build
```

### Verifying the Deployment

To verify that your frontend is deployed correctly:

1. Visit `https://your-github-username.github.io/your-repo-name`
2. You should see the LEXICON frontend application
3. Check the browser console to ensure it's connecting to the correct API URL

## Environment Variables

### Backend (Heroku) Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Set by Heroku PostgreSQL addon |
| `REDIS_URL` | Redis connection string | Set by Heroku Redis addon |
| `SECRET_KEY` | Secret key for JWT signing | Must be set manually |
| `JWT_ALGORITHM` | Algorithm for JWT signing | HS256 |
| `ENVIRONMENT` | Application environment | Should be set to "production" |
| `DEBUG` | Enable debug mode | Should be set to "false" in production |
| `GITHUB_ORG` | GitHub username or organization | Must be set manually |
| `GITHUB_REPO` | GitHub repository name | Must be set manually |
| `GITHUB_PAGES_URL` | URL to GitHub Pages site | Must be set manually |

### Frontend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_API_URL` | URL to the backend API | Set during build by GitHub Actions |

## Troubleshooting

### Common Issues

#### Backend Deployment Issues

1. **Database Migration Failures**
   - Check Heroku logs: `heroku logs --tail`
   - Manually run migrations: `heroku run python -m scripts.init_db`

2. **Application Crashes**
   - Check Heroku logs: `heroku logs --tail`
   - Verify environment variables: `heroku config`

3. **CORS Issues**
   - Ensure `GITHUB_PAGES_URL` is set correctly in Heroku config
   - Check that the frontend is using the correct API URL

#### Frontend Deployment Issues

1. **API Connection Failures**
   - Open browser console and check for CORS errors
   - Verify that `REACT_APP_API_URL` is set correctly
   - Ensure the backend CORS configuration includes the GitHub Pages URL

2. **Blank Page After Deployment**
   - Check if the GitHub Pages site is published (Settings > Pages)
   - Verify that the build process completed successfully in GitHub Actions

3. **Old Content Being Served**
   - Clear browser cache
   - Check if the GitHub Pages deployment completed successfully

### Getting Help

If you encounter issues not covered here:

1. Check the GitHub Actions logs for detailed error messages
2. Review the Heroku logs: `heroku logs --tail`
3. Open an issue in the GitHub repository with detailed information about the problem
