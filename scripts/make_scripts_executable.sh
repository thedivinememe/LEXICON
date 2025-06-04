#!/bin/bash

# Make deployment scripts executable
echo "Making scripts executable..."
chmod +x scripts/deploy_to_heroku.sh
chmod +x scripts/deploy_to_github_pages.sh
chmod +x scripts/check_for_secrets.sh

echo "Scripts are now executable. You can run them with:"
echo "./scripts/deploy_to_heroku.sh"
echo "./scripts/deploy_to_github_pages.sh"
echo "./scripts/check_for_secrets.sh"
