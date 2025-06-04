# Security Best Practices for LEXICON

This document outlines security best practices for the LEXICON project, particularly focusing on the management of sensitive credentials such as API keys.

## Managing API Keys and Sensitive Credentials

### Local Development

1. **Use Environment Variables**
   - Store all sensitive credentials in environment variables
   - Use the `.env` file for local development
   - The `.env` file is already included in `.gitignore` to prevent accidental commits

2. **Example Environment File**
   - The `.env.example` file serves as a template
   - It should NEVER contain actual credentials, only placeholders
   - Example: `OPENAI_API_KEY=your_openai_api_key_here`

3. **Setting Up Local Environment**
   - Copy `.env.example` to `.env`: `cp .env.example .env`
   - Replace placeholders with your actual credentials
   - Never commit the `.env` file to the repository

### Production Deployment

1. **Heroku**
   - Set environment variables using the Heroku dashboard or CLI:
     ```bash
     heroku config:set OPENAI_API_KEY=your_actual_key --app your-app-name
     ```
   - Verify settings: `heroku config --app your-app-name`

2. **GitHub Actions**
   - Store sensitive values as GitHub Secrets
   - Access them in workflows using `${{ secrets.SECRET_NAME }}`
   - Example in a workflow file:
     ```yaml
     env:
       OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
     ```

3. **Docker**
   - Pass environment variables to containers at runtime
   - Never build images with hardcoded credentials
   - Use Docker secrets for swarm deployments

## Preventing Credential Leaks

1. **Git Hooks**
   - Consider using pre-commit hooks to prevent committing sensitive data
   - Tools like `git-secrets` can scan for patterns that look like credentials

2. **Regular Audits**
   - Periodically review code and commits for accidentally exposed credentials
   - If credentials are exposed, revoke and rotate them immediately

3. **GitHub Push Protection**
   - GitHub's push protection feature helps prevent committing secrets
   - If you encounter a blocked push due to detected secrets:
     - Remove the secret from the commit
     - Make a new commit without the secret
     - Push again

4. **Credential Rotation**
   - Regularly rotate credentials, especially for production environments
   - Update all relevant deployment environments when rotating credentials

## What to Do If Credentials Are Exposed

1. **Revoke and Regenerate**
   - Immediately revoke the exposed credentials
   - Generate new credentials

2. **Check for Unauthorized Usage**
   - Monitor for any unauthorized usage of the exposed credentials
   - Check API usage logs for suspicious activity

3. **Review Git History**
   - If credentials were committed to the repository:
     - Do NOT try to remove them using `git filter-branch` or similar commands on a shared repository
     - Instead, consider the credentials compromised and rotate them
     - Add the credential patterns to `.gitignore` to prevent future occurrences

4. **Notify Relevant Parties**
   - If the exposure affects others, notify them promptly
   - Follow any required disclosure procedures for your organization

## Additional Resources

- [GitHub Documentation on Secret Scanning](https://docs.github.com/en/code-security/secret-scanning/about-secret-scanning)
- [Heroku Configuration and Config Vars](https://devcenter.heroku.com/articles/config-vars)
- [OpenAI API Keys Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
