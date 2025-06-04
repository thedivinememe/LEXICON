@echo off
echo ===== LEXICON Secret Scanner =====
echo This script will scan the codebase for potential secrets and sensitive information.

python scripts\check_for_secrets.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Potential secrets were found. Please review the findings and fix any issues.
    echo For guidance on managing secrets, see docs\SECURITY.md
) else (
    echo.
    echo No potential secrets were found. Good job!
)

echo.
echo ===== Scan Complete =====
