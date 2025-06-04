#!/bin/bash

echo "===== LEXICON Secret Scanner ====="
echo "This script will scan the codebase for potential secrets and sensitive information."
echo ""

python scripts/check_for_secrets.py "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "Potential secrets were found. Please review the findings and fix any issues."
    echo "For guidance on managing secrets, see docs/SECURITY.md"
else
    echo ""
    echo "No potential secrets were found. Good job!"
fi

echo ""
echo "===== Scan Complete ====="
