#!/usr/bin/env python
"""
Script to check for potential sensitive information in the codebase.
This script scans files for patterns that might indicate API keys, passwords, or other secrets.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Pattern, Tuple, Set

# Patterns to look for (regex)
SECRET_PATTERNS = [
    # API Keys
    (r'(api[_-]?key|apikey|api[_-]?token|access[_-]?token)["\']?\s*[=:]\s*["\']([a-zA-Z0-9_\-\.]{20,})["\']', "API Key"),
    (r'sk[-_](?:test|live|proj)[-_][a-zA-Z0-9]{24,}', "OpenAI API Key"),
    (r'github_(?:pat|token|key)["\']?\s*[=:]\s*["\']([a-zA-Z0-9_\-]{20,})["\']', "GitHub Token"),
    (r'(aws|s3)_(?:access_key|secret_key)["\']?\s*[=:]\s*["\']([a-zA-Z0-9_\-\/+]{20,})["\']', "AWS Key"),
    
    # Database credentials
    (r'(db_password|database_password|db_pass|database_pass)["\']?\s*[=:]\s*["\']([^"\']{8,})["\']', "Database Password"),
    (r'(postgres|mysql|mongodb|redis):\/\/[^:]+:([^@]+)@', "Database Connection String with Password"),
    
    # JWT and other secrets
    (r'(jwt_secret|secret_key|encryption_key)["\']?\s*[=:]\s*["\']([^"\']{8,})["\']', "Secret Key"),
]

# Files and directories to ignore
IGNORE_DIRS = {
    ".git", 
    "node_modules", 
    "venv", 
    "env", 
    "__pycache__", 
    ".pytest_cache",
    "build",
    "dist",
    "*.egg-info"
}

IGNORE_FILES = {
    ".gitignore", 
    ".env.example", 
    "README.md", 
    "SECURITY.md",
    "check_for_secrets.py",  # Don't scan this file itself
}

IGNORE_EXTENSIONS = {
    ".pyc", 
    ".pyo", 
    ".so", 
    ".o", 
    ".a", 
    ".dll", 
    ".exe", 
    ".bin",
    ".png", 
    ".jpg", 
    ".jpeg", 
    ".gif", 
    ".svg", 
    ".ico",
    ".pdf", 
    ".doc", 
    ".docx", 
    ".xls", 
    ".xlsx", 
    ".ppt", 
    ".pptx"
}

def should_ignore(path: Path) -> bool:
    """Determine if a file or directory should be ignored."""
    if path.is_dir() and (path.name in IGNORE_DIRS or any(pattern in path.name for pattern in IGNORE_DIRS if "*" in pattern)):
        return True
    
    if path.is_file():
        if path.name in IGNORE_FILES:
            return True
        if path.suffix in IGNORE_EXTENSIONS:
            return True
    
    return False

def scan_file(file_path: Path, patterns: List[Tuple[str, str]]) -> List[Tuple[int, str, str]]:
    """Scan a file for potential secrets."""
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                for pattern, pattern_name in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        findings.append((i, pattern_name, match.group(0)))
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
    
    return findings

def scan_directory(directory: Path, patterns: List[Tuple[str, str]]) -> List[Tuple[Path, int, str, str]]:
    """Recursively scan a directory for potential secrets."""
    all_findings = []
    
    for item in directory.iterdir():
        if should_ignore(item):
            continue
        
        if item.is_file():
            findings = scan_file(item, patterns)
            for line_num, pattern_name, match in findings:
                all_findings.append((item, line_num, pattern_name, match))
        
        elif item.is_dir():
            sub_findings = scan_directory(item, patterns)
            all_findings.extend(sub_findings)
    
    return all_findings

def main():
    parser = argparse.ArgumentParser(description="Scan codebase for potential secrets and sensitive information.")
    parser.add_argument("--directory", "-d", type=str, default=".", help="Directory to scan (default: current directory)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more detailed output")
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return 1
    
    print(f"Scanning {directory} for potential secrets...")
    findings = scan_directory(directory, SECRET_PATTERNS)
    
    if not findings:
        print("No potential secrets found!")
        return 0
    
    print(f"\nFound {len(findings)} potential secrets:")
    for file_path, line_num, pattern_name, match in findings:
        rel_path = file_path.relative_to(directory)
        if args.verbose:
            print(f"{rel_path}:{line_num} - {pattern_name}: {match}")
        else:
            # Truncate the match to avoid displaying the full secret
            truncated_match = match[:20] + "..." if len(match) > 20 else match
            print(f"{rel_path}:{line_num} - {pattern_name}: {truncated_match}")
    
    print("\nWarning: This tool may produce false positives. Review findings manually.")
    print("For guidance on managing secrets, see docs/SECURITY.md")
    
    return 1 if findings else 0

if __name__ == "__main__":
    sys.exit(main())
