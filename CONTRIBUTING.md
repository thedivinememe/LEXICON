# Contributing to LEXICON

Thank you for your interest in contributing to LEXICON! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment information (OS, Python version, etc.)

### Suggesting Features

We welcome feature suggestions! Please create an issue with:

- A clear, descriptive title
- A detailed description of the proposed feature
- Any relevant examples or mockups
- Explanation of why this feature would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

## Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/lexicon.git
   cd lexicon
   ```

2. Run the setup script
   ```bash
   # On Linux/macOS
   ./scripts/setup.sh
   
   # On Windows
   scripts\setup.bat
   ```

3. Create a development database
   ```bash
   python scripts/init_db.py --sample
   ```

4. Create a vector index
   ```bash
   python scripts/create_vector_index.py --test
   ```

5. Start the development server
   ```bash
   uvicorn src.main:app --reload
   ```

## Testing

Run tests with pytest:

```bash
pytest
```

For specific test files:

```bash
pytest tests/test_vectorizer.py
```

## Code Style

We follow these coding standards:

- Use [Black](https://black.readthedocs.io/) for code formatting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints
- Write docstrings in the Google style

You can check your code style with:

```bash
black --check src tests
flake8 src tests
mypy src
```

## Documentation

- Update documentation when changing code
- Write clear docstrings for all functions, classes, and modules
- Keep the README.md updated

## Versioning

We use [Semantic Versioning](https://semver.org/). Please make sure your changes are compatible with this versioning scheme.

## License

By contributing to LEXICON, you agree that your contributions will be licensed under the project's MIT License.
