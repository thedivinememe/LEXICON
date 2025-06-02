#!/usr/bin/env python
"""
Setup script for LEXICON.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lexicon",
    version="0.1.0",
    author="LEXICON Team",
    author_email="info@example.com",
    description="Memetic Atomic Dictionary with Vectorized Objects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lexicon",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.0.0",
        "python-multipart>=0.0.6",
        "torch>=2.2.0",
        "transformers>=4.35.0",
        "numpy>=1.24.3",
        "scipy>=1.11.4",
        "faiss-cpu>=1.7.4",  # or faiss-gpu for GPU support
        "asyncpg>=0.29.0",
        "sqlalchemy>=2.0.23",
        "alembic>=1.12.1",
        "redis>=5.0.1",
        "strawberry-graphql[fastapi]>=0.215.1",
        "websockets>=12.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "click>=8.1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "isort>=5.12.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lexicon=src.main:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
