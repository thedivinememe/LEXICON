@echo off
REM Setup script for LEXICON on Windows

echo Setting up LEXICON...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found. Please install Python 3.11 or higher.
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Found Python %PYTHON_VERSION%

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo Failed to create virtual environment.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo Failed to activate virtual environment.
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -U pip setuptools wheel
pip install -e .
if %ERRORLEVEL% neq 0 (
    echo Failed to install dependencies.
    exit /b 1
)

REM Create necessary directories
echo Creating necessary directories...
mkdir data 2>nul
mkdir data\faiss_index 2>nul
mkdir models 2>nul
mkdir logs 2>nul

REM Download BERT model
echo Downloading BERT model...
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('bert-base-uncased'); AutoTokenizer.from_pretrained('bert-base-uncased')"
if %ERRORLEVEL% neq 0 (
    echo Failed to download BERT model.
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    echo # LEXICON Environment Variables > .env
    echo DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lexicon >> .env
    echo REDIS_URL=redis://localhost:6379 >> .env
    echo SECRET_KEY=development_secret_key >> .env
    echo DEBUG=true >> .env
    echo VECTOR_INDEX_PATH=./data/faiss_index/index.faiss >> .env
)

echo.
echo LEXICON setup complete!
echo.
echo To start the application:
echo   1. Ensure PostgreSQL and Redis are running
echo   2. Initialize the database: python scripts\init_db.py --sample
echo   3. Create the vector index: python scripts\create_vector_index.py --test
echo   4. Start the server: uvicorn src.main:app --reload
echo.
echo To run tests:
echo   pytest
echo.
echo To deactivate the virtual environment:
echo   deactivate
echo.
