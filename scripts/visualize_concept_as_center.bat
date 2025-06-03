@echo off
REM Run the concept-centric visualization script

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Run the script
if "%1"=="" (
    echo Running with default concept (existence)
    python "%SCRIPT_DIR%visualize_concept_as_center.py" --open
) else if "%1"=="--all" (
    echo Visualizing all concepts as centers
    python "%SCRIPT_DIR%visualize_concept_as_center.py" --all --open
) else (
    echo Visualizing concept '%1' as center
    python "%SCRIPT_DIR%visualize_concept_as_center.py" --concept "%1" --open
)

if %ERRORLEVEL% neq 0 (
    echo Error running visualization script
    exit /b 1
)

echo Visualization complete
