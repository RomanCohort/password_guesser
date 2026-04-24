@echo off
REM Build script for Password Guesser (Windows)

echo === Password Guesser Build Script ===
echo.

setlocal EnableDelayedExpansion

REM Check command
if "%1"=="" goto all
if "%1"=="pip" goto pip
if "%1"=="docker" goto docker
if "%1"=="exe" goto exe
if "%1"=="test" goto test
if "%1"=="clean" goto clean
goto usage

:pip
echo [INFO] Building pip package...
python -m pip install --upgrade build
python -m build
echo [INFO] Package built in dist\
dir dist
goto end

:docker
echo [INFO] Building Docker image...
docker build -t password-guesser:latest .
echo [INFO] Docker image built: password-guesser:latest
goto end

:exe
echo [INFO] Building executable with PyInstaller...
pip install pyinstaller
pyinstaller --onefile --name password-guesser --add-data "config.yaml;." --add-data "web/templates;web/templates" --add-data "web/static;web/static" --hidden-import torch --hidden-import yaml password_guesser/cli.py
echo [INFO] Executable built in dist\password-guesser.exe
goto end

:test
echo [INFO] Running tests...
pip install pytest
pytest tests\ -v
goto end

:clean
echo [INFO] Cleaning build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%d in (*.egg-info) do rmdir /s /q "%%d"
if exist .pytest_cache rmdir /s /q .pytest_cache
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
echo [INFO] Clean complete
goto end

:all
echo [INFO] Checking Python version...
python --version
call :clean
call :pip
echo [INFO] Build complete!
echo [INFO] Install with: pip install dist\password_guesser-1.0.0-py3-none-any.whl
goto end

:usage
echo Usage: %0 {pip^|docker^|exe^|test^|clean^|all}
echo.
echo Commands:
echo   pip     - Build pip package (wheel + sdist)
echo   docker  - Build Docker image
echo   exe     - Build standalone executable
echo   test    - Run tests
echo   clean   - Remove build artifacts
echo   all     - Clean and build pip package (default)
exit /b 1

:end
endlocal
