#!/bin/bash
# Build script for Password Guesser

set -e

echo "=== Password Guesser Build Script ==="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    python_version=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python version: $python_version"
}

# Build pip package
build_pip() {
    print_info "Building pip package..."
    python -m pip install --upgrade build
    python -m build
    print_info "Package built in dist/"
    ls -la dist/
}

# Build Docker image
build_docker() {
    print_info "Building Docker image..."
    docker build -t password-guesser:latest .
    print_info "Docker image built: password-guesser:latest"
}

# Build executable with PyInstaller
build_exe() {
    print_info "Building executable with PyInstaller..."
    pip install pyinstaller

    pyinstaller --onefile \
        --name password-guesser \
        --add-data "config.yaml:." \
        --add-data "web/templates:web/templates" \
        --add-data "web/static:web/static" \
        --hidden-import torch \
        --hidden-import yaml \
        --collect-all torch \
        password_guesser/cli.py

    print_info "Executable built in dist/password-guesser"
}

# Run tests
run_tests() {
    print_info "Running tests..."
    pip install pytest
    pytest tests/ -v || print_warn "Some tests failed"
}

# Clean build artifacts
clean() {
    print_info "Cleaning build artifacts..."
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf .pytest_cache/
    rm -rf .mypy_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    print_info "Clean complete"
}

# Main
case "${1:-all}" in
    pip)
        check_python
        build_pip
        ;;
    docker)
        build_docker
        ;;
    exe)
        check_python
        build_exe
        ;;
    test)
        run_tests
        ;;
    clean)
        clean
        ;;
    all)
        check_python
        clean
        build_pip
        print_info "Build complete!"
        print_info "Install with: pip install dist/password_guesser-1.0.0-py3-none-any.whl"
        ;;
    *)
        echo "Usage: $0 {pip|docker|exe|test|clean|all}"
        echo ""
        echo "Commands:"
        echo "  pip     - Build pip package (wheel + sdist)"
        echo "  docker  - Build Docker image"
        echo "  exe     - Build standalone executable"
        echo "  test    - Run tests"
        echo "  clean   - Remove build artifacts"
        echo "  all     - Clean and build pip package (default)"
        exit 1
        ;;
esac
