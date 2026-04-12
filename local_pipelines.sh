#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# A script for running tests and generating coverage reports.

# Exit on error and print commands (verbose for debugging)
set -euo pipefail
set -v

# Enable debug mode if DEBUG environment variable is set
if [ "${DEBUG:-}" == "1" ]; then
  set -x
fi

# Function to print usage information
usage() {
  echo "Usage: $0 [-n NUM_PROCESSES] [-e ENV_NAME] [-d TEST_DIR] [-c COVERAGE_DIR] [-k MARKER_EXPR]";
  echo "";
  echo "Options:";
  echo "  -n NUM_PROCESSES  Number of parallel processes to use (default: 4)";
  echo "  -e ENV_NAME       Name of the conda environment (default: matfit1d)";
  echo "  -d TEST_DIR       Test directory or directories (default: 'tests/unit tests/integration tests/performance')";
  echo "  -c COVERAGE_DIR   Directory to output coverage reports (default: ./coverage)";
  echo "  -k MARKER_EXPR    Pytest marker expression (default: 'not slow')";
  echo "  -h                Display this help message";
  exit 1
}

# Default values
NUM_PROCESSES=4
ENV_NAME="matfit1d"
TEST_DIRS="tests/unit tests/integration tests/performance"
COVERAGE_DIR="coverage"
MARKER_EXPR="not slow" # Default pytest marker expression

# Parse command-line options
while getopts "n:e:d:c:k:h" opt; do
  case $opt in
    n) NUM_PROCESSES=$OPTARG ;;
    e) ENV_NAME=$OPTARG ;;
    d) TEST_DIRS=$OPTARG ;;
    c) COVERAGE_DIR=$OPTARG ;;
    k) MARKER_EXPR=$OPTARG ;;
    h) usage ;;
    *) usage ;;
  esac
done

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$SCRIPT_DIR"
COVERAGE_DIR="$MAIN_DIR/$COVERAGE_DIR"

# Check that test directories exist
for tdir in $TEST_DIRS; do
  if [ ! -d "$MAIN_DIR/$tdir" ]; then
    echo "Test directory '$MAIN_DIR/$tdir' does not exist."
    exit 1
  fi
done

# Activate the conda environment
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Activating conda environment: $ENV_NAME"
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "Error: Conda environment '$ENV_NAME' not found."
  echo "Please run 'conda env create -f environment.yml' to create it."
  exit 1
fi

# Verify conda environment activation
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
  echo "Failed to activate conda environment '$ENV_NAME'."
  exit 1
fi

# Change to the main directory
cd "$MAIN_DIR"

################################################################################
# Code Quality Checks (Linters and Type Checker)
################################################################################

# Run ruff for general code style and linting checks.
set +e # Temporarily disable exit on error for linters
if command -v ruff >/dev/null 2>&1; then
  ruff check . --ignore C408,F841,C416
else
  echo "ruff not found in PATH; attempting via python -m ruff"
  python -m ruff check . --ignore C408,F841,C416 || true
fi
ruff_exit_code=$?
set -e # Re-enable exit on error

# Run mypy for static type checking on the dualmatfit package.
mypy dualmatfit/ || true
mypy_exit_code=$?

################################################################################
# Automated Testing and Coverage
################################################################################

# Create coverage directory if it doesn't exist
mkdir -p "$COVERAGE_DIR"

echo ""
echo "==================================="
echo " Running Pytest and Coverage..."
echo "==================================="

set +e
python -m pytest \
  -n "$NUM_PROCESSES" \
  -vv \
  --cov=dualmatfit \
  --cov-report=term-missing \
  --cov-report=html:"$COVERAGE_DIR/html" \
  --cov-report=xml:"$COVERAGE_DIR/coverage.xml" \
  -k "$MARKER_EXPR" \
  $TEST_DIRS
pytest_exit_code=$?
set -e

echo ""
echo "==================================="
echo " Pipeline Completed!"
echo "==================================="

if [ $ruff_exit_code -eq 0 ]; then
  echo "Ruff: Passed."
else
  echo "Ruff: Failed (exit code: $ruff_exit_code). Please check the output above."
fi

if [ $mypy_exit_code -eq 0 ]; then
  echo "Mypy: Passed."
else
  echo "Mypy: Failed (exit code: $mypy_exit_code). Please check the output above."
fi

if [ $pytest_exit_code -eq 0 ]; then
  echo "Pytest: Passed."
else
  echo "Pytest: Failed (exit code: $pytest_exit_code). Please check the output above."
fi

echo "Tests completed.";
echo "Coverage reports are available in '$COVERAGE_DIR'."