#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# A script for running tests and generating coverage reports.

# Exit on error
set -euo pipefail

# Enable debug mode if DEBUG environment variable is set
if [ "${DEBUG:-}" = "1" ]; then
  set -vx
fi

# Function to print usage information
usage() {
  echo "Usage: $0 [-n NUM_PROCESSES] [-e ENV_NAME] [-d TEST_DIR] [-c COVERAGE_DIR] [-m MARKER_EXPR] [-k MARKER_EXPR]";
  echo "";
  echo "Options:";
  echo "  -n NUM_PROCESSES  Number of parallel processes to use (default: 4)";
  echo "  -e ENV_NAME       conda environment name (default: matfit1d)";
  echo "  -d TEST_DIR       Test directory or directories (default: 'tests/unit tests/integration tests/performance')";
  echo "  -c COVERAGE_DIR   Directory to output coverage reports (default: ./coverage)";
  echo "  -m MARKER_EXPR    Pytest marker expression (default: 'not slow')";
  echo "  -k MARKER_EXPR    Backward-compatible alias for -m";
  echo "  -h                Display this help message";
  exit 1;
}

# Default values
NUM_PROCESSES=4
ENV_NAME="matfit1d"
TEST_DIRS="tests/unit tests/integration tests/performance"
COVERAGE_DIR="coverage"
MARKER_EXPR="not slow" # Default pytest marker expression

# Parse command-line options
while getopts "n:e:d:c:m:k:h" opt; do
  case $opt in
    n) NUM_PROCESSES=$OPTARG ;;
    e) ENV_NAME=$OPTARG ;;
    d) TEST_DIRS=$OPTARG ;;
    c) COVERAGE_DIR=$OPTARG ;;
    m | k) MARKER_EXPR=$OPTARG ;;
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

# Locate conda binary
CONDA_BIN="${CONDA_EXE:-}"
if [ -z "$CONDA_BIN" ]; then
  for candidate in \
      "$HOME/anaconda3/bin/conda" \
      "$HOME/miniconda3/bin/conda" \
      "$(command -v conda 2>/dev/null || true)"; do
    if [ -x "$candidate" ]; then
      CONDA_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "$CONDA_BIN" ]; then
  echo "Error: conda not found. Install Anaconda/Miniconda or set CONDA_EXE."
  exit 1
fi

# Verify the conda environment exists
if ! "$CONDA_BIN" env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "Error: conda environment '$ENV_NAME' not found."
  echo "Create it with: conda env create -f environment.yml"
  exit 1
fi

echo "Using conda environment: $ENV_NAME  (via $CONDA_BIN)"

# Helper: run a command inside the conda env
conda_run() {
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" "$@"
}

# Change to the main directory
cd "$MAIN_DIR"

################################################################################
# Code Quality Checks (Linters and Type Checker)
################################################################################

# Run ruff for general code style and linting checks.
set +e # Temporarily disable exit on error for linters
conda_run python -m ruff check . --ignore C408,F841,C416
ruff_exit_code=$?
conda_run python -m mypy dualmatfit/
mypy_exit_code=$?
set -e # Re-enable exit on error

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
conda_run python -m pytest \
  -n "$NUM_PROCESSES" \
  -vv \
  --cov=dualmatfit \
  --cov-report=term-missing \
  --cov-report=html:"$COVERAGE_DIR/html" \
  --cov-report=xml:"$COVERAGE_DIR/coverage.xml" \
  -m "$MARKER_EXPR" \
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
echo "conda env: $ENV_NAME  |  conda: $CONDA_BIN"