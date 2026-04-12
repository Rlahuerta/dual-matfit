# Dual Material Fitting
 - keywords: Dual Material, Mixed Variational, Inverse Problem, Material Fitting

A Python library for solving mixed variational problems with multiple fields, designed explicitly for fitting hyperelastic material models (like the Holzapfel-Gasser-Ogden model) to experimental data from extension tests, such as those performed using an Instron machine.

## Introduction

The **Dual Material Fitting** project provides tools for:

- Solving mixed variational problems with multiple fields (e.g., displacement and pressure).
- Processing and analyzing experimental data from mechanical tests (e.g., extension tests using an Instron machine).
- Fitting hyperelastic material models to experimental data using optimization techniques.

This library is intended for researchers and engineers working in computational mechanics and material modeling, who need to fit complex material models to experimental data.

## Features

- **Root-Finding Algorithms**: Implementations of Newton-Raphson methods for solving nonlinear variational problems.
- **Support for Multiple Fields**: Solve problems with two or more fields, such as displacement and pressure, using the `RootNewton` class.
- **Data Processing**: The `InstronData` class handles experimental data, providing methods for data extraction, processing, and visualization.
- **Refactored Codebase**: Improved code structure for better maintainability and readability.
- **Unit Testing**: Comprehensive unit tests for ensuring code reliability.
- **Automated Testing Script**: A bash script for running tests and code coverage analysis, emulating DevOps pipelines.

## Installation

### Prerequisites

- Python 3.11, 3.12, or 3.13
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/) (recommended for faster environment creation)
- **Note**: Python 3.11+ is required due to dependencies on modern type hints and performance improvements in NumPy/SciPy

### Setup

1. **Clone the Repository**

```bash
git clone https://github.com/Rlahuerta/dual-matfit.git
cd dual-matfit
```

2. **Create and activate the Conda environment**

```bash
conda env create -f environment.yml
conda activate matfit1d
pip install -e .
```

To update an existing environment after pulling changes:

```bash
conda env update -f environment.yml --prune
```

3. **Verify the installation**

```bash
python -c "import dualmatfit; print(dualmatfit.__version__)"
```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=dualmatfit --cov-report=term-missing

# Run in parallel
pytest -n auto

# Run the full pipeline script (lint, type check, test, coverage)
bash local_pipelines.sh
```

### Key References

The implementation is based on these seminal papers:

1. **Holzapfel, G. A., Gasser, T. C., & Ogden, R. W. (2000)**. *A new constitutive framework for arterial wall mechanics and a comparative study of material models*. Journal of Elasticity, 61(1-3), 1-48. [DOI: 10.1023/A:1010835316564](https://doi.org/10.1023/A:1010835316564)

2. **Gasser, T. C., Ogden, R. W., & Holzapfel, G. A. (2006)**. *Hyperelastic modelling of arterial layers with distributed collagen fibre orientations*. Journal of the Royal Society Interface, 3(6), 15-35. [DOI: 10.1098/rsif.2005.0073](https://doi.org/10.1098/rsif.2005.0073)

3. **Holzapfel, G. A., & Ogden, R. W. (2010)**. *Constitutive modelling of arteries*. Proceedings of the Royal Society A, 466(2118), 1551-1597. [DOI: 10.1098/rspa.2010.0058](https://doi.org/10.1098/rspa.2010.0058)
