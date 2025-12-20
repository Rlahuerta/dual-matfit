# Dual Material Fitting
 - keywords: Dual Material, Mixed Variational, Inverse Problem, Material Fitting

A Python library for solving mixed variational problems with multiple fields, specifically designed for fitting hyperelastic material models (like the Holzapfel-Gasser-Ogden model) to experimental data from extension tests, such as those performed using an Instron machine.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Processing Experimental Data](#processing-experimental-data)
  - [Solving the Variational Problem](#solving-the-variational-problem)
  - [Running Unit Tests](#running-unit-tests)
  - [Automated Testing Script](#automated-testing-script)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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
- **Note**: Python 3.11+ is required due to dependencies on modern type hints and performance improvements in NumPy/SciPy
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pytest](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [virtualenv](https://virtualenv.pypa.io/)

### Setup

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/dual-material-fitting.git
cd dual-material-fitting
```

#### Create a Hyperlink to create local virtual environment
```bash
ln -s /home/tytan/NAS/Repositories/PycharmProjects/DualMatFit/poetry.lock poetry.lock
ln -s /home/tytan/NAS/Repositories/PycharmProjects/DualMatFit/pyproject.toml pyproject.toml
```

#### The Python local virtual environment (via Poetry)
```bash
poetry config virtualenvs.create true 
poetry config virtualenvs.in-project true --local
poetry install
```

#### Using the docker-compose File

When you run `docker-compose up --build` (windows), `docker compose up --build` (linux) or any other Docker Compose command that requires reading the `docker-compose.yml` file.

## Migration from Python 3.7-3.10

If you are upgrading from an older version of DualMatFit that supported Python 3.7-3.10, please see the [Migration Guide](docs/migration/python-version-upgrade.md) for detailed instructions.

**Quick Summary:**
- Python 3.11+ is now required (3.7-3.10 are no longer supported)
- Performance improvements: 10-60% faster execution
- Better type safety and IDE support
- Security updates for supported Python versions

To quickly migrate:
```bash
# Install Python 3.11+
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

## Documentation

### Mathematical Algorithms

Comprehensive mathematical documentation is available in the [`docs/algorithms/`](docs/algorithms/) directory:

- **[HGO Material Model](docs/algorithms/hgo_model.md)**: Complete mathematical formulation including strain energy decomposition, fiber dispersion, and material parameter ranges. Based on Holzapfel et al. (2000, 2005, 2010) and Gasser et al. (2006).

- **[Mixed Formulations](docs/algorithms/mixed_formulations.md)**: Three variational formulations for handling near-incompressibility (u, u-p, u-p-θ), volumetric locking solutions, and selection guidelines. Based on Simo et al. (1985) and Sussman-Bathe (1987).

- **[Stabilization Parameters](docs/algorithms/stabilization.md)**: Detailed guide for numerical stabilization including matrix regularization, barrier methods, trust regions, L2 regularization, and parameter selection methodologies.

See the [**Algorithm Documentation Index**](docs/algorithms/README.md) for a complete overview.

### Key References

The implementation is based on these seminal papers:

1. **Holzapfel, G. A., Gasser, T. C., & Ogden, R. W. (2000)**. *A new constitutive framework for arterial wall mechanics and a comparative study of material models*. Journal of Elasticity, 61(1-3), 1-48. [DOI: 10.1023/A:1010835316564](https://doi.org/10.1023/A:1010835316564)

2. **Gasser, T. C., Ogden, R. W., & Holzapfel, G. A. (2006)**. *Hyperelastic modelling of arterial layers with distributed collagen fibre orientations*. Journal of the Royal Society Interface, 3(6), 15-35. [DOI: 10.1098/rsif.2005.0073](https://doi.org/10.1098/rsif.2005.0073)

3. **Holzapfel, G. A., & Ogden, R. W. (2010)**. *Constitutive modelling of arteries*. Proceedings of the Royal Society A, 466(2118), 1551-1597. [DOI: 10.1098/rspa.2010.0058](https://doi.org/10.1098/rspa.2010.0058)

### Python Version Requirements

For details on Python version support and migration, see:
- [Python Version Quick Reference](docs/PYTHON_VERSION.md)
- [Migration Guide: Python 3.7-3.10 to 3.11+](docs/migration/python-version-upgrade.md)
