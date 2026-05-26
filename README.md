# Dual Material Fitting

`dualmatfit` is a Python package for fitting fiber-reinforced hyperelastic
constitutive models to one-dimensional aortic ring-extension data. It was
developed for the dual-estimation framework used in the manuscript
*Capturing Regional Variation in Aortic Mechanics: Dual-Estimation Method for
Material Parameter Identification and Biological Correlation*.

The package combines symbolic continuum-mechanics formulation, numerical
forward solves, robust inverse fitting, and plotting utilities for comparing
regional aortic mechanics across proximal ascending/aortic-arch, descending
thoracic, and descending abdominal segments.

**Keywords:** aorta, uniaxial extension test, fiber-reinforced material,
Holzapfel-Gasser-Ogden model, nonlinear optimization, inverse problems.

## Associated publication

This software accompanies:

> **Capturing Regional Variation in Aortic Mechanics: Dual-Estimation Method for Material Parameter Identification and Biological Correlation**
>
> Ricardo Doll Lahuerta, Ayumi A. Miyakawa, Marina J. S. Maizato, Renato Crajoinas, Bruno Durante da Silva, José Eduardo Krieger, Eduardo Moacyr Krieger, Idágene A. Cestari
>
> *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences* (RSPA-2025-1043)

The manuscript introduces a regularized dual-estimation strategy for identifying
regional rat-aorta material parameters from uniaxial extension tests. A global
baseline estimator first captures each subject's overall mechanical signature;
local refinements then estimate segment-specific parameters while remaining
regularized toward that baseline. The fitted parameters and model-decomposed
stress contributions are interpreted as regularized mechanical descriptors, not
as direct measurements of isolated tissue constituents.

## Scientific scope

The implemented model follows a compressible HGO-type formulation with
Modified Anisotropic (MA) principles: the anisotropic fiber energy is evaluated
from full right Cauchy-Green invariants rather than only isochoric invariants.
This avoids the standard volumetric-isochoric split issue in which anisotropic
fiber terms can become insensitive to volumetric deformation.

The model represents the aortic ring as a single-layer, fiber-reinforced
composite with:

- isotropic ground-matrix response, typically neo-Hookean;
- volumetric regularization for slightly compressible, near-incompressible tissue;
- two symmetric collagen-fiber families with stiffness, nonlinearity, orientation, and dispersion parameters;
- a three-field mixed variational formulation using displacement, independent dilatation, and pressure-like fields;
- plane-stress simple-tension reduction for one-dimensional ring-extension fitting.

The fitting objective uses a robust Cauchy loss, Tikhonov regularization, and a
volume regularization term. These choices are important because HGO parameter
identification from biological uniaxial data is non-convex, outlier-sensitive,
and often non-unique.

## Main capabilities

- Parse and preprocess Instron uniaxial extension data.
- Build symbolic strain-energy and variational forms with SymPy.
- Convert symbolic expressions to numerical functions with NumPy/JAX backends.
- Solve forward uniaxial extension problems with Newton-type root solvers.
- Fit baseline and locally refined material parameters with robust regularized objectives.
- Use local and global optimizers, including SciPy optimizers, IPOPT wrappers, basinhopping, differential evolution, and SHGO.
- Compute sensitivities through finite-difference and adjoint-style derivative utilities.
- Generate force-stretch, stress-decomposition, parameter-trend, and manuscript-support plots.
- Run identifiability and covariance diagnostics for interpreting parameter confidence.

## Repository layout

| Path | Purpose |
| --- | --- |
| `dualmatfit/data/` | Experimental data parsing and scaling utilities |
| `dualmatfit/formulation/` | Symbolic tensors, material laws, variational forms, and lambdification |
| `dualmatfit/solvers/` | Nonlinear forward solves and derivative utilities |
| `dualmatfit/optimization/` | Cost functions, regularization, optimization drivers, IPOPT integration |
| `dualmatfit/fitting/` | High-level material fitting workflow and result persistence |
| `dualmatfit/plotting/` | Experimental, analytical, and solution plotting utilities |
| `scripts/` | Reproducible workflows for fitting, plotting, and manuscript diagnostics |
| `tests/` | Unit, integration, and performance tests |

## Installation

### Requirements

- Python 3.11, 3.12, or 3.13
- Conda, Mamba, or compatible Conda distribution
- System libraries for scientific Python and IPOPT when building outside the
  provided Conda environment

### Conda setup

```bash
source ~/anaconda3/bin/activate root

conda env create -f environment.yml
conda activate matfit1d
pip install -e .
```

If you want to use the IPOPT optimizer outside the provided Conda environment,
install the optional extra after creating your environment:

```bash
pip install -e ".[ipopt]"
```

To update an existing environment after pulling changes:

```bash
conda env update -f environment.yml --prune
conda activate matfit1d
pip install -e .
```

Verify the installation:

```bash
python -c "import dualmatfit; print(dualmatfit.__version__)"
```

JAX is configured to default to CPU execution unless `JAX_PLATFORMS` is set
explicitly by the user or runtime environment.

## Docker

The Dockerfile follows the same Conda workflow as the local setup. It installs
Miniforge, creates the `matfit1d` environment from `environment.yml`, and makes
that environment the default runtime path.

```bash
docker build -t dualmatfit .
docker run --rm -it dualmatfit bash
```

## Running tests and checks

```bash
# Run all configured tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=dualmatfit --cov-report=term-missing

# Run tests in parallel
pytest -n auto

# Run the local pipeline through the matfit1d Conda environment
bash local_pipelines.sh
```

Quality checks:

```bash
ruff check dualmatfit/
ruff format --check dualmatfit/
mypy dualmatfit/
```

## Minimal usage example

Built package artifacts do **not** bundle the experimental HDF5 dataset. Pass
`h5_path` explicitly unless you are running from a repository checkout that also
contains `instron_data/final_data.h5`.

```python
from dualmatfit.fitting.core import AnisoMaterialFit

selection = {
    "rato_17": {
        "Ar": ["A", "B", "C"],
        "Tr": ["A", "B"],
        "Ab": ["A", "B", "C"],
    }
}

fit = AnisoMaterialFit(
    selection,
    h5_path="path/to/final_data.h5",
    itype="nh",
    mtype=3,
    dvol=True,
    kappa=True,
    iso_split=False,
    ncontrol=50,
    opt_type="L-BFGS-B",
    opt_glb=True,
    lambdify="jax",
)

fit.exp_test_eval(plot=False)
fit.find_baseline_parameters(
    ftype="cauchy_robust",
    c=40,
    alpha=1e-4,
    epsilon=1e-3,
    dvol=True,
)
fit.find_optimal_parameters(
    ftype="cauchy_robust",
    c=40,
    alpha=1e-3,
    epsilon=1e-3,
    dvol=True,
)
fit.save_data()
fit.plot_fit(global_opt=True)
```

If you prefer `opt_type="ipopt"`, install the optional IPOPT dependencies first.

For the manuscript-style workflow, see `scripts/aniso_mat_fit.py`,
`scripts/precompute_paper_data.py`, `scripts/plot_experimental_visuals.py`, and
`scripts/plot_analytical_visuals.py`.

## Interpretation notes

The manuscript's conclusions rely on careful interpretation of fitted parameters:

- `k_1` is a mechanical correlate of collagen-related stiffening, not a direct
  measurement of collagen content.
- `mu` reflects an effective non-collagenous matrix contribution under the
  fitted loading mode, not total elastin abundance alone.
- `D` is regularization-influenced in the uniaxial setting because transverse
  strains are not directly measured.
- `kappa` should be interpreted as an effective loaded-state dispersion
  descriptor; it should not be transferred to biaxial, inflation-extension, or
  three-dimensional simulations without re-identification or independent
  structural constraints.
- Smooth local parameter trends are guided by Tikhonov regularization and should
  be treated as regularized descriptors rather than proof of perfectly smooth
  biological gradients.

## Key references

1. Holzapfel, G. A., Gasser, T. C., & Ogden, R. W. (2000). *A new constitutive framework for arterial wall mechanics and a comparative study of material models*. Journal of Elasticity, 61(1-3), 1-48. <https://doi.org/10.1023/A:1010835316564>
2. Gasser, T. C., Ogden, R. W., & Holzapfel, G. A. (2006). *Hyperelastic modelling of arterial layers with distributed collagen fibre orientations*. Journal of the Royal Society Interface, 3(6), 15-35. <https://doi.org/10.1098/rsif.2005.0073>
3. Nolan, D. R., Gower, A. L., Destrade, M., Ogden, R. W., & McGarry, J. P. (2014). *A robust anisotropic hyperelastic formulation for the modelling of soft tissue*. Journal of the Mechanical Behavior of Biomedical Materials, 39, 48-60.
4. Simo, J. C., & Taylor, R. L. (1982). *Penalty function formulations for incompressible nonlinear elastostatics*. Computer Methods in Applied Mechanics and Engineering, 35(1), 107-118.

## Funding

This work was supported by Fundação de Amparo à Pesquisa do Estado de São Paulo
(FAPESP) under grants 2012/50283-6 and 2019/21236-9. The Article Processing
Charge for the associated publication was funded by Coordenação de
Aperfeiçoamento de Pessoal de Nível Superior (CAPES; ROR identifier:
00x0ma614).

## License

This repository is source-available under the **PolyForm Noncommercial License
1.0.0**. The code is free to study, use, modify, and redistribute for
permitted noncommercial purposes, but commercial use is prohibited. This is not
an OSI-compliant open-source license. See `LICENSE` for the full terms.
