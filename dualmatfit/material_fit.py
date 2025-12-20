# -*- coding: utf-8 -*-
"""
Material Fitting Module for Anisotropic Arterial Tissue

This module provides classes for fitting anisotropic hyperelastic material models
to experimental uniaxial extension test data from arterial tissue samples.

Classes
-------
AnisoModelSolve
    Base class for solving anisotropic material models
AnisoMaterialFit
    Extended class for material parameter fitting with optimization

Constants
---------
DEFAULT_BULK_MODULUS : float
    Default bulk modulus value (MPa) from "On the Compressibility of Arterial Tissue"
    Range: 42.14-99.03 kPa → Using median 56.67 kPa = 0.05667 MPa
DEFAULT_VOLUMETRIC_TYPE : str
    Default volumetric strain energy formulation ('simo92')
DEFAULT_NUM_CONTROL_POINTS : int
    Default number of control points for material fitting (15)
DEFAULT_SIMPLIFY_TIMEOUT : int
    Default timeout for symbolic simplification (1 second)

References
----------
.. [1] On the Compressibility of Arterial Tissue
       Bulk modulus range: 42.14-99.03 kPa
"""
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Hashable, Optional, Union

import numpy as np
import pandas as pd
from pyarrow import list_
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from dualmatfit.rato_info import excel_data
from dualmatfit.variational_form import VariationalFormulation, ring_geom
from dualmatfit.extension_solution import ExtensionSolution
from dualmatfit.least_square import CostFunction, CostIntegrator
from dualmatfit.drivers import opt_solvers
from dualmatfit.experimental import InstronData, MaterialSetup
from dualmatfit.plot import stress_plot, exp_test_plot
from dualmatfit.plotting.experimental_visuals import plot_material_fit
from dualmatfit.plotting.parameters import NAME_SECTIONS
from dualmatfit.path_manager import PathManager, PathConfiguration, PathLike
from dualmatfit.io_utils import MaterialFitIO, load_parquet_results

from dualmatfit.logging_config import get_logger
logger = get_logger('fitting')

__all__ = [
    'AnisoModelSolve',
    'AnisoMaterialFit',
]

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS AND CONFIGURATION

# Material Properties
# Reference: "On the Compressibility of Arterial Tissue"
# Bulk modulus range: 42.14-99.03 kPa
DEFAULT_BULK_MODULUS = 56.67 / 1000.0  # Median Value [MPa] = 0.05667 MPa

# Volumetric Strain Energy Types
# Options: 'bathe87', 'simo92', 'doll8' (works for fung model)
DEFAULT_VOLUMETRIC_TYPE = 'simo92'

# ============================================================================
# Fitting Configuration

DEFAULT_NUM_CONTROL_POINTS = 15  # Number of control points for material fitting
DEFAULT_SIMPLIFY_TIMEOUT = 1     # Symbolic simplification timeout (seconds)

# Data Structure Constants
EXPERIMENTAL_DATA_COLUMNS_PER_SECTION = 3  # Each section has 3 columns: force, stretch, position
SECTION_CODE_LENGTH = 2  # Length of section code (e.g., "Ar", "Tr", "Ab")
POSITION_CODE_INDEX = -1  # Index of position code (A, B, or C) in section key
RAT_KEY_PREFIX_LENGTH = 1  # Length of prefix to strip from HDF5 keys (leading "/")

# Physical Constants
UNSTRETCHED_STATE = 1.0  # Stretch ratio λ = 1.0 (no deformation)

# ============================================================================
# Plot Configuration
DEFAULT_FIGURE_SIZE = (14, 10)  # Figure size (width, height) in inches
DEFAULT_DPI = 120  # Resolution for saved figures
GRID_ALPHA_MINOR = 0.2  # Transparency for minor grid lines
GRID_ALPHA_MAJOR = 0.6  # Transparency for major grid lines

# Plot Limits (MPa)
DEFAULT_PLOT_LIMITS = {
    'iso': [-1.5, 1.0],    # Isotropic stress limits
    'ani': [-0.2, 4.2],    # Anisotropic stress limits
    'sum': [-0.2, 3.5],    # Total stress limits
    'lx': [0.95, 2.0]      # Stretch ratio limits
}

# Alternative plot limits for different scenarios
ALTERNATIVE_PLOT_LIMITS = {
    'iso': [-1.2, 1.4],
    'ani': [-0.2, 3.6],
    'sum': [-0.2, 3.0],
    'lx': [0.9, 1.8]
}

# Plot Scaling Factors
PLOT_MIN_SCALE = 0.5            # Scale factor for minimum plot limits
PLOT_MAX_SCALE = 1.5            # Scale factor for maximum plot limits
PLOT_FALLBACK_SCALE = 0.1       # Fallback scale when minimum is near zero
PLOT_ENERGY_THRESHOLD = 0.05    # Threshold for minimum energy values
PLOT_TICK_Y_DIVISOR = 10.0      # Divisor for y-axis tick spacing
PLOT_TICK_Y_MULTIPLIER = 4.0    # Multiplier for minor tick spacing
PLOT_MIN_LIMIT = 0.01           # Minimum value for plot limits

# Tick Configuration
TICK_MAJOR_DIVISOR = 16.0       # Divisor for major tick spacing along aorta sections
TICK_MINOR_SUBDIVISIONS = 4     # Number of subdivisions between major ticks
TICK_LABEL_TRIM = -1            # Index to trim last label from tick labels

# Optimization Configuration
DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS = 50           # Default iterations for baseline parameter optimization
DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS = 100             # Default iterations for local parameter optimization
DEFAULT_GLOBAL_ITERATIONS = 1  # Default basin-hopping iterations
DEFAULT_SOLUTION_NCONTROL = 15  # Number of control points for solution evaluation
HIGH_RESOLUTION_NCONTROL = 100  # Number of control points for high-resolution plots


class AnisoModelSolve:

    def __init__(self,
                 selection: Dict[str, Any],
                 itype: str = 'nh',
                 mtype: int = 1,
                 dvol: bool = True,
                 kappa: bool = True,
                 iso_split: bool = False,
                 hv: bool = False,
                 vol_type: str = None,
                 bulk: float = None,
                 ncontrol: int = 15,
                 lambdify: str = 'numpy',
                 simplify_tensors: bool = False,
                 simplify_timeout: int = 1,
                 main_path: Optional[Path] = None,
                 h5_path: Optional[Path] = None,
                 path_config: Optional['PathConfiguration'] = None,
                 **kwargs,
                 ):

        """
        Initialize AnisoModelSolve for material fitting.

        Parameters
        ----------
        :param selection: Selection of samples for solution (Dict with rat IDs and sections)
        :param mtype: Type of Mixed Variational Formulation (1 := u, 2 := u,p, 3 := u,t,p)
        :param itype: Type of Isochoric contribution (arterial elastin), e.g., 'nh', 'fung'
        :param dvol: Apply bulk modulus as design material parameter
        :param kappa: Flag to apply fiber dispersion model on the non-collagenous matrix
        :param vol_type: Volumetric strain energy formulation type ('simo92', 'bathe87', 'doll8').
                        If None, uses DEFAULT_VOLUMETRIC_TYPE.
        :param bulk: Bulk modulus value [MPa]. If None, uses DEFAULT_BULK_MODULUS.
        :param ncontrol: Number of control points for material fitting
        :param lambdify: Sympy lambdify backend ('numpy', 'jax', 'scipy', 'sympy', 'mpmath')
        :param main_path: Main path for saving results (Path object, optional)
        :param h5_path: Path to HDF5 data file (Path object, optional)
        :param path_config: Custom PathConfiguration. If None, uses defaults.
        
        Performance Parameters
        ----------------------
        :param simplify_tensors: Enable symbolic simplification of tensor expressions.
                                 When False (default), VariationalFormulation initialization
                                 is ~63x faster (0.35s vs 22s). Enable for production when
                                 simplified expressions are needed for code generation or analysis.
        :param simplify_timeout: Maximum time (seconds) for each simplification attempt.
                                Shorter values speed up initialization. Default: 1 second.
        
        Examples
        --------
        Fast mode (default, recommended for testing and optimization)::
        
            fit = AnisoMaterialFit(
                selection={'rato_17': {'Ar': ['A', 'B', 'C']}},
                itype='nh', mtype=1, dvol=True, kappa=True,
                ncontrol=10
                # simplify_tensors=False is default
            )
        
        Thorough mode (for production with simplified expressions)::
        
            fit = AnisoMaterialFit(
                selection={'rato_17': {'Ar': ['A', 'B', 'C']}},
                itype='nh', mtype=1, dvol=True, kappa=True,
                ncontrol=10,
                simplify_tensors=True,   # Enable simplification
                simplify_timeout=10      # Allow more time for complex expressions
            )
        """

        # --- Selection and Model Configuration ---
        # self.selection = selection
        self.selection = {key_i.replace('_', '-'): val_i for key_i, val_i in selection.items()}
        self.selection_ref = selection

        # Isotropic Parameters
        self.itype = itype

        # --- Variational Settings ---
        self.mtype = mtype
        self.vtype = vol_type if vol_type is not None else DEFAULT_VOLUMETRIC_TYPE
        self.dvol = dvol
        self.bulk = bulk if bulk is not None else DEFAULT_BULK_MODULUS

        self.hv = hv
        self.kappa = kappa
        self.iso_split = iso_split

        self.ncontrol: int = int(ncontrol)
        self.selection_nrm = self._eval_sections_keys()

        # Sympy lambdify options
        self.lambdify = lambdify
        
        # Performance settings for VariationalFormulation
        self._simplify_tensors = simplify_tensors
        self._simplify_timeout = simplify_timeout

        # --- Material Setup (Pre Processing) ---
        self.ese_types = ['iso', 'vol', 'ani', 'full']

        material_setup = MaterialSetup(
            itype=itype, 
            bulk=self.bulk, 
            dvol=dvol, 
            kappa=kappa
        )
        self.ds_vars, self.aorta_seq, self.mat_corr, self.ese_corr = material_setup()

        # --- Initialize State Variables ---
        # Information about the section of the test samples
        self.instron_data = excel_data()

        # baseline Material Parameters
        np_zeros = np.zeros(len(self.selection), dtype=float)
        init_zeros = {var_i: np_zeros.copy() for var_i in self.ds_vars.index}
        self.baseline_ds_vars = pd.DataFrame(init_zeros, index=list(self.selection.keys()))

        self.sections_ids = ['Ar', 'Tr', 'Ab']
        self.aorta_labels = self.ese_corr.columns.to_list()
        self.aorta_seq_idx = np.arange(len(self.aorta_labels))

        # Local Material Parameters
        self.optimal_ds_vars = {key_i.replace('_', '-'): None for key_i in self.selection.keys()}
        self.model_opt_res = {}

        # --- Plot Configuration ---
        self.xmajor_ticks = None
        self.xminor_ticks = None
        self.label_x: List = []

        self._setup_ticks()

        # --- Path Configuration ---
        self.path_manager = PathManager(config=path_config, base_path=main_path)
        
        # --- Load Experimental Data ---
        # Determine H5 file path
        if h5_path is None:
            h5_file_path = self.path_manager.config.get_h5_path()
        else:
            h5_file_path = Path(h5_path)
        
        # Validate H5 file exists
        self.h5_path = self.path_manager.validate_file_exists(h5_file_path)
        
        # Load data from H5 file using context manager for resource safety
        with pd.HDFStore(str(self.h5_path), mode='r') as h5_store:
            h5_keys = h5_store.keys()
            self.exp_test_data: Dict[str, pd.DataFrame] = {key_i: h5_store[key_i] for key_i in h5_keys}

        # --- Setup Results Directory ---
        if main_path is not None:
            self.results_dir = self.path_manager._resolve_path(Path(main_path))
        else:
            self.results_dir = self.path_manager.config.get_results_dir(
                self.mtype, self.itype, self.kappa, self.dvol, False
            )
        
        # Backward compatibility alias
        self.path_main = self.results_dir
        
        # Use clear, descriptive attribute names
        self.rat_solution_dirs: Dict[str, Path] = {}
        self.section_dirs: Dict[str, Dict[str, Path]] = {}
        
        # Backward compatibility aliases (deprecated, will be removed in future)
        self.solution_path = self.rat_solution_dirs
        self.local_solution_path = self.section_dirs

    def _setup_var_form(self, ds: Dict[str, Any]) -> VariationalFormulation:
        """
        Setup variational formulation for a given section geometry.

        Creates and configures a VariationalFormulation object with the appropriate
        material model parameters, mixing formulation, and performance settings.

        Parameters
        ----------
        ds: integration area

        Returns
        -------
        VariationalFormulation
            Configured variational formulation object with:
            - Strain energy density functions (isotropic + anisotropic)
            - Stress tensor expressions
            - Material parameter definitions

        Notes
        -----
        The variational formulation includes:
        - Isotropic contribution (elastin): Neo-Hookean or Fung-type
        - Anisotropic contribution (collagen): HGO model with dispersion
        - Volumetric contribution: Compressibility constraint
        - Mixed formulation: u (displacement), p (pressure), t (tension) fields

        Performance settings are inherited from class initialization:
        - simplify_tensors: Control symbolic simplification
        - simplify_timeout: Timeout for each simplification attempt

        See Also
        --------
        VariationalFormulation : Main class for strain energy formulations
        ring_geom : Function to compute geometric parameters
        """
        
        # Variational Object
        var_form = VariationalFormulation(
            ds=ds,
            itype=self.itype,
            mix=self.mtype,
            kappa=self.kappa,
            dvol=self.dvol,
            bulk=self.bulk,
            iso_split=self.iso_split,
            vol_type=self.vtype,
            hv=self.hv,
            was=True,
            # Performance parameters for fast initialization
            simplify_tensors=self._simplify_tensors,
            simplify_timeout=self._simplify_timeout,
        )

        return var_form

    def _sec_prep(self, idx: str, pd_exp_data: pd.DataFrame, info_data: Dict[str, Any], 
                  plot: bool = False) -> Dict[str, Any]:
        """
        Prepare section-specific data for material fitting.

        Processes experimental data for a specific section and position, creates
        the variational formulation, and sets up the extension solution solver.

        Parameters
        ----------
        idx : str
            Position identifier within the section ('A', 'B', or 'C')
        pd_exp_data : pd.DataFrame
            Experimental data from Instron testing machine with columns:
            - Force measurements [N]
            - Extension measurements [mm]
            - Time stamps [s]
        info_data : Dict[str, Any]
            Geometric and material information for the section:
            - Dimensions (inner/outer radius, thickness)
            - Initial configuration
            - Sample identifiers
        plot : bool, optional
            If True, generate and save position signal plots, by default False

        Returns
        -------
        Dict[str, Any]
            Dictionary containing prepared components:
            - 'ext_solution': ExtensionSolution object for computing stress-stretch
            - 'instron': InstronData object with processed experimental data

        Notes
        -----
        This method performs the following steps:
        1. Computes ring geometry from section dimensions
        2. Processes experimental data (filtering, normalization)
        3. Creates variational formulation for the section
        4. Sets up extension solution solver with chosen backend
        5. Optionally generates diagnostic plots

        The InstronData object contains:
        - Filtered and smoothed force-extension curves
        - Control points for optimization
        - Stretch ratio and PK1 stress calculations

        See Also
        --------
        ring_geom : Compute geometric parameters for ring sections
        InstronData : Experimental data processing class
        ExtensionSolution : Solver for extension problems
        """

        ds = ring_geom(info_data, idx)

        # Instron Data
        instron_data = InstronData(pd_exp_data, info_data[idx], ncontrol=self.ncontrol)

        # Solve Extension Model
        ext_solver = ExtensionSolution(var_form=self._setup_var_form(ds), module=self.lambdify)

        # FIXME: fix this method at InstronData or deprecate it
        # if plot:
        #     instron_data.plot(f'position_{idx}_signal.png')

        return {'ext_solution': ext_solver, 'instron': instron_data}

    def _get_index(self, test_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Create boolean index DataFrame indicating which sections are present in test cases.

        Generates a matrix showing which anatomical sections (Ar, Tr, Ab) are available
        for each test case position, useful for filtering and processing data.

        Parameters
        ----------
        test_info : Dict[str, Any]
            Dictionary with test case keys and their information.
            Keys typically include section-position identifiers like 'Ar-A', 'Tr-B', etc.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame with:
            - Index: Section IDs ['Ar', 'Tr', 'Ab']
            - Columns: Test case keys
            - Values: True if section is in test case key, False otherwise

        Examples
        --------
        >>> test_info = {'Ar-A': data1, 'Ar-B': data2, 'Tr-A': data3}
        >>> index_df = self._get_index(test_info)
        >>> print(index_df)
                Ar-A  Ar-B  Tr-A
        Ar      True  True  False
        Tr     False False   True
        Ab     False False  False

        Notes
        -----
        This index is used to efficiently filter and process only the relevant
        sections for each test case, avoiding unnecessary computation on missing data.
        """

        sec_data_i = {}
        for key_k in self.sections_ids:
            sec_data_i[key_k] = []
            for j, skey_j in enumerate(test_info.keys()):
                sec_data_i[key_k].append(key_k in skey_j)

        pd_index = pd.DataFrame(sec_data_i).T
        pd_index.columns = list(test_info.keys())

        return pd_index

    def _setup_ticks(self) -> None:
        """
        Configure major and minor tick positions for aorta section plots.

        Computes tick positions along the aorta axis for visualization, with major
        ticks at section boundaries and minor ticks for finer grid resolution.

        Notes
        -----
        Tick Configuration:
        - Major ticks: Divide aorta into 16 equal segments
        - Minor ticks: Subdivide each major interval into 4 parts
        - Labels: Anatomical section names at major tick positions

        The ticks are computed based on:
        - self.aorta_seq_idx: Array indices for aorta sections
        - self.aorta_labels: Names of anatomical sections
        - TICK_MAJOR_DIVISOR: Number of major divisions (16)
        - TICK_MINOR_SUBDIVISIONS: Subdivisions per major tick (4)

        Sets Instance Attributes
        ------------------------
        xmajor_ticks : np.ndarray
            Positions of major ticks along aorta axis
        xminor_ticks : np.ndarray
            Positions of minor ticks along aorta axis
        label_x : List[str]
            Labels for major tick positions (section names or empty strings)

        Examples
        --------
        The resulting ticks might look like:
        ```
        Major ticks: [0, 1.875, 3.75, 5.625, ...]
        Minor ticks: [0, 0.469, 0.938, 1.406, ...]
        Labels: ['AoA', '', 'DTAo', '', 'DAAo', ...]
        ```
        """

        major_inc = self.aorta_seq_idx.max() / TICK_MAJOR_DIVISOR
        xmajor_ticks = np.arange(0., self.aorta_seq_idx.max() + 1., major_inc)
        np_idx = np.where(xmajor_ticks <= self.aorta_seq_idx.max())[0]
        self.xmajor_ticks = xmajor_ticks[np_idx]

        minor_inc = major_inc / TICK_MINOR_SUBDIVISIONS
        xminor_ticks = np.arange(0., self.aorta_seq_idx.max() + 1., minor_inc)
        np_idx = np.where(xminor_ticks <= self.aorta_seq_idx.max())[0]
        self.xminor_ticks = xminor_ticks[np_idx]

        label_x = []
        for xm_j in xmajor_ticks:
            if xm_j in self.aorta_seq_idx:
                label_x.append(self.aorta_labels[int(xm_j)])
            else:
                label_x.append('')

        self.label_x = label_x[:TICK_LABEL_TRIM]

    def _eval_sections_keys(self) -> Dict[str, List[str]]:
        """
        Normalize and flatten section keys for experimental data processing.

        Converts the nested selection dictionary structure into a flat list of
        section-position identifiers for each rat, making it easier to match
        against experimental data keys.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping rat IDs to lists of section-position keys.
            Keys: Rat identifiers (e.g., 'rato-17')
            Values: Lists of section keys (e.g., ['Ar-A', 'Ar-B', 'Tr-C'])

        Examples
        --------
        Input selection:
        ```python
        self.selection = {
            'rato-17': {'Ar': ['A', 'B'], 'Tr': ['C']}
        }
        ```

        Output:
        ```python
        {
            'rato-17': ['Ar-A', 'Ar-B', 'Tr-C']
        }
        ```

        Notes
        -----
        The normalized keys follow the format: "{Section}-{Position}"
        where:
        - Section: Anatomical region code ('Ar', 'Tr', 'Ab')
        - Position: Circumferential position ('A', 'B', 'C')

        This normalization is essential for:
        - Matching experimental data columns
        - Filtering relevant test cases
        - Organizing optimization results
        """

        selection_nrm = {}

        for name_i, val_i in self.selection.items():
            selection_nrm[name_i] = []

            for sec_j, val_j in val_i.items():
                for sec_jk in val_j:
                    selection_nrm[name_i].append(f"{sec_j}-{sec_jk}")

        return selection_nrm

    def _create_rat_directory(self, rat_id: str) -> Path:
        """
        Create and return directory for rat/specimen results.

        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., '/rato_17' or 'rato_17')

        Returns
        -------
        Path
            Absolute path to created rat directory

        Notes
        -----
        Directory is created with parents=True, exist_ok=True via PathManager.
        Handles rat_id with or without leading slash.
        """
        rat_dir = self.path_manager.get_rat_solution_dir(self.results_dir, rat_id)
        return self.path_manager.ensure_dir(rat_dir)

    def _create_section_directory(self, rat_dir: Path, section: str) -> Path:
        """
        Create and return directory for section results within rat directory.

        Parameters
        ----------
        rat_dir : Path
            Parent rat directory path
        section : str
            Section identifier (e.g., 'Ar', 'Tr', 'Ab')

        Returns
        -------
        Path
            Absolute path to created section directory

        Notes
        -----
        Directory is created with parents=True, exist_ok=True via PathManager.
        """
        section_dir = self.path_manager.get_section_dir(rat_dir, section)
        return self.path_manager.ensure_dir(section_dir)

    def exp_test_eval(self, material_parameters: Optional[Dict[str, pd.DataFrame]] = None, plot: bool = False):
        """
        Evaluate experimental test data and compute model solutions.

        This method processes experimental uniaxial extension test data for all selected
        samples and sections. It sets up the variational formulation, computes model
        solutions, and optionally generates plots comparing experimental data with
        model predictions.

        Parameters
        ----------
        material_parameters : Dict[str, pd.DataFrame], optional
            Dictionary of material parameters for each rat ID.
            Keys: rat IDs (e.g., 'rato_17')
            Values: DataFrames with material parameter values
            If None, uses initial parameter values from self.ds_vars
            
        plot : bool, default=False
            If True, generates energy-strain plots for each test case.
            Plots are saved to the output directory.

        Returns
        -------
        None
            Results are stored in self.model_res dictionary with structure:
            {rat_id: {section_key: {
                'instron': InstronData object,
                'lsq': CostFunction object,
                'extension_solution': ExtensionSolution object
            }}}

        Notes
        -----
        This method performs the following steps for each sample:
        1. Loads experimental data from Instron testing machine
        2. Preprocesses data (filters, normalizes)
        3. Sets up variational formulation for the section geometry
        4. Creates extension solution object
        5. Computes cost function for optimization
        6. Optionally generates comparison plots

        The method processes sections in the following anatomical order:
        - Ar: Ascending aorta (root)
        - Tr: Transverse aortic arch  
        - Ab: Abdominal aorta

        Examples
        --------
        Evaluate with initial parameters:
        
        >>> model = AnisoModelSolve(selection, itype='nh')
        >>> model.exp_test_eval()
        
        Evaluate with optimized parameters and generate plots:
        
        >>> params = {'rato_17': df_parameters}
        >>> model.exp_test_eval(material_parameters=params, plot=True)

        See Also
        --------
        solve : Compute model solution for specific parameters
        _sec_prep : Prepare section-specific data
        """

        # --- Filter Test Cases ---
        test_cases = {key_j: "rato" in key_j for key_j in self.exp_test_data.keys()}

        for key_i, val_i in test_cases.items():
            if val_i:
                key_ref_i = key_i[RAT_KEY_PREFIX_LENGTH:]
                key_rnm_i = key_ref_i.replace('_', '-')

                if key_ref_i in self.selection:
                    # getting experimental data
                    pd_exp_data_i = self.exp_test_data[key_i]
                    key_clm_i = pd_exp_data_i.columns

                    # --- Extract Section Indices ---
                    sections_i = {}
                    for j, jk in enumerate(range(0, pd_exp_data_i.shape[1], EXPERIMENTAL_DATA_COLUMNS_PER_SECTION)):
                        key_res_ij = re.findall(r"\((.*?)\)", key_clm_i[jk])[0]
                        if key_res_ij in self.selection_nrm[key_ref_i]:
                            sections_i[key_res_ij] = np.arange(jk, jk + EXPERIMENTAL_DATA_COLUMNS_PER_SECTION)

                    # --- Create Output Directories ---
                    rat_dir = self._create_rat_directory(key_i)
                    self.solution_path[key_ref_i] = rat_dir

                    # --- Process Sections ---
                    dict_solu_res_ij = {}
                    dict_local_path_ij = {}
                    pd_solu_mat_i = None

                    if material_parameters is not None and material_parameters.get(key_ref_i) is not None:
                        if isinstance(material_parameters[key_ref_i], pd.DataFrame):
                            if material_parameters[key_ref_i].shape[0] > 0:
                                pd_solu_mat_i = material_parameters[key_ref_i].copy()

                    if pd_solu_mat_i is None:
                        pd_solu_mat_i = self.aorta_seq.copy(deep=True)

                    for sec_pos_ij, arg_ij in sections_i.items():
                        pd_exp_data_ij = pd_exp_data_i.iloc[:, arg_ij]
                        pd_exp_data_ij.dropna(inplace=True)

                        sec_ij = sec_pos_ij[:SECTION_CODE_LENGTH]
                        pos_ij = sec_pos_ij[POSITION_CODE_INDEX]

                        if self.instron_data[key_rnm_i].get(sec_ij) is not None:
                            instron_data_ij = self.instron_data[key_rnm_i][sec_ij]

                            if instron_data_ij.get(pos_ij) is not None:
                                logger.info(f'\n {key_i[RAT_KEY_PREFIX_LENGTH:]}, Preprocessing, Section: {sec_ij}, Position: {pos_ij}')

                                ####################################################################
                                # Create section directory using PathManager
                                if dict_local_path_ij.get(sec_ij) is None:
                                    section_dir = self._create_section_directory(rat_dir, sec_ij)
                                    dict_local_path_ij[sec_ij] = section_dir

                                solu_setup_ij = self._sec_prep(pos_ij, pd_exp_data_ij, instron_data_ij,
                                                               plot=plot)

                                dict_solu_res_ij[sec_pos_ij] = solu_setup_ij

                                if pd_solu_mat_i.loc[sec_pos_ij, "mlx"] == 0.:
                                    pd_solu_mat_i.loc[sec_pos_ij, "mlx"] = solu_setup_ij["instron"].np_tstretch_ref.max()

                    # --- Store Results ---
                    self.optimal_ds_vars[key_ref_i] = pd_solu_mat_i.loc[sections_i.keys(), :]

                    self.model_opt_res[key_ref_i] = dict_solu_res_ij
                    self.local_solution_path[key_ref_i] = dict_local_path_ij

    def solve(self):
        """
        FIXME: I am not sure if this method is been used or need to be deprecated

        Solve the material model for all test cases using current parameters.

        Computes model predictions for stress-stretch response using the current
        material parameters stored in self.baseline_ds_vars. This method evaluates
        the constitutive model at the experimental stretch values and compares
        with measured data.

        Returns
        -------
        None
            Results are stored in self.model_res dictionary.
            Each entry contains the computed stress-stretch curves and
            comparison with experimental data.

        Notes
        -----
        This is a convenience method that calls exp_test_eval() with the
        current baseline design variables. It's typically used after optimization
        to compute final model predictions.

        The solve process:
        1. Uses parameters from self.baseline_ds_vars
        2. Evaluates variational formulation
        3. Computes PK1 stress tensor
        4. Converts to Cauchy stress
        5. Stores results for plotting
        
        See Also
        --------
        exp_test_eval : Main evaluation method
        """

        for key_ref_i, model_i in self.model_opt_res.items():
            for psec_ij, model_psec_ij in model_i.items():
                mat_param_ij = self.optimal_ds_vars[key_ref_i].loc[psec_ij, :].to_dict()

                list_mat_ij = []
                list_mat_param_key_ij = list(self.var_form.dict_mat_vars.keys())
                list_mat_param_sym_ij = list(self.var_form.dict_mat_vars.values())

                for mat_par_w in self.var_form.mat_vars:
                    if mat_par_w in list_mat_param_sym_ij:
                        key_w = list_mat_param_key_ij[list_mat_param_sym_ij.index(mat_par_w)]

                        if key_w == "D":
                            list_mat_ij.append(mat_param_ij['bulk'])
                        else:
                            list_mat_ij.append(mat_param_ij[key_w])

                np_mat_ij = np.array(list_mat_ij, dtype=float)
                plt_fname_ij = f"OPT_{psec_ij[-1]}.png"

                model_psec_ij['instron'].eval(plt_fname_ij, model_psec_ij['ext_solution'], np_mat_ij)

    def _plot_ese(self,
                  solution_data: Dict[Hashable, Tuple[Dict[str, np.ndarray | np.ndarray], pd.Series] | Any],
                  aorta_mat_info: pd.DataFrame,
                  ):
        """
        FIXME: this function:   I am not sure if this method is been used or need to be deprecated

        :param solution_data:
        :param aorta_mat_info:

        :return:
        """

        # Create a colormap
        cmap = sns.cubehelix_palette(as_cmap=True)

        fig, ax = plt.subplots(len(self.ese_types), figsize=DEFAULT_FIGURE_SIZE, sharex=True, dpi=DEFAULT_DPI)
        fig.suptitle(f'Aorta Strain Energy Pattern (Inverse of Compliance) - l_x {aorta_mat_info["mlx"].min()}')

        np_idx = np.arange(0, len(self.aorta_labels))
        np_zeros = np.zeros(self.ncontrol - 1, dtype=float)

        sr_line_index = pd.Series(np_idx, index=self.aorta_labels)
        df_line_strain = pd.DataFrame(dict.fromkeys(solution_data.keys(), np_zeros))
        info_strain = {etype_i: df_line_strain.copy(deep=True) for etype_i in self.ese_types}

        for sec_i, val_i in solution_data.items():
            for etype_k, ese_val_k in val_i.ese.items():
                if isinstance(ese_val_k, list):
                    np_ese_val_k = np.array(ese_val_k, dtype=float)
                else:
                    np_ese_val_k = ese_val_k.copy()

                if info_strain.get(etype_k) is not None:
                    if info_strain[etype_k].shape[0] == np_ese_val_k.shape[0]:
                        info_strain[etype_k].loc[:, sec_i] = np_ese_val_k

        for i, (ese_type_i, ese_val_i) in enumerate(info_strain.items()):
            for ik, (sec_ik, ese_val_ik) in enumerate(ese_val_i.iterrows()):

                ax[i].set_xticks(self.xmajor_ticks)
                ax[i].set_xticks(self.xminor_ticks, minor=True)

                ax[i].set_xlim([-0.5, np_idx.max() + 0.5])

                ese_min_i = np.around(PLOT_MIN_SCALE * ese_val_ik.min(), 2)
                if ese_min_i < PLOT_ENERGY_THRESHOLD:
                    ese_min_i = -PLOT_FALLBACK_SCALE * ese_val_ik.max()

                ese_max_i = np.around(PLOT_MAX_SCALE * ese_val_ik.max(), 2)
                ese_max_i = max(ese_max_i, PLOT_MIN_LIMIT)

                ymin_ticks_i = np.abs(ese_max_i / PLOT_TICK_Y_DIVISOR)
                ymax_ticks_i = PLOT_TICK_Y_MULTIPLIER * ymin_ticks_i

                ymajor_ticks_i = np.arange(0., ese_max_i, ymin_ticks_i)
                yminor_ticks_i = np.arange(0., ese_max_i, ymax_ticks_i)

                ax[i].set_yticks(ymajor_ticks_i)
                ax[i].set_yticks(yminor_ticks_i, minor=True)
                ax[i].set_ylim([ese_min_i, ese_max_i])

                ax[i].grid(which='minor', alpha=GRID_ALPHA_MINOR)
                ax[i].grid(which='major', alpha=GRID_ALPHA_MAJOR)

                color_i = cmap(ik)

                ax[i].plot(sr_line_index[ese_val_i.columns].values, ese_val_ik.values, color=color_i,
                           label=f"ctrl-{ik}")
                ax[i].plot(sr_line_index[ese_val_i.columns].values, ese_val_ik.values, 'o', color=color_i)

                ax[i].set_ylabel(ese_type_i)
                ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax[-1].set_xlabel('Aorta Section Idx')
        ax[-1].set_xticklabels(self.label_x)

        return fig


class AnisoMaterialFit(AnisoModelSolve):
    """
    Similar Material Fitting Implementation:

    Article:    Direct and inverse identification of constitutive parameters from the structure of soft tissues.
                Part 2: dispersed arrangement of collagen fibers

    Article:    Arterial clamping: Finite element simulation and in vivo validation
                mu: 23.63 +- 4.13 [kPa], D: 650. [kPa] (Isotropic)
                k_1: 32.51 +- 6.13 [kPa], k_2: 3.05 +- 1.32 [-], kappa: 0.16 +- 0.01 [-], alpha: +- 5. [deg]

    Article:    Modelling non-symmetric collagen fibre dispersion in arterial walls (Dispersion)
    """

    def __init__(self,
                 selection: Dict[str, Any],
                 opt_type: str = 'L-BFGS-B',
                 opt_glb: bool = True,
                 stabilization: float = 0.,
                 **kwargs,
                 ):
        """
        Args:
            :param selection:           Selection of Rats for Solution

        Kwargs:
            :param opt_type:            Optimization Method Selection
            :param opt_glb:             Global Solution Search Flag
            :param stabilization:       Stabilization Parameter

        """

        # Call the __init__ of the parent class
        super().__init__(selection=selection, **kwargs,)

        # Optimization Parameters
        self.path_local_solution = ""
        self.opt_type: str = opt_type.replace('-', '')
        self.opt_type_label = opt_type
        self.opt_glb: bool = opt_glb
        self.aorta_seq["method"] = opt_type
        self.aorta_model_results = {}
        self.plt_lines = {'A': '-', 'B': '--', 'C': '-.'}
        
        # Initialize I/O handler (delegates file operations)
        self.io_handler = MaterialFitIO(self.path_manager, self.opt_type)

        # Main Solution Path Address - use PathManager
        if kwargs.get("work_path"):
            # Override with custom work path
            work_path = Path(kwargs.get("work_path"))
            self.results_dir = self.path_manager._resolve_path(work_path)
            self.path_main = self.results_dir  # Backward compatibility
        else:
            # Update results_dir for optimization (global vs local)
            self.results_dir = self.path_manager.config.get_results_dir(
                self.mtype, self.itype, self.kappa, self.dvol, opt_glb
            )
            self.path_main = self.results_dir  # Backward compatibility
        
        # Ensure results directory exists
        self.path_manager.ensure_dir(self.results_dir)
        
        self._stab = stabilization

        self._baseline_ftype = "lsq"
        self._local_ftype = "lsq_sum"
        self._dtype = "adjoint"

    def _sec_prep(self,
                  idx: str,
                  pd_exp_data: pd.DataFrame,
                  info_data: Dict[str, Any],
                  plot: bool = False,
                  ) -> Dict[str, Any]:
        """
        Prepare section-specific data and cost function for optimization.

        Extended version of parent method that additionally creates the cost
        function for inverse parameter identification.

        Parameters
        ----------
        idx : str
            Position identifier within the section ('A', 'B', or 'C')
        pd_exp_data : pd.DataFrame
            Experimental data from Instron testing machine
        info_data : Dict[str, Any]
            Geometric and material information for the section
        plot : bool, optional
            If True, generate and save position signal plots, by default False

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'lsq': CostFunction object for least squares optimization
            - 'instron': InstronData object with experimental data
            - 'ext_solution': ExtensionSolution solver
            - 'xdisp': Maximum extension displacement [mm]

        Notes
        -----
        This method extends the parent _sec_prep by adding:
        1. Cost function setup for parameter optimization
        2. Design variable bounds configuration
        3. Least squares objective function

        The cost function type is controlled by self._dtype:
        - 'adjoint': Adjoint-based gradient computation (fast)
        - 'fdm': Finite difference gradient (fallback)

        See Also
        --------
        CostFunction : Least squares cost function for optimization
        AnisoModelSolve._sec_prep : Parent method for basic setup
        """

        # --- Extract Section Geometry ---
        ds = ring_geom(info_data, idx)

        # --- Setup Variational Formulation ---
        instron_data = InstronData(pd_exp_data, info_data[idx], self.ncontrol)

        var_form = self._setup_var_form(ds)
        ext_solver = ExtensionSolution(var_form, module=self.lambdify)

        # --- Create Cost Function ---
        # Material + design variables Bounds + Sympy Lambdify Module
        pd_mat_params = self.ds_vars[["values", "lower", "upper", "limit", "variable"]].copy(deep=True)

        # Cost Function for the inverse problem
        lsq_mat_fun = CostFunction(var_form=var_form,
                                   load_ref=instron_data.np_tload_ref,
                                   stretch_x=instron_data.np_tstretch_ref,
                                   dsvars=pd_mat_params,
                                   ftype="lsq",
                                   module=self.lambdify,
                                   dtype=self._dtype,
                                   )

        # Constraint Function for the inverse problem
        # aniso_inv_fun = AnisoInvariantFunction(var_form=var_form,
        #                                        stretch_x=instron_data.np_tstretch_ref,
        #                                        dsvars=pd_mat_params,
        #                                        constraints_bounds={'lower': 1., 'upper': 4.},
        #                                        module=self.lambdify,
        #                                        )

        # --- Build Result Dictionary ---
        opt_setup = {'lsq': lsq_mat_fun,
                     'instron': instron_data,
                     'ext_solution': ext_solver,
                     'xdisp': instron_data.np_textn_ref[-1],
                     # 'constraint': aniso_inv_fun,
                     }

        # FIXME: fix this method at InstronData or deprecate it
        # if plot:
        #     instron_data.plot(f'position_{idx}_signal.png')

        return opt_setup

    def exp_test_eval(self,
                      plot: bool = False,
                      **kwargs,
                      ) -> None:
        """
        Evaluate experimental test data and prepare optimization setup.

        Processes all experimental test cases, extracts sections, and prepares
        cost functions for parameter optimization. This is the main entry point
        for setting up the inverse problem.

        Parameters
        ----------
        plot : bool, optional
            If True, generate diagnostic plots for each test case, by default False
        **kwargs : dict
            Additional keyword arguments (reserved for future use)

        Returns
        -------
        None
            Results are stored in instance attributes:
            - self.optimal_ds_vars: Material parameters for each section
            - self.model_opt_res: Optimization setup (cost functions, solvers)
            - self.path_local_solution: Paths for saving results

        Notes
        -----
        Processing Pipeline:
        1. Identify rat test cases from experimental data
        2. For each test case:
           a. Extract relevant sections based on selection
           b. Create test case directory
           c. Process each section-position combination
           d. Setup cost functions for optimization
        3. Store results in instance dictionaries

        This method prepares everything needed for optimization but does not
        perform the actual parameter fitting. Use find_baseline_parameters() or
        find_optimal_parameters() for optimization.

        See Also
        --------
        _identify_test_cases : Identify rat test cases
        _process_individual_test_case : Process single test case
        find_baseline_parameters : Global parameter optimization
        find_optimal_parameters : Section-specific optimization

        Examples
        --------
        >>> fit = AnisoMaterialFit(selection, opt_type='ipopt')
        >>> fit.exp_test_eval(plot=False)
        >>> # Now ready for optimization
        >>> fit.find_baseline_parameters()
        """

        test_cases = self._identify_test_cases()

        optimal_ds_vars, model_opt_res, path_local_solution = {}, {}, {}

        for key_i, is_rat_test_i in test_cases.items():
            if is_rat_test_i:
                opt_res_i, opt_mat_i, local_path_i = self._process_individual_test_case(key_i, plot)

                if opt_res_i is not None and opt_mat_i is not None and local_path_i is not None:
                    key_ref_i = key_i[1:].replace('_', '-')
                    optimal_ds_vars[key_ref_i] = opt_res_i
                    model_opt_res[key_ref_i] = opt_mat_i
                    path_local_solution[key_ref_i] = local_path_i

        self.optimal_ds_vars = optimal_ds_vars
        self.model_opt_res = model_opt_res
        self.path_local_solution = path_local_solution

    def _identify_test_cases(self) -> Dict[str, bool]:
        """
        Identify test cases that involve rats.

        Returns:
            Dict[str, bool]: Dictionary with test case keys and a boolean indicating if it's a rat test.
        """
        return {key: "rato" in key for key in self.exp_test_data.keys()}

    def _process_individual_test_case(self,
                                      key: str,
                                      plot: bool,
                                      ):
        """
        Process each individual test case.

        Args:
            key (str): The key of the test case.
            plot (bool): Whether to plot results.
        """
        key_ref = key[RAT_KEY_PREFIX_LENGTH:].replace('_', '-')

        if key[RAT_KEY_PREFIX_LENGTH:] in self.selection_ref:
            pd_exp_data = self.exp_test_data[key]
            sections = self._extract_sections(pd_exp_data, key_ref)

            path_local = self._setup_test_case_directory(key)
            self.solution_path[key_ref] = path_local

            return self._process_sections(key_ref, sections, pd_exp_data, path_local, plot)

        else:
            return None, None, None

    def _setup_test_case_directory(self, key: str) -> Path:
        """
        Setup the directory for a test case.

        Args:
            key (str): The key of the test case (rat identifier).

        Returns:
            Path: The path to the directory.
        """
        return self._create_rat_directory(key)

    def _extract_sections(self, pd_exp_data: pd.DataFrame, key_ref: str) -> Dict[str, np.ndarray]:
        """
        Extract sections from experimental data.

        Args:
            pd_exp_data (pd.DataFrame): Experimental data.
            key_ref (str): Key reference of the test.

        Returns:
            Dict[str, range]: Dictionary with section keys and corresponding column ranges.
        """
        sections = {}
        exp_columns = pd_exp_data.columns.to_list()

        for j, col_range_j in enumerate(range(0, len(exp_columns), EXPERIMENTAL_DATA_COLUMNS_PER_SECTION)):
            section_key = re.findall(r"\((.*?)\)", pd_exp_data.columns[col_range_j])[0]
            if section_key in self.selection_nrm[key_ref]:
                sections[section_key] = np.arange(col_range_j, col_range_j + EXPERIMENTAL_DATA_COLUMNS_PER_SECTION)

        return sections

    def _process_sections(self,
                          key_ref: str,
                          sections: Dict[str, np.ndarray],
                          pd_exp_data: pd.DataFrame,
                          path_local: Path,
                          plot: bool,
                          ):
        """
        Process the sections of a test case.

        Args:
            key_ref (str): Key reference of the test.
            sections (Dict[str, range]): Sections to process.
            pd_exp_data (pd.DataFrame): Experimental data.
            path_local (Path): Local path for the test case.
            plot (bool): Whether to plot results.

        Return
        """

        dict_opt_res = {}
        dict_local_path = {}
        df_opt_mat_params = self.aorta_seq.copy(deep=True)
        instron_data_ref = self.instron_data[key_ref]

        for sec_key_i, col_range_i in sections.items():
            pd_section_data = pd_exp_data.iloc[:, col_range_i].dropna()

            sec_i = sec_key_i[:SECTION_CODE_LENGTH]
            pos_i = sec_key_i[POSITION_CODE_INDEX]

            if instron_data_ref.get(sec_i) is not None:
                instron_data_sec_i = instron_data_ref[sec_i]

                if instron_data_sec_i.get(pos_i) is not None:
                    logger.info(f'\n {key_ref}, Preprocessing, Section: {sec_i}, Position: {pos_i}')

                    ####################################################################
                    # Create section directory using PathManager
                    if dict_local_path.get(sec_i) is None:
                        section_dir_i = self._create_section_directory(path_local, sec_i)
                        dict_local_path[sec_i] = section_dir_i

                    ####################################################################
                    opt_solu_setup_i = self._sec_prep(pos_i,
                                                      pd_section_data,
                                                      instron_data_sec_i,
                                                      plot=plot)

                    df_opt_mat_params.loc[sec_key_i, "mlx"] = opt_solu_setup_i["instron"].np_tstretch_ref.max()
                    dict_opt_res[sec_key_i] = opt_solu_setup_i

        return df_opt_mat_params, dict_opt_res, dict_local_path

    def find_baseline_parameters(self,
                                 ftype: Optional[str] = None,
                                 miter: int = None,
                                 **kwargs,
                                 ) -> pd.DataFrame:
        """
        Find baseline (global) material parameters across all test cases.

        Performs global optimization to find a single set of material parameters
        that best fits all experimental data simultaneously. This is the primary
        method for material parameter identification in the fitting workflow.

        Parameters
        ----------
        ftype : str, optional
            Cost function type for least squares fitting.
            Options:
                - 'cauchy_robust': Robust Cauchy loss (default, recommended)
                - 'l2': Standard L2 norm (least squares)
                - 'huber': Huber loss (robust)
                - 'soft_l1': Soft L1 loss
            If None, uses 'cauchy_robust'.
                
        miter : int, optional
            Maximum number of iterations for local optimization.
            If None, uses DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS.
            Larger values allow more thorough convergence.
            
        **kwargs : dict
            Additional optimization parameters:
            
            rho : float, optional
                KS aggregation parameter for constraint handling.
                
            c : float, default=40.0
                Robustness parameter for robust loss functions.
                Smaller values = more robust to outliers.
                
            giter : int, default=3
                Number of global iterations (basin-hopping).
                
            alpha : float, optional
                Fiber angle constraint (radians).
                If provided, fixes fiber angle to this value.
                
            rescale : str, optional
                Stress rescaling method ('std', 'minmax', None).
                
            dvol : bool, default=True
                Include volumetric terms in optimization.
                
            epsilon : float, optional
                Regularization parameter for cost function.
                
            xi : np.ndarray, optional
                Initial parameter guess [mu, D, k1, k2, alpha, kappa].
                
            bh_step : str, default='random_displacement'
                Basin-hopping step strategy.
                
        Returns
        -------
        pd.DataFrame
            DataFrame with one row containing optimized baseline parameters.
            Columns: Parameter names (mu, D, k_1, k_2, alpha, kappa, etc.)
            Values: Optimized parameter values
            
        Notes
        -----
        This method:
        1. Aggregates all experimental data into single cost function
        2. Performs global optimization (if opt_glb=True)
        3. Returns single parameter set representing baseline tissue properties
        
        For section-specific parameters, use find_optimal_parameters() instead.
        
        Examples
        --------
        Basic usage with defaults:
        
        >>> fit = AnisoMaterialFit(selection, opt_type='ipopt', opt_glb=True)
        >>> df_baseline = fit.find_baseline_parameters()
        
        With custom settings:
        
        >>> df_baseline = fit.find_baseline_parameters(
        ...     ftype='cauchy_robust',
        ...     miter=100,
        ...     giter=5,
        ...     c=50.0
        ... )
        
        See Also
        --------
        find_optimal_parameters : Section-specific optimization
        """

        # --- Extract Optimization Parameters ---
        # KS function parameter
        if kwargs.get("rho") is not None:
            rho = kwargs.get("rho")
        else:
            rho = None

        # Cauchy Loss function parameter
        if kwargs.get("c") is not None:
            cm = kwargs.get("c")
        else:
            cm = None

        # L2: Adding Tikhonov Regularization
        if kwargs.get("alpha") is not None:
            alpha = kwargs.get("alpha")
        else:
            alpha = None

        # Regularization Rescaling method ('direct' or 'inverse'). Defaults to 'direct'.
        if kwargs.get("rescale") is not None:
            rescale = kwargs.get("rescale")
        else:
            rescale = None

        # Volume Regularization
        if kwargs.get("epsilon") is not None:
            epsilon = kwargs.get("epsilon")
        else:
            epsilon = None

        if kwargs.get("giter") is not None:
            giter = kwargs.get("giter")
        else:
            giter = DEFAULT_GLOBAL_ITERATIONS

        # Basinhopping Step parameter
        if kwargs.get("bh_step") is not None:
            bh_step = kwargs.get("bh_step")
        else:
            bh_step = "random_displacement"

        # Volume Stabilization Parameter
        if kwargs.get("dvol") is not None:
            dvol = kwargs.get("dvol")
        else:
            dvol = False
        
        # Set default miter if not provided
        if miter is None:
            miter = DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS

        # Cost function type
        if ftype is not None:
            self._baseline_ftype = ftype

        # --- Initialize Design Variables ---
        df_mat_params = self.ds_vars[["values", "lower", "upper", "limit", "variable"]].copy(deep=True)

        # Initial Guess for the design variable
        if kwargs.get("xi") is not None:
            xi = kwargs.get("xi")
            np_xi = np.asarray(xi)

            if df_mat_params["values"].values.shape == np_xi.shape:
                df_mat_params.loc[:, "values"] = np_xi

        # --- Run Global Optimization ---
        for key_rnm_i, val_i in self.model_opt_res.items():
            list_lsq_i = [val_ik['lsq'] for key_ik, val_ik in val_i.items()]
            lsq_fun_i = CostIntegrator(lsq_mat_fun=list_lsq_i,
                                       ftype=ftype,
                                       c=cm,
                                       rho=rho,
                                       alpha=alpha,
                                       rescale=rescale,
                                       vol_reg=dvol,
                                       epsilon=epsilon
                                       )

            opt_args_i = [self.opt_type, lsq_fun_i, df_mat_params.loc[lsq_fun_i.inp_mat_keys, :]]

            with (warnings.catch_warnings()):
                logger.info(f'\n ==> Solving Global Baseline Parameters: {key_rnm_i}')
                opt_res_i = opt_solvers(*opt_args_i,
                                        miter=miter,
                                        giter=giter,
                                        glb=self.opt_glb,
                                        bh_step=bh_step,
                                        )

                if len(opt_res_i.x) < self.ds_vars.index.shape[0]:
                    raise ValueError(
                        f"Error with Baseline optimization process: "
                        f"result length {len(opt_res_i.x)} < expected {self.ds_vars.index.shape[0]}"
                    )

                self.baseline_ds_vars.loc[key_rnm_i] = opt_res_i.series

        return self.baseline_ds_vars

    def _eval_problem(self,
                      min_lx: float,
                      opt_material: OptimizeResult,
                      instron_data: InstronData,
                      lsq: CostFunction,
                      plot_fname: str = "",
                      ncontrol: int = None,
                      ) -> Tuple[Dict[str, Any], pd.Series]:
        """
        Evaluate optimization problem and generate fit plots.

        Computes model solution with optimized parameters, generates comparison
        plots with experimental data, and extracts strain energy components.

        Parameters
        ----------
        min_lx : float
            Maximum stretch ratio to evaluate (λ_max)
        opt_material : OptimizeResult
            Optimization result containing fitted material parameters
        instron_data : InstronData
            Experimental data object with measured force-stretch curves
        lsq : CostFunction
            Cost function object for computing model predictions
        plot_fname : str, optional
            Filename for saving comparison plot, by default ""
        ncontrol : int, optional
            Number of points for solution evaluation. If None, uses self.ncontrol

        Returns
        -------
        opt_solution : Dict[str, Any]
            Dictionary containing:
            - 'pk1': First Piola-Kirchhoff stress components
            - 'cauchy': Cauchy stress components
            - 'ese': Strain energy density components
            - 'lx': Stretch ratios evaluated
        sr_dsvars : pd.Series
            Series of optimized material parameter values

        Notes
        -----
        This method:
        1. Sets bulk modulus if volumetric effects are disabled
        2. Computes model solution at specified stretch ratios
        3. Generates comparison plots (experimental vs model)
        4. Extracts and sums strain energy components

        The plot shows:
        - Experimental data points
        - Model prediction curve
        - Component breakdown (isotropic, anisotropic, volumetric)

        Strain Energy Components:
        - 'iso':    Isotropic (elastin) contribution
        - 'ani':    Anisotropic (collagen) contribution
        - 'vol':    Volumetric constraint/ contribution
        - 'full':   Total strain energy density

        See Also
        --------
        plot_material_fit : Generate comparison plots
        CostFunction.solve : Compute model solution
        """

        # --- Validate Inputs ---
        if ncontrol is None:
            ncontrol = self.ncontrol

        if not self.dvol:
            opt_material.series["bulk"] = self.bulk

        # --- Compute Model Solution ---
        kwargs_sol = {}
        if min_lx > UNSTRETCHED_STATE:
            np_lx = np.linspace(UNSTRETCHED_STATE, min_lx, num=ncontrol)
            kwargs_sol["stretch_x"] = np_lx

        opt_solution = lsq.solve(opt_material.series, **kwargs_sol)

        # --- Generate Comparison Plots ---
        plot_path = Path(plot_fname)
        fname = plot_path.stem
        path_loc = str(plot_path.parent)

        plot_material_fit(instron_data, opt_solution, path_loc, filename_prefix=fname)

        # --- Extract Strain Energy Components ---
        for key_l in self.ese_types:
            if opt_solution["ese"].get(key_l) is not None:
                opt_material[key_l] = sum(opt_solution["ese"][key_l])

        return opt_solution, opt_material.series

    def _configure_optimization_parameters(self,
                                        miter: Optional[int],
                                        ftype: Optional[str],
                                        sections: Optional[List[str]],
                                        **kwargs,
                                           ) -> Dict[str, Any]:
        """
        Configure and validate optimization parameters.
        
        Extracts optimization settings from kwargs, applies defaults,
        and validates parameter ranges. Centralizes all parameter handling
        for reproducibility and maintainability.
        
        Parameters
        ----------
        miter : int, optional
            Maximum iterations for local optimization
        ftype : str, optional
            Cost function type
        sections : List[str], optional
            Sections to optimize (if None, optimizes all)
        **kwargs : dict
            Additional optimization parameters
            
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary with keys:
            - miter: int - Maximum iterations
            - ftype: str - Cost function type
            - sections: List[str] - Sections to optimize
            - cm: float - Cauchy parameter (c)
            - rho: float - KS aggregation parameter
            - alpha: float - Tikhonov regularization
            - beta: float - Rescaling regularization parameter
            - rescale: str - Rescaling method
            - epsilon: float - Volume regularization
            - giter: int - Global iterations (basin-hopping)
            - dvol: bool - Volume stabilization flag
            
        Notes
        -----
        - Applies sensible defaults from module constants
        - Validates parameter ranges
        - Logs configuration for reproducibility
        - Updates instance variable self._local_ftype if ftype provided

        """
        # --- Initialize Configuration ---
        config = {'miter': miter if miter is not None else DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS,
                  'giter': kwargs.get('giter', DEFAULT_GLOBAL_ITERATIONS)}
        
        # --- Cost Function Settings ---
        if ftype is not None:
            self._local_ftype = ftype

        config['ftype'] = ftype
        
        # --- Loss Function Parameters ---
        config['cm'] = kwargs.get('c')           # Cauchy parameter
        config['rho'] = kwargs.get('rho')        # KS function parameter
        config['alpha'] = kwargs.get('alpha')    # Tikhonov regularization
        config['beta'] = kwargs.get('beta')      # Rescaling regularization
        config['rescale'] = kwargs.get('rescale')  # Rescaling method
        config['epsilon'] = kwargs.get('epsilon')  # Volume regularization
        
        # --- Stabilization Settings ---
        config['dvol'] = kwargs.get('dvol', False)
        
        # --- Section Configuration ---
        if sections is None:
            config['sections'] = [f"{sec}-{pos}" for sec in ["Ar", "Tr", "Ab"] for pos in ["A", "B", "C"]]
        else:
            config['sections'] = sections
        
        # --- Validation ---
        if config['miter'] < 1:
            raise ValueError(f"miter must be positive integer, got {config['miter']}")
        
        if config['giter'] < 1:
            raise ValueError(f"giter must be positive integer, got {config['giter']}")
        
        # Log configuration for reproducibility
        logger.info(f"Optimization configuration: miter={config['miter']}, "
                   f"giter={config['giter']}, sections={len(config['sections'])}, "
                   f"ftype={config['ftype']}, dvol={config['dvol']}")
        
        return config

    def _get_local_solution_path(self, rat_id: str, local_path: Optional[str] = None) -> Path:
        """
        Get absolute path for saving local optimization solutions.
        
        Constructs the full path for saving optimization results, optionally
        appending a subdirectory to organize multiple optimization runs.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'rato-17')
        local_path : str, optional
            Subdirectory name within the rat's solution path.
            If None, uses the base solution path.
            Useful for organizing multiple optimization iterations.
            
        Returns
        -------
        Path
            Absolute path for saving results. Resolved to avoid symlink issues.
        
        Notes
        -----
        - Always returns absolute path via resolve()
        - Creates parent directories if needed (via ensure_directory_exists)
        - Path object allows easy concatenation with / operator
        
        See Also
        --------
        PathManager : Central path management system
        solution_path : Dictionary of rat solution paths
        """
        base_path = self.solution_path[rat_id]
        if local_path is not None:
            return (base_path / local_path).resolve()

        return base_path

    @staticmethod
    def _prepare_cost_and_data(rat_data: Dict[str, Any],
                               df_mat_params: pd.DataFrame,
                               ) -> Tuple[List[CostFunction], List[InstronData]]:
        """
        Prepare cost functions and experimental data for optimization.
        
        Extracts cost function and experimental data objects from rat-specific
        data, updating cost functions with current material parameters.
        
        Parameters
        ----------
        rat_data : Dict[str, Any]
            Rat-specific data dictionary from model_opt_res containing:
            - Keys: Section-position codes (e.g., 'Ar-A', 'Tr-B')
            - Values: Dict with 'lsq', 'instron', 'ext_solution' keys
        df_mat_params : pd.DataFrame
            Material parameter DataFrame with current baseline values.
            Used to update cost function variables.
            
        Returns
        -------
        cost_functions : List[CostFunction]
            List of cost function objects, updated with current parameters.
            Order matches the iteration order of rat_data.values()
        instron_data_list : List[InstronData]
            List of experimental data objects.
            Order matches cost_functions list.
            
        Notes
        -----
        This method:
        1. Iterates through rat_data values
        2. Updates each cost function with df_mat_params
        3. Collects updated cost functions in order
        4. Collects corresponding experimental data
        
        The order preservation is critical - cost_functions[i] corresponds
        to instron_data_list[i].
        
        Side Effects
        ------------
        - Modifies cost function objects in rat_data via update_variables()
        
        See Also
        --------
        CostFunction.update_variables : Update cost function parameters
        model_opt_res : Dictionary of per-rat optimization data
        """
        list_cost_functions = []
        list_instron_data = []
        
        for val_k in rat_data.values():
            # Update cost function with current material parameters
            val_k['lsq'].update_variables(df_mat_params)
            list_cost_functions.append(val_k['lsq'])
            list_instron_data.append(val_k['instron'])
        
        return list_cost_functions, list_instron_data

    def _optimize_single_position(self,
                                  rat_id: str,
                                  section_key: str,
                                  section_name: str,
                                  position_code: str,
                                  cost_function: CostFunction,
                                  instron_data: InstronData,
                                  max_stretch: float,
                                  df_mat_params: pd.DataFrame,
                                  local_solution_path: Path,
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize material parameters for a single section-position.
        
        This is the atomic optimization unit, handling cost function setup,
        optimization execution, result evaluation, and plot generation for
        one anatomical position.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'rato-17') for logging
        section_key : str
            Full section-position key (e.g., 'Ar-A', 'Tr-B')
        section_name : str
            Section code only (e.g., 'Ar', 'Tr', 'Ab')
        position_code : str
            Position letter ('A', 'B', or 'C')
        cost_function : CostFunction
            Cost function object for this position
        instron_data : InstronData
            Experimental data for this position
        max_stretch : float
            Maximum stretch ratio to evaluate (λ_max)
        df_mat_params : pd.DataFrame
            Material parameter DataFrame (baseline values)
        local_solution_path : Path
            Base path for saving results
        config : Dict[str, Any]
            Optimization configuration from _configure_optimization_parameters()
            
        Returns
        -------
        Dict[str, Any]
            Result dictionary with keys:
            - 'pk1': First Piola-Kirchhoff stress components
            - 'cauchy': Cauchy stress components  
            - 'ese': Strain energy density components
            - 'lx': Stretch ratios evaluated
            - 'mat_param': Optimized material parameters (pd.Series)
            
        Notes
        -----
        Processing steps:
        1. Log optimization start
        2. Create CostIntegrator with single cost function
        3. Run optimization (or use baseline if not in config['sections'])
        4. Evaluate model at specified stretch ratios
        5. Generate comparison plot
        6. Return complete results dictionary
        
        The method decides whether to optimize based on config['sections']:
        - If section_key in list: Run full optimization
        - Otherwise: Use baseline parameters (empty=True flag)
        
        Side Effects
        ------------
        - Generates PNG plot file in section subdirectory
        - Logs optimization progress
        
        See Also
        --------
        CostIntegrator : Aggregates cost functions
        opt_solvers : Optimization solver interface
        _eval_problem : Evaluates and plots optimization result
        """
        # --- Log and Initialize ---
        logger.info(f'\n {rat_id}, Solving Local Parameters, Section: {section_name}, Position: {position_code}')
        
        df_mat_params_copy = df_mat_params.copy(deep=True)
        
        # --- Setup Cost Integrator ---
        integrator = CostIntegrator(
            [cost_function],
            ftype=config['ftype'],
            c=config['cm'],
            rho=config['rho'],
            alpha=config['alpha'],
            epsilon=config['epsilon'],
            beta=config['beta'],
            rescale=config['rescale'],
            vol_reg=config['dvol']
        )
        
        # --- Prepare Optimization Arguments ---
        opt_params = df_mat_params_copy.loc[integrator.inp_mat_keys, :].copy(deep=True)
        opt_args = [self.opt_type, integrator, opt_params]
        
        # --- Run Optimization ---
        if section_key in config['sections']:
            opt_result = opt_solvers(
                *opt_args, 
                miter=config['miter'], 
                giter=config['giter'], 
                glb=False
            )
        else:
            # Use baseline parameters without optimization
            opt_result = opt_solvers(
                *opt_args, 
                miter=config['miter'], 
                giter=config['giter'], 
                empty=True
            )
        
        # --- Generate Results and Plot ---
        plot_path = (local_solution_path / section_name / f"{self.opt_type}_{position_code}.png").resolve()
        
        # Evaluate model and generate plot
        result_dict, optimized_params = self._eval_problem(
            min_lx=max_stretch,
            opt_material=opt_result,
            instron_data=instron_data,
            lsq=cost_function,
            plot_fname=str(plot_path),
            ncontrol=HIGH_RESOLUTION_NCONTROL
        )
        
        # Add optimized parameters to result
        result_dict["mat_param"] = optimized_params
        
        return result_dict

    def _optimize_section_positions(self,
                                    rat_id: str,
                                    section_info: pd.DataFrame,
                                    cost_functions: List[CostFunction],
                                    instron_data_list: List[InstronData],
                                    max_stretches: pd.Series,
                                    df_mat_params: pd.DataFrame,
                                    local_solution_path: Path,
                                    config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all section-position combinations for a rat.
        
        Iterates through anatomical sections (Ar, Tr, Ab) and their positions
        (A, B, C), running optimization for each valid combination using the
        atomic optimization method.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'rato-17') for logging
        section_info : pd.DataFrame
            Boolean DataFrame indicating which section-positions are valid.
            Index: section names (e.g., 'Ar', 'Tr', 'Ab')
            Columns: position codes (e.g., 'Ar-A', 'Tr-B', 'Ab-C')
            Values: True if position should be optimized, False otherwise
        cost_functions : List[CostFunction]
            List of cost function objects, one per section-position.
            Order matches the iteration order of section_info.
        instron_data_list : List[InstronData]
            List of experimental data objects, one per section-position.
            Order matches cost_functions.
        max_stretches : pd.Series
            Maximum stretch ratios (λ_max) for each section-position.
            Index: section-position codes (e.g., 'Ar-A', 'Tr-B')
        df_mat_params : pd.DataFrame
            Material parameter DataFrame with baseline values
        local_solution_path : Path
            Base path for saving optimization results
        config : Dict[str, Any]
            Optimization configuration from _configure_optimization_parameters()
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Nested dictionary of results:
            - Keys: section-position codes (e.g., 'Ar-A', 'Tr-B')
            - Values: Result dictionaries from _optimize_single_position()
              containing 'pk1', 'cauchy', 'ese', 'lx', 'mat_param'
            
        Notes
        -----
        Iteration Strategy:
        1. Outer loop: Anatomical sections (rows of section_info)
        2. Inner loop: Positions within each section (columns)
        3. For each valid position (val_k == True):
           - Call _optimize_single_position()
           - Store results in output dictionary
        
        The section_info DataFrame controls which positions are processed.
        Invalid positions (val_k == False) are skipped automatically.
        
        Index Correspondence:
        - `ijk` is the linear index into cost_functions and instron_data_list
        - Increments for each column in section_info, row by row
        - Example: For 3 sections × 3 positions, ijk goes 0→8
        
        Side Effects
        ------------
        - Generates plot files via _optimize_single_position()
        - Updates self.optimal_ds_vars with optimized parameters
        - Logs progress for each position
        
        Examples
        --------
        >>> results = self._optimize_section_positions(
        ...     rat_id='rato-17',
        ...     section_info=df_sec_info,
        ...     cost_functions=cost_fns,
        ...     instron_data_list=instron_list,
        ...     max_stretches=max_stretch_series,
        ...     df_mat_params=params_df,
        ...     local_solution_path=Path('/results'),
        ...     config=config
        ... )
        >>> len(results)  # Number of optimized positions
        9
        >>> 'Ar-A' in results
        True
        
        See Also
        --------
        _optimize_single_position : Atomic optimization for one position
        _get_index : Generates section_info DataFrame
        """
        # --- Initialize Results Container ---
        model_results = {}
        
        # --- Iterate Over Sections and Positions ---
        for ij, (section_name, position_row) in enumerate(section_info.iterrows()):
            for ijk, (position_key, is_valid) in enumerate(position_row.items()):
                if is_valid:
                    # --- Optimize Single Position ---
                    result_dict = self._optimize_single_position(
                        rat_id=rat_id,
                        section_key=position_key,
                        section_name=section_name,
                        position_code=position_key[-1],
                        cost_function=cost_functions[ijk],
                        instron_data=instron_data_list[ijk],
                        max_stretch=max_stretches[position_key].item(),
                        df_mat_params=df_mat_params,
                        local_solution_path=local_solution_path,
                        config=config
                    )
                    
                    # --- Store Results ---
                    model_results[position_key] = result_dict
                    self.optimal_ds_vars[rat_id].loc[position_key, result_dict["mat_param"].index] = result_dict["mat_param"]
        
        return model_results

    def _save_rat_optimization_results(self,
                                       rat_id: str,
                                       model_results: Dict[str, Dict[str, Any]],
                                       df_mat_params: pd.DataFrame,
                                       local_solution_path: Path,
                                       plot_ese: bool = False) -> None:
        """
        Save and plot optimization results for a single rat.
        
        Performs post-optimization tasks: saves parameter data to files,
        generates comparison plots (stress, force, energy patterns), and
        stores results in instance variables.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'rato-17')
        model_results : Dict[str, Dict[str, Any]]
            Optimization results from _optimize_section_positions()
            Keys: section-position codes
            Values: Result dictionaries with stress/strain data
        df_mat_params : pd.DataFrame
            Material parameter DataFrame with baseline values
        local_solution_path : Path
            Path for saving result files
        plot_ese : bool, optional
            If True, generate strain energy pattern plot, by default False
            
        Side Effects
        ------------
        - Saves Excel and Parquet files with optimized parameters
        - Generates PNG plot files (stress, force, optionally energy)
        - Updates self.aorta_model_results dictionary
        - Closes matplotlib figures
        
        Generated Files
        ---------------
        - opt_mat_param_{opt_type}.xlsx : Parameter table
        - opt_mat_param_{opt_type}.parquet.gzip : Compressed data
        - {opt_type}_pk1_stress_*.png : Stress component plots
        - {opt_type}_uniaxial_test.png : Force-stretch curves
        - energy_strain_pattern_{opt_type}.png : Energy plots (if plot_ese=True)
        
        Notes
        -----
        Data Saving Strategy:
        1. Concatenate baseline parameters with optimized values
        2. Save to both Excel (human-readable) and Parquet (efficient)
        3. Generate comparison plots showing experimental vs fitted data
        
        Plot Generation:
        - Stress plots: PK1 stress components vs stretch
        - Force plots: Force vs stretch for uniaxial test
        - Energy plots: Strain energy density patterns (optional)
        
        Examples
        --------
        >>> self._save_rat_optimization_results(
        ...     rat_id='rato-17',
        ...     model_results=results_dict,
        ...     df_mat_params=params_df,
        ...     local_solution_path=Path('/results/rato-17'),
        ...     plot_ese=True
        ... )
        # Files created in /results/rato-17/
        
        See Also
        --------
        _save_data : Low-level file saving method
        _setup_plot_fit : Prepares data for plotting
        _plot_stress : Generates stress component plots
        exp_test_plot : Generates force-stretch plots
        _plot_ese : Generates energy pattern plots
        """
        # --- Store Results ---
        if len(model_results) > 0:
            self.aorta_model_results[rat_id] = model_results
        
        # --- Prepare Combined DataFrame ---
        df_baseline = pd.concat([
            self.baseline_ds_vars.loc[rat_id].to_frame(name='baseline'),
            df_mat_params[["lower", "upper"]]
        ], axis=1)
        
        df_opt_combined = pd.concat([df_baseline.T, self.optimal_ds_vars[rat_id]], axis=0)
        
        # --- Save Data Files ---
        self._save_data(local_solution_path, df_opt_combined)
        
        # --- Generate Plots ---
        plot_data = self._setup_plot_fit({rat_id: model_results})
        
        # Stress component plots
        self._plot_stress(plot_data, DEFAULT_PLOT_LIMITS, str(self.solution_path[rat_id]))
        
        # Force-stretch plot
        fig_force = exp_test_plot(plot_data, limits=DEFAULT_PLOT_LIMITS, lines=self.plt_lines)
        fig_force.savefig(f'{local_solution_path}/{self.opt_type}_uniaxial_test.png')
        plt.close(fig_force)
        
        # --- Optional Energy Pattern Plot ---
        if plot_ese:
            fig_energy = self._plot_ese(model_results, self.optimal_ds_vars[rat_id])
            fig_energy.savefig(f'{local_solution_path}/energy_strain_pattern_{self.opt_type}.png')
            plt.close(fig_energy)

    def _generate_aggregate_plots(self, full_solution: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Generate aggregate plots combining all rats.
        
        Creates summary plots showing combined results from all optimized rats,
        useful for cross-rat comparison and overall assessment.
        
        Parameters
        ----------
        full_solution : Dict[str, Dict[str, Dict[str, Any]]]
            Complete optimization results for all rats.
            Structure:
            {
                'rato-17': {
                    'Ar-A': {result_dict},
                    'Tr-B': {result_dict},
                    ...
                },
                'rato-18': {...},
                ...
            }
            
        Side Effects
        ------------
        - Generates aggregate PNG plot files in main results directory
        - Closes matplotlib figures after saving
        
        Generated Files
        ---------------
        - {opt_type}_uniaxial_test.png : Combined force-stretch curves
        - {opt_type}_pk1_stress_*.png : Combined stress component plots
        
        Notes
        -----
        Aggregate plots overlay data from all rats, allowing visual comparison
        of material behavior across specimens. Useful for identifying outliers
        and assessing parameter consistency.
        
        Examples
        --------
        >>> full_sol = {
        ...     'rato-17': {...},
        ...     'rato-18': {...}
        ... }
        >>> self._generate_aggregate_plots(full_sol)
        # Files created in self.path_main/
        
        See Also
        --------
        _setup_plot_fit : Prepares combined data for plotting
        exp_test_plot : Generates force-stretch plots
        _plot_stress : Generates stress component plots
        """
        # Setup combined plot data
        plot_data = self._setup_plot_fit(full_solution)
        
        # Aggregate force-stretch plot
        fig_force = exp_test_plot(plot_data, limits=DEFAULT_PLOT_LIMITS, lines=self.plt_lines)
        fig_force.savefig(f'{self.path_main}/{self.opt_type}_uniaxial_test.png')
        plt.close(fig_force)
        
        # Aggregate stress plots
        self._plot_stress(plot_data, DEFAULT_PLOT_LIMITS, str(self.path_main))

    def find_optimal_parameters(self,
                                miter: int = None,
                                plot: bool = False,
                                ftype: str = None,
                                sections: List[str] = None,
                                local_path: str = None,
                                **kwargs
                                ) -> pd.DataFrame:
        """
        Find optimal section-specific material parameters.

        Performs local optimization for each anatomical section and position,
        allowing parameters to vary spatially along the aorta. This captures
        regional heterogeneity in material properties.

        Parameters
        ----------
        miter : int, optional
            Maximum iterations for local optimization.
            If None, uses DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS (100).
        plot : bool, optional
            If True, generate strain energy pattern plots, by default False
        ftype : str, optional
            Cost function type for optimization:
            - 'lsq': Standard least squares
            - 'lsq_sum': Sum of least squares across sections
            - 'cauchy_robust': Robust Cauchy loss
            If None, uses self._local_ftype
        sections : List[str], optional
            List of section-position keys to optimize (e.g., ['Ar-A', 'Tr-B']).
            If None, optimizes all sections: Ar/Tr/Ab × A/B/C = 9 combinations
        local_path : str, optional
            Subdirectory within solution path for saving results.
            If None, saves directly to solution path
        **kwargs : dict
            Additional optimization parameters:
            - c (float):        Robustness parameter for robust loss functions
            - rho (float):      KS aggregation parameter
            - alpha (float):    Tikhonov regularization parameter
            - beta (float):     Re-scaling regularization parameter
            - rescale (str):    Rescaling method ('std', 'minmax', None)
            - epsilon (float):  Volumetric regularization parameter
            - giter (int):      Global iterations (basin-hopping)
            - dvol (bool):      Include volumetric stabilization

        Returns
        -------
        pd.DataFrame
            Empty DataFrame (optimization results stored in self.optimal_ds_vars)

        Notes
        -----
        Optimization Strategy:
        1. Uses baseline parameters as initial guess for each section
        2. Optimizes each section-position independently
        3. Generates comparison plots (experimental vs fitted)
        4. Saves results (Excel + Parquet formats)

        The method processes sections in order: Ar → Tr → Ab, with positions
        A, B, C for each. Only sections in the `sections` list are optimized;
        others use baseline parameter values.

        Generated Files (per rat):
        - opt_mat_param_{opt_type}.xlsx: Parameter table
        - opt_mat_param_{opt_type}.parquet.gzip: Compressed data
        - {opt_type}_pk1_stress_{section}.png: Stress plots
        - {opt_type}_uniaxial_test.png: Force-stretch plots

        See Also
        --------
        find_baseline_parameters : Global parameter optimization
        _eval_problem : Evaluate and plot single optimization
        _save_data : Save optimization results

        Examples
        --------
        Optimize all sections:

        >>> fit = AnisoMaterialFit(selection, opt_type='ipopt')
        >>> fit.exp_test_eval()
        >>> fit.find_baseline_parameters()  # Get initial guess
        >>> fit.find_optimal_parameters(miter=100, plot=True)

        Optimize specific sections only:

        >>> sections_to_fit = ['Ar-A', 'Ar-B', 'Tr-A']
        >>> fit.find_optimal_parameters(sections=sections_to_fit, miter=50)
        """

        # --- Configure Optimization Parameters ---
        config = self._configure_optimization_parameters(miter, ftype, sections, **kwargs)

        # --- Initialize Output Variables ---
        df_opt_ds_vars_cfg_i = pd.DataFrame()
        df_mat_params = self.ds_vars[["values", "lower", "upper", "limit", "variable"]]
        full_solution = {}

        # --- Optimize Each Rat ---
        for key_i, val_i in self.model_opt_res.items():
            df_mat_params_i = df_mat_params.copy(deep=True)
            df_mat_params_i.loc[:, "values"] = self.baseline_ds_vars.loc[key_i, :]

            # --- Setup Paths ---
            local_solution_path_i = self._get_local_solution_path(key_i, local_path)

            df_sec_info_i = self._get_index(val_i)

            # --- Prepare Cost Functions ---
            list_lsq_i, list_instron_i = self._prepare_cost_and_data(val_i, df_mat_params_i)

            min_mlx_i = self.optimal_ds_vars[key_i].loc[:, "mlx"]
            
            # --- Optimize Section Positions ---
            model_res_ij = self._optimize_section_positions(
                rat_id=key_i,
                section_info=df_sec_info_i,
                cost_functions=list_lsq_i,
                instron_data_list=list_instron_i,
                max_stretches=min_mlx_i,
                df_mat_params=df_mat_params_i,
                local_solution_path=local_solution_path_i,
                config=config
            )

            # --- Save Results ---
            self._save_rat_optimization_results(
                rat_id=key_i,
                model_results=model_res_ij,
                df_mat_params=df_mat_params_i,
                local_solution_path=local_solution_path_i,
                plot_ese=plot
            )
            
            full_solution[key_i] = model_res_ij

        # --- Generate Aggregate Plots ---
        self._generate_aggregate_plots(full_solution)

        return df_opt_ds_vars_cfg_i

    def _save_data(self, file_path: PathLike, dsvars: pd.DataFrame) -> None:
        """
        Save material parameters to multiple file formats.

        Exports optimization results in both Excel and compressed Parquet formats
        for different use cases (human-readable vs efficient storage).

        Parameters
        ----------
        file_path : PathLike
            Base directory path for saving files (str or Path)
        dsvars : pd.DataFrame
            DataFrame containing material parameters with columns:
            - Parameter names as index
            - 'Baseline': Baseline parameter values
            - Section-specific values (e.g., 'Ar-A', 'Tr-B')
            - 'lower', 'upper': Parameter bounds

        Returns
        -------
        None
            Files are saved to disk:
            - {file_path}/opt_mat_param_{opt_type}.xlsx
            - {file_path}/opt_mat_param_{opt_type}.parquet.gzip

        Notes
        -----
        File Formats:
        - Excel (.xlsx): Human-readable, good for inspection and sharing
        - Parquet (.gzip): Compressed binary, efficient for large datasets

        The directory is created automatically if it doesn't exist.

        Future Enhancement:
        - TODO: Add joblib format for complete solution objects

        See Also
        --------
        save_data : Public method for saving all results
        load_results : Load previously saved results
        """
        # Delegate to IO handler
        self.io_handler.save_optimization_results(dsvars, file_path)

    def save_data(self):
        """
        Save optimization results to disk.

        Saves all material parameters, optimization results, and model predictions
        to the configured output paths. Data is saved in Parquet format for
        efficient storage and loading.

        File Structure
        --------------
        For each rat ID, creates:
        - ``{rat_id}/optimal_dsvars.parquet``: Section-specific optimized parameters
        - ``{rat_id}/baseline_dsvars.parquet``: Baseline parameters across sections
        - Model results cached for quick loading

        Notes
        -----
        This method should be called after optimization to persist results.
        The saved data can be reloaded using load_results().

        Saved data includes:
        - Optimized material parameters (mu, D, k1, k2, alpha, kappa)
        - Parameter bounds (lower/upper)
        - Maximum stretch values for each section
        - Optimization metadata (iterations, convergence, cost)

        Examples
        --------
        After parameter optimization:
        
        >>> fit = AnisoMaterialFit(selection)
        >>> df_baseline = fit.find_baseline_parameters()
        >>> fit.find_optimal_parameters()
        >>> fit.save_data()  # Save all results
        
        See Also
        --------
        load_results : Load previously saved results
        _save_data : Internal method for saving individual files
        """

        for key_i in self.model_opt_res.keys():
            local_path_i = f'{self.solution_path[key_i]}'
            pd_baseline_dsvars_i = self.baseline_ds_vars.loc[key_i].to_frame(name='baseline')
            pd_opt_dsvars_i = pd.concat([pd_baseline_dsvars_i.T, self.optimal_ds_vars[key_i]], axis=0)

            self._save_data(local_path_i, pd_opt_dsvars_i)

    def load_results(self, run: bool = False):
        """
        Load previously saved optimization results from disk.

        Restores material parameters, optimization results, and optionally
        recomputes model solutions from saved parameters.

        Parameters
        ----------
        run : bool, default=False
            If True, recomputes model solutions using loaded parameters.
            If False, only loads parameters without recomputing solutions.
            
        Notes
        -----
        This method loads data saved by save_data(). It's useful for:
        - Resuming interrupted workflows
        - Generating plots from previous optimizations
        - Comparing results across different runs
        - Avoiding re-running expensive optimizations

        The method automatically detects available saved files and loads:
        - Optimal design variables (section-specific parameters)
        - Baseline design variables (average parameters)
        - Maximum stretch values for each section

        If run=True, also computes:
        - Stress-stretch curves for all sections
        - Model predictions at experimental stretch points
        - Comparison with experimental data

        Examples
        --------
        Load results without recomputing:
        
        >>> fit = AnisoMaterialFit(selection)
        >>> fit.load_results(run=False)
        >>> # Can now plot saved results
        >>> fit.plot_fit()
        
        Load and recompute solutions:
        
        >>> fit = AnisoMaterialFit(selection)
        >>> fit.load_results(run=True)
        >>> # Solutions recomputed with loaded parameters
        >>> fit.plot_fit()
        
        Raises
        ------
        FileNotFoundError
            If no saved results are found in the expected paths
            
        See Also
        --------
        save_data : Save optimization results
        """

        # --- Initialize Index List ---
        index_ds_vars = self.ds_vars.index.to_list()

        # --- Iterate Over Saved Results ---
        for rat_i, sec_info_i in self.path_local_solution.items():
            # --- Find Data Directory ---
            path_list = [Path(path_ik) for path_ik in sec_info_i.values()]
            if path_list:
                common_dir = path_list[0].parent
            else:
                continue

            joblib_file_i = common_dir / f"opt_xyz_mat_param_{self.opt_type}.joblib"
            parquet_file_i = common_dir / f"opt_mat_param_{self.opt_type}.parquet.gzip"

            # --- Load From File ---
            if joblib_file_i.is_file():
                logger.info(f" Not implemented load feature for joblib file: {str(joblib_file_i)}")

            elif parquet_file_i.is_file():
                # Use centralized loading function
                df_optimal_i = load_parquet_results(parquet_file_i, index_ds_vars)
                
                if df_optimal_i is None:
                    sr_optimal_baseline_i = self.ds_vars.loc[:, "ini"]
                    continue

                # --- Extract Baseline Parameters ---
                if 'mean' in df_optimal_i.index:
                    sr_optimal_baseline_i = df_optimal_i.loc["mean", index_ds_vars]
                    sr_optimal_baseline_i.name = "baseline"

                elif 'baseline' in df_optimal_i.index:
                    sr_optimal_baseline_i = df_optimal_i.loc["baseline", index_ds_vars]

                else:
                    sr_optimal_baseline_i = self.ds_vars["values"].copy()
                    sr_optimal_baseline_i.name = "baseline"
                    logger.warning(f"Baseline row not found in {str(parquet_file_i)}! Using 'ini' values.")

                # --- Update Instance State ---
                df_optimal_i = df_optimal_i[df_optimal_i.columns.intersection(self.optimal_ds_vars[rat_i].columns)]
                df_optimal_i = df_optimal_i.loc[df_optimal_i.index.intersection(self.optimal_ds_vars[rat_i].index)]

                self.optimal_ds_vars[rat_i].update(df_optimal_i)
                self.baseline_ds_vars.loc[rat_i, :] = sr_optimal_baseline_i

            else:
                sr_optimal_baseline_i = self.ds_vars.loc[:, "ini"]

            # --- Optionally Recompute Solutions ---
            model_res_i = {}

            for sec_key_k, models_k in self.model_opt_res[rat_i].items():
                if run:
                    mlx_k = self.optimal_ds_vars[rat_i].loc[sec_key_k, "mlx"]
                    np_lx = np.linspace(UNSTRETCHED_STATE, mlx_k, num=HIGH_RESOLUTION_NCONTROL)
                    model_res_i[sec_key_k] = models_k['lsq'].solve(sr_optimal_baseline_i, stretch_x=np_lx)

                self.aorta_model_results[rat_i] = model_res_i

        logger.info(" Previously saved results have been loaded!")

    def _setup_plot_fit(self, model_fit: Dict[str, Any]) -> Dict[str, Dict[str, List[Any]]]:
        """
        Organize model results for plotting by anatomical section.

        Groups model solutions and experimental data by section (Ar, Tr, Ab) to
        facilitate section-wise comparison plots.

        Parameters
        ----------
        model_fit : Dict[str, Any]
            Dictionary mapping rat IDs to their model results.
            Structure: {rat_id: {section_key: solution_dict}}

        Returns
        -------
        Dict[str, Dict[str, List[Any]]]
            Organized plot data with structure:
            {
                'Ar': {'data': [...], 'test': [...], 'name': [...]},
                'Tr': {'data': [...], 'test': [...], 'name': [...]},
                'Ab': {'data': [...], 'test': [...], 'name': [...]}
            }
            where:
            - 'data': List of model solution dictionaries
            - 'test': List of InstronData objects (experimental)
            - 'name': List of [rat_id, section_key] identifiers

        Notes
        -----
        This organization allows plotting all data from a specific anatomical
        section together, making it easy to compare:
        - Different specimens from the same section
        - Different positions (A, B, C) within a section
        - Model predictions vs experimental data

        The grouping is based on section prefix:
        - 'Ar': Ascending aorta (root)
        - 'Tr': Transverse arch
        - 'Ab': Abdominal aorta

        See Also
        --------
        _plot_stress : Generate stress plots for organized data
        stress_plot : Plotting function for stress components
        """

        plot_solution = {key_i: {"data": [], "test": [], "name": []} for key_i in ['Ar', 'Tr', 'Ab']}

        for seg_k in ['Ar', 'Tr', 'Ab']:
            for key_i, model_i in model_fit.items():
                for sec_j, solu_j in model_i.items():
                    if seg_k in sec_j:
                        plot_solution[seg_k]["name"].append([key_i, sec_j])
                        plot_solution[seg_k]["data"].append(solu_j)
                        plot_solution[seg_k]["test"].append(self.model_opt_res[key_i][sec_j]["instron"])

        return plot_solution

    def _plot_stress(self,
                     plot_solution: Dict[str, Dict[str, List[Any]]],
                     plot_limits: Dict[str, List[float]],
                     plot_path: str,
                     ) -> None:
        """
        Generate and save stress component plots for each anatomical section.

        Creates separate plots showing PK1 stress components (isotropic,
        anisotropic, total) for each section and saves them to disk.

        Parameters
        ----------
        plot_solution : Dict[str, Dict[str, List[Any]]]
            Organized plot data from _setup_plot_fit(), grouped by section
        plot_limits : Dict[str, List[float]]
            Axis limits for stress plots with keys:
            - 'iso': [min, max] for isotropic stress [MPa]
            - 'ani': [min, max] for anisotropic stress [MPa]
            - 'sum': [min, max] for total stress [MPa]
            - 'lx': [min, max] for stretch ratio
        plot_path : str
            Directory path for saving plot files

        Returns
        -------
        None
            Plots are saved as PNG files:
            - {plot_path}/{opt_type}_pk1_stress_Ar.png
            - {plot_path}/{opt_type}_pk1_stress_Tr.png
            - {plot_path}/{opt_type}_pk1_stress_Ab.png

        Notes
        -----
        Each plot shows:
        - Experimental data points (markers)
        - Model predictions (lines with different styles for positions A/B/C)
        - Three subplots: isotropic, anisotropic, and total stress
        - Stretch ratio on x-axis, stress on y-axis

        Line styles are defined in self.plt_lines:
        - 'A': solid line
        - 'B': dashed line
        - 'C': dash-dot line

        See Also
        --------
        stress_plot : Core plotting function
        NAME_SECTIONS : Dictionary mapping section codes to full names
        """

        for seg_k, solution_k in plot_solution.items():
            fig_stress_k = stress_plot(solution_k,
                                       limits=DEFAULT_PLOT_LIMITS,
                                       ptitle=f'{NAME_SECTIONS[seg_k]} Segments',
                                       lines=self.plt_lines)

            plot_fname_k = f'{plot_path}/{self.opt_type}_pk1_stress_{seg_k}.png'
            fig_stress_k.savefig(plot_fname_k)
            plt.close(fig_stress_k)

    def plot_fit(self,
                 global_opt: bool = False,
                 plot_path: Optional[str] = None,
                 ):
        """
        Generate plots comparing experimental data with model predictions.

        Creates comprehensive visualization of the fitting results including:
        - Stress-stretch curves for each anatomical section
        - Experimental data points vs. model predictions
        - Section-wise and combined plots

        Parameters
        ----------
        global_opt : bool, default=False
            If True, generates plots using baseline (global) parameters.
            If False, uses section-specific optimized parameters.
            
        plot_path : str, optional
            Custom directory path for saving plots.
            If None, uses self.path_main.
            Directory is created if it doesn't exist.
            
        Notes
        -----
        Generated Plots
        ---------------
        For each anatomical section (Ar, Tr, Ab):
        
        1. **PK1 Stress Plots**: ``{opt_type}_pk1_stress_{section}.png``
           - Shows First Piola-Kirchhoff stress components
           - Includes isotropic, anisotropic, and total contributions
           
        2. **Force-Stretch Plots**: ``{opt_type}_uniaxial_test.png``
           - Compares experimental force data with model predictions
           - One subplot per anatomical section
           - Includes all specimen positions (A, B, C)
        
        3. **Global Fit Plots** (if global_opt=True):
           - Shows quality of baseline parameter fit across all sections
           - Saved as ``{rat_id}_{opt_type}_uniaxial_glb_test.png``

        Plot Features
        -------------
        - Experimental data: Markers/points
        - Model predictions: Solid lines
        - Different colors for different specimens
        - Anatomical section labels
        - Stress units: MPa or kPa
        - Stretch units: mm/mm (dimensionless)

        Examples
        --------
        Plot with section-specific parameters:
        
        >>> fit = AnisoMaterialFit(selection)
        >>> fit.find_optimal_parameters()
        >>> fit.plot_fit(global_opt=False)
        
        Plot with baseline parameters:
        
        >>> fit.find_baseline_parameters()
        >>> fit.plot_fit(global_opt=True)
        
        Custom output location:
        
        >>> fit.plot_fit(plot_path='/path/to/plots', global_opt=False)
        
        See Also
        --------
        _plot_stress : Generate stress component plots
        _setup_plot_fit : Prepare plot data structure
        
        References
        ----------
        .. [1] NAME_SECTIONS dict maps section codes to full names:
               'Ar' → 'AoA' (Ascending Aorta)
               'Tr' → 'DTAo' (Descending Thoracic Aorta)
               'Ab' → 'DAAo' (Descending Abdominal Aorta)
        """

        # --- Evaluate Global Model ---
        if global_opt:
            for rat_i, sec_info_i in self.model_opt_res.items():
                # --- Setup Output Path ---
                if plot_path is not None:
                    fig_path_loc_i = Path(plot_path) / f'{rat_i}_{self.opt_type}_uniaxial_glb_test.png'
                else:
                    fig_path_loc_i = list(self.solution_path.values())[-1] / f'{self.opt_type}_uniaxial_glb_test.png'

                self.path_manager.ensure_parent_dir(fig_path_loc_i)
                ds_baseline_vars_i = self.baseline_ds_vars.loc[rat_i, :]

                # --- Compute Model Solutions ---
                if self.aorta_model_results.get(rat_i) is None:
                    ext_solu_j = {}
                else:
                    ext_solu_j = self.aorta_model_results[rat_i]

                for sec_j, solu_j in sec_info_i.items():
                    np_lx_j = np.linspace(UNSTRETCHED_STATE, self.optimal_ds_vars[rat_i].loc[sec_j, "mlx"], num=HIGH_RESOLUTION_NCONTROL)
                    ext_solu_j[sec_j] = solu_j['ext_solution'].solve(ds_baseline_vars_i, np_lx_j)

                self.aorta_model_results[rat_i] = ext_solu_j

                # --- Generate Global Fit Plot ---
                plot_solu_i = self._setup_plot_fit({rat_i: ext_solu_j})
                try:
                    fig_force_i = exp_test_plot(plot_solu_i, limits=DEFAULT_PLOT_LIMITS, lines=self.plt_lines)
                    fig_force_i.savefig(fig_path_loc_i)
                    plt.close(fig_force_i)
                except (ValueError, TypeError, IOError, OSError) as e:
                    logger.info(f" An error occurred while plotting the global fit for {rat_i}: {e}")
                    continue

        # --- Setup Plot Output ---
        fig_name = f'{self.opt_type}_uniaxial_test.png'

        if plot_path is None:
            plot_path = self.path_main

        plot_path = Path(plot_path)
        fig_path_loc = plot_path / fig_name
        plot_solution = self._setup_plot_fit(self.aorta_model_results)

        # --- Generate Stress Component Plots ---
        self._plot_stress(plot_solution, DEFAULT_PLOT_LIMITS, str(plot_path))

        # --- Generate Force-Stretch Plot ---
        fig_force = exp_test_plot(plot_solution, limits=DEFAULT_PLOT_LIMITS, lines=self.plt_lines)
        fig_force.savefig(fig_path_loc)
        plt.close(fig_force)

    def correlation(self, plot: bool = True) -> None:
        """
        Compute parameter correlations (not yet implemented).

        This method will analyze correlations between optimized material
        parameters across different sections and specimens.

        Parameters
        ----------
        plot : bool, optional
            If True, generate correlation plots, by default True

        Raises
        ------
        NotImplementedError
            This method is not yet implemented

        Notes
        -----
        Planned functionality:
        - Compute pairwise correlation matrix for material parameters
        - Identify highly correlated parameters (redundancy)
        - Generate correlation heatmaps
        - Analyze parameter sensitivity and identifiability

        This analysis would be useful for:
        - Understanding parameter coupling
        - Reducing model complexity
        - Improving optimization strategies
        - Validating parameter uniqueness
        """
        raise NotImplementedError
