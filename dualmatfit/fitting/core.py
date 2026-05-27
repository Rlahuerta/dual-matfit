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

from dualmatfit.data.rato_info import excel_data
from dualmatfit.formulation.variational import VariationalFormulation, ring_geom
from dualmatfit.solvers.extension import ExtensionSolution
from dualmatfit.optimization.cost import CostFunction, CostIntegrator
from dualmatfit.optimization.drivers import opt_solvers
from dualmatfit.data.experimental import InstronData, MaterialSetup
from dualmatfit.plotting.experimental_visuals import stress_plot, exp_test_plot, plot_material_fit
from dualmatfit.plotting.parameters import NAME_SECTIONS
from dualmatfit.utils.path_manager import PathManager, PathConfiguration, PathLike
from dualmatfit.utils.io_utils import MaterialFitIO, load_parquet_results
from dualmatfit.fitting.covariance import (
    robust_covariance_from_cost,
    covariance_from_gauss_newton,
    save_covariance_report,
    CovarianceReport,
)
from dualmatfit.fitting.constants import *  # noqa: F401,F403
from dualmatfit.fitting.optimization import FitOptimizationMixin
from dualmatfit.fitting.persistence import FitPersistenceMixin
from dualmatfit.fitting.visualization import FitVisualizationMixin

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('fitting')

__all__ = [
    'AnisoModelSolve',
    'AnisoMaterialFit',
]

# Suppress specific noisy warnings, but allow deprecation and error warnings through
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


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
        :param h5_path: Path to the HDF5 data file, or to a directory containing
                        the configured HDF5 file name. When omitted, the
                        repository-local default is used only if it exists.
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
        self.h5_path = self.path_manager.resolve_h5_data_path(h5_path)
        
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


class AnisoMaterialFit(
    FitOptimizationMixin,
    FitPersistenceMixin,
    FitVisualizationMixin,
    AnisoModelSolve,
):
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
