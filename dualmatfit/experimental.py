# -*- coding: utf-8 -*-
"""
Experimental data handling for material testing.

This module provides classes for processing and managing data from 
Instron extension tests and other experimental setups.
"""
import re
import warnings

import numpy as np
import pandas as pd
# import sympy as sy
# from pathlib import Path
from typing import Union, Tuple, List, Dict, Any, Optional

from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler

from dualmatfit.logging_config import get_logger
logger = get_logger('experimental')

__all__ = [
    'InstronData',
    'MaterialSetup',
]


class InstronData:
    """
    Class to handle and process data from an Instron Extension Test.
    It stores experimental and processed data but does not handle plotting directly.
    """

    def __init__(self,
                 df_data: pd.DataFrame,
                 info_data: dict,
                 ncontrol: int = 10,
                 **kwargs: Any,
                 ):
        """
        Initialize the InstronData class

        Parameters:
        - pd_data (pd.DataFrame):       DataFrame containing the experimental data.
        - info_data (Dict[str, Any]):   Dictionary containing test information.
        - ncontrol (int, optional):     Number of control points for interpolation. Defaults to 10.
        """
        # Preprocessing pd_data
        if df_data is None or df_data.empty:
            raise ValueError("Input DataFrame 'df_data' cannot be None or empty.")

        for clm_i in ['Time', 'Extension', 'Load']:
            chk_i = [clm_i in clm_k for clm_k in df_data.columns]
            if sum(chk_i) == 0:
                raise ValueError(f"Input DataFrame 'df_data' missing column '{clm_i}'.")

        self.pd_exp_data = df_data.copy().dropna()
        self.pd_exp_data.dropna(inplace=True)
        if self.pd_exp_data.empty:
            raise ValueError(
                "DataFrame became empty after dropping NaN values. "
                "Check input data for missing values."
            )

        if info_data is None:
            raise ValueError("Input dictionary 'info_data' cannot be None.")

        self.info_data = info_data
        self.ncontrol = ncontrol
        self.key_clm = self.pd_exp_data.columns

        # Initialize data attributes
        self.np_time: np.ndarray = np.array([])
        self.np_extn: np.ndarray = np.array([])
        self.np_load: np.ndarray = np.array([])
        self.np_tinc: Optional[np.ndarray] = None # Time control points
        self.np_textn: Optional[np.ndarray] = None # Extension at control points
        self.np_tload: Optional[np.ndarray] = None # Load at control points
        self.ref_extn: Optional[float] = None # Reference extension (usually at t=0 or first control point)
        self.ref_load: Optional[float] = None # Reference load
        self.np_tinc_rs: Optional[np.ndarray] = None # Rescaled time control points
        self.np_extn_ref: Optional[np.ndarray] = None # Raw extension relative to reference
        self.np_load_ref: Optional[np.ndarray] = None # Raw load relative to reference
        self.np_textn_ref: Optional[np.ndarray] = None # Control extension relative to reference (excluding ref point)
        self.np_tstretch_ref: Optional[np.ndarray] = None # Control stretch relative to reference
        self.np_tload_ref: Optional[np.ndarray] = None # Control load (Force) relative to reference
        self.np_tpk1_ref: Optional[np.ndarray] = None # Control PK1 stress relative to reference
        self.np_tstretch: Optional[np.ndarray] = None # Raw stretch relative to reference
        self.disp_mean: Optional[float] = None # Mean of raw extension

        # High-resolution interpolated data for plotting model results against
        self.high_res_tinc: Optional[np.ndarray] = None
        self.high_res_extn: Optional[np.ndarray] = None
        self.high_res_load: Optional[np.ndarray] = None
        self.high_res_stretch: Optional[np.ndarray] = None
        self.high_res_force: Optional[np.ndarray] = None
        self.high_res_pk1: Optional[np.ndarray] = None

        # Get the time, extension, and load data from the DataFrame
        self._extract_data()
        self.disp_mean = self.np_extn.mean() if self.np_extn.size > 0 else 0.0

        # Interpolator (kept in case they are needed for other non-plotting tasks)
        self.rbfi_textn = None
        self.rbfi_tload = None

        if self.np_time.size > 1:
             try:
                 self.rbfi_textn = interpolate.Rbf(self.np_time, self.np_extn, function='thin_plate')
                 self.rbfi_tload = interpolate.Rbf(self.np_time, self.np_load, function='thin_plate')

             except (ValueError, np.linalg.LinAlgError) as e:
                 # ValueError: Invalid input data for interpolation
                 # LinAlgError: Singular matrix during RBF computation
                 logger.info(f"Warning: Rbf interpolation function creation failed: {e}")

        self.time_control = info_data.get('tcontrol', [self.np_time.min(), self.np_time.max()] if self.np_time.size > 0 else [0, 1])
        self.rfct = 2.                                      # Ring Factor (um anel pode ser dividido em duas fitas)
        self.lx_r = self.info_data['dp'] / self.rfct        # cada fita = 1/2 do perimeter
        self.ds = self.info_data.get('ds', 1.0)             # Cross-sectional area

        if abs(self.lx_r) < 1e-9:
            logger.info(f"Warning: Reference length lx_r is very small or zero ({self.lx_r}). Check 'dp' in info_data.")

        if abs(self.ds) < 1e-9:
            logger.info(f"Warning: Cross-sectional area ds is very small or zero ({self.ds}). Check 'ds' in info_data.")

        self.scaler = MinMaxScaler(feature_range=(0., 1.))

        # Initialize control points and compute derived properties
        self._initialize_control_points()

    def _initialize_control_points(self):
        """
        Initialize control points using interpolation and compute reference values.
        Also computes high-resolution interpolated data.
        """
        if self.np_time.size < 2 or self.ncontrol <= 0 or self.rbfi_textn is None or self.rbfi_tload is None:
            warnings.warn(
                f"Insufficient data points ({self.np_time.size}). Using raw data.",
                UserWarning
            )
            warnings.warn(
                f"ncontrol ({self.ncontrol}) <= 0, or interpolators not created. Using raw data.",
                UserWarning
            )
            # Fallback: Use raw data points as 'control' points
            self.np_tinc = self.np_time
            self.np_textn = self.np_extn
            self.np_tload = self.np_load
            self.ref_extn = self.np_extn[0] if self.np_extn.size > 0 else 0.0
            self.ref_load = self.np_load[0] if self.np_load.size > 0 else 0.0

            if self.np_time.size > 1:
                 self.np_tinc_rs = self.scaler.fit_transform(self.np_tinc.reshape(-1, 1)).flatten()

            # Set high-res data to raw data in fallback case
            self.high_res_tinc = self.np_time
            self.high_res_extn = self.np_extn
            self.high_res_load = self.np_load

        else:
            # Proceed with interpolation
            num_high_res_points = max(200, 5 * self.ncontrol) # Ensure enough points for smooth plots
            self.high_res_tinc = np.linspace(self.time_control[0], self.time_control[1], num=num_high_res_points)
            self.np_tinc = np.linspace(self.time_control[0], self.time_control[1], num=self.ncontrol)

            try:
                self.np_textn = self.rbfi_textn(self.np_tinc)
                self.np_tload = self.rbfi_tload(self.np_tinc)

                self.high_res_extn = self.rbfi_textn(self.high_res_tinc)
                self.high_res_load = self.rbfi_tload(self.high_res_tinc)

                # Use the first control point as reference
                self.ref_extn = self.np_textn[0]
                self.ref_load = self.np_tload[0]

                self.np_tinc_rs = self.scaler.fit_transform(self.np_tinc.reshape(-1, 1)).flatten()

            except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                # ValueError: Invalid data for interpolation or scaling
                # IndexError: Empty arrays when accessing [0]
                # LinAlgError: Singular matrix during interpolation
                logger.info(f"Error during interpolation for control/high-res points: {e}. Falling back to raw data.")
                # Fallback logic copied from above
                self.np_tinc = self.np_time
                self.np_textn = self.np_extn
                self.np_tload = self.np_load
                self.ref_extn = self.np_extn[0] if self.np_extn.size > 0 else 0.0
                self.ref_load = self.np_load[0] if self.np_load.size > 0 else 0.0

                if self.np_time.size > 1:
                    self.np_tinc_rs = self.scaler.fit_transform(self.np_tinc.reshape(-1, 1)).flatten()

                self.high_res_tinc = self.np_time
                self.high_res_extn = self.np_extn
                self.high_res_load = self.np_load

        # Compute derived properties based on available data (raw or interpolated)
        self._compute_derived_properties()

    def _compute_derived_properties(self):
        """Computes stretch, force, PK1 etc. based on available raw and control point data."""
        if self.ref_extn is None or self.ref_load is None:
             logger.warning(" Reference extension/load not set. Cannot compute derived properties.")
             return

        # --- Raw Data Derived Properties ---
        if self.np_extn.size > 0:
            self.np_extn_ref = self.np_extn - self.ref_extn
            self.np_tstretch = self._compute_stretch(self.np_extn, self.ref_extn)

        if self.np_load.size > 0:
            self.np_load_ref = self.np_load - self.ref_load # Raw load relative to reference

        # --- Control Point Derived Properties (relative to reference) ---
        if self.np_textn is not None and self.np_textn.size > 1:
            # Exclude the first point which is the reference
            self.np_textn_ref = self.np_textn[1:] - self.ref_extn
            self.np_tstretch_ref = self._compute_stretch(self.np_textn[1:], self.ref_extn)

        if self.np_tload is not None and self.np_tload.size > 1:
            # Exclude the first point which is the reference
            self.np_tload_ref = self._compute_force(self.np_tload[1:], self.ref_load)
            self.np_tpk1_ref = self._compute_pk1(self.np_tload[1:], self.ref_load)

        # --- High-Resolution Derived Properties ---
        if self.high_res_extn is not None and self.high_res_extn.size > 0:
             self.high_res_stretch = self._compute_stretch(self.high_res_extn, self.ref_extn)

        if self.high_res_load is not None and self.high_res_load.size > 0:
             self.high_res_force = self.rfct * self._compute_force(self.high_res_load, self.ref_load)
             self.high_res_pk1 = self.rfct * self._compute_pk1(self.high_res_load, self.ref_load)

    def _extract_sample_id(self) -> str:
        """
        Extract sample ID from info_data or generate a default.
        """
        # Prioritize info_data if available
        if 'sample_id' in self.info_data:
            return str(self.info_data['sample_id'])

        if 'rato' in self.info_data and 'section' in self.info_data and 'position' in self.info_data:
             return f"{self.info_data['rato']}-{self.info_data['section']}-{self.info_data['position']}"

        # Fallback to column name (less reliable)
        if self.key_clm is not None:
            if len(self.key_clm) > 0:
                first_col = str(self.key_clm[0])
                # Simple extraction assuming format like 'RateXX-YY-Z...'
                parts = first_col.split('-')
                if len(parts) >= 3:
                    return f"{parts[0]}-{parts[1]}-{parts[2]}"

                pattern = r'\((.*)\)'
                match = re.search(pattern, first_col)
                return match.group(1)

        return 'UnknownSample' # Default if no info available

    def _extract_data(self):
        """
        Extract time, extension, and load data from the DataFrame.
        Looks for columns containing 'Time', 'Extension', and 'Load' (case-insensitive).
        """
        if self.pd_exp_data.empty:
            logger.warning(" Cannot extract data from empty DataFrame.")
            return

        try:
            # Find columns using case-insensitive regex
            time_col_name = self.pd_exp_data.filter(regex='(?i)Time').columns
            extn_col_name = self.pd_exp_data.filter(regex='(?i)Extension').columns
            load_col_name = self.pd_exp_data.filter(regex='(?i)Load').columns

            if not time_col_name.empty:
                self.np_time = np.squeeze(self.pd_exp_data[time_col_name[0]].values)
            else:
                logger.warning(" 'Time' column not found.")

            if not extn_col_name.empty:
                self.np_extn = np.squeeze(self.pd_exp_data[extn_col_name[0]].values)
            else:
                logger.warning(" 'Extension' column not found.")

            if not load_col_name.empty:
                self.np_load = np.squeeze(self.pd_exp_data[load_col_name[0]].values)
            else:
                logger.warning(" 'Load' column not found.")

            # Ensure they are 1D arrays
            if self.np_time.ndim > 1: self.np_time = np.squeeze(self.np_time)
            if self.np_extn.ndim > 1: self.np_extn = np.squeeze(self.np_extn)
            if self.np_load.ndim > 1: self.np_load = np.squeeze(self.np_load)

        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"An error occurred during data extraction: {e}") from e

    def _compute_stretch(self, extension: np.ndarray, reference: Union[np.ndarray, float]) -> np.ndarray:
        """
        Compute stretch relative to a reference extension.

        Parameters:
        - extension (np.ndarray): Extension data.
        - reference (float): Reference extension.

        Returns:
        - np.ndarray: Computed stretch values.
        """
        # Avoid division by zero or small numbers if reference length is zero or close to it
        if abs(self.lx_r) < 1e-9:
             logger.info(f"Warning: Reference length lx_r is close to zero ({self.lx_r}). Stretch calculation may be inaccurate.")
             return 1.0 + (extension - reference) # Or handle as error if appropriate

        return 1. + (extension - reference) / self.lx_r

    def _compute_force(self, force: np.ndarray, reference: Union[np.ndarray, float]) -> np.ndarray:
        """
        Compute force relative to a reference force (tape case).

        Parameters:
        - force (np.ndarray): Force data.
        - reference (float): Reference force.

        Returns:
        - np.ndarray: Computed force values.
        """
        return (force - reference) / self.rfct

    def _compute_pk1(self, force: np.ndarray, reference: Union[np.ndarray, float]) -> np.ndarray:
        """
        Compute PK1 stress relative to a reference force (tape case).

        Parameters:
        - force (np.ndarray): Force data.
        - reference (float): Reference force.

        Returns:
        - np.ndarray: Computed PK1 stress values.
        """

        # Requires area, which is in info_data
        if abs(self.ds) < 1e-9:
            logger.info(f"Warning: Cross-sectional area ({self.ds}) or ring factor ({self.rfct}) is close to zero. PK1 calculation may be inaccurate.")
            return np.zeros_like(force) # Return zero stress if area is zero

        # Use the force calculation
        force_per_strip = self._compute_force(force, reference)
        # PK1 = Force / Reference Area
        return force_per_strip / self.ds

    # --- Public Accessors ---
    def get_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns raw time, extension, and load data."""
        return self.np_time, self.np_extn, self.np_load

    def get_control_points(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns interpolated time, extension, and load at control points."""
        return self.np_tinc, self.np_textn, self.np_tload

    def get_reference_values(self) -> Tuple[Optional[float], Optional[float]]:
        """Returns the reference extension and load."""
        return self.ref_extn, self.ref_load

    def get_control_data_relative(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns control point stretch, force, and PK1 stress relative to reference."""
        return self.np_tstretch_ref, self.np_tload_ref, self.np_tpk1_ref

    def get_high_res_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns high-resolution interpolated time, extension, and load."""
        return self.high_res_tinc, self.high_res_extn, self.high_res_load

    def get_high_res_data_relative(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns high-resolution stretch, force, and PK1 stress relative to reference."""
        # Note: high_res_force and high_res_pk1 already account for rfct in _compute_derived_properties
        return self.high_res_stretch, self.high_res_force, self.high_res_pk1

    def get_raw_data_relative(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns raw extension and load relative to their reference values."""
        return self.np_extn_ref, self.np_load_ref

    def get_raw_stretch(self) -> Optional[np.ndarray]:
        """Returns raw stretch computed relative to reference extension."""
        return self.np_tstretch

    def get_info(self) -> Dict[str, Any]:
        """Returns the information dictionary."""
        return self.info_data

    def get_sample_id(self) -> str:
        """Returns the sample ID."""
        return self._extract_sample_id()

    def get_plot_limit_index(self) -> int:
        """
        Get the index limit for plotting raw data, based on the maximum stretch
        reached by either the raw data or the control points.

        Returns:
        - int: Index limit for plotting raw data.
        """
        max_stretch_to_plot = None
        if self.np_tstretch_ref is not None and self.np_tstretch_ref.size > 0:
            max_stretch_to_plot = self.np_tstretch_ref.max()
        elif self.np_tstretch is not None and self.np_tstretch.size > 0:
            # If no control points, use max raw stretch
            max_stretch_to_plot = self.np_tstretch.max()

        if max_stretch_to_plot is None or self.np_tstretch is None or self.np_tstretch.size == 0:
            return len(self.np_extn) if self.np_extn is not None else 0

        # Find index in the *raw* data where stretch reaches or exceeds the max_stretch_to_plot
        idx = np.searchsorted(self.np_tstretch, max_stretch_to_plot, side='right')

        buffer_points = 10
        limit = min(idx + buffer_points, len(self.np_tstretch))

        # Ensure at least some data is included
        if limit == 0 and len(self.np_tstretch) > 0:
            limit = 1

        return limit

    # --- Function/Methods to be Removed ---
    def _lsq_disp_fun(self, tinc_i: float) -> np.ndarray:
        """
        Least squares function for disp_mean (Keep this if used elsewhere)

        :param tinc_i:  Time increment

        :return:    Least squares function value
        """

        return self.disp_mean - self.rbfi_textn(tinc_i)

    def _get_index_limit(self) -> int:
        """
        Get the index limit for plotting raw data, based on the maximum stretch
        reached by either the raw data or the control points.

        Returns:
        - int: Index limit for plotting raw data.
        """

        # Use the max stretch from raw data or control points
        max_stretch_to_plot = self.np_tstretch_ref.max() if self.np_tstretch_ref is not None and self.np_tstretch_ref.size > 0 else (self.np_tstretch.max() if self.np_tstretch is not None and self.np_tstretch.size > 0 else None)

        if max_stretch_to_plot is None:
             return len(self.np_extn) if self.np_extn is not None else 0 # No data or control points

        # Find index in the *raw* data where stretch reaches or exceeds the max_stretch_to_plot
        # Use searchsorted for efficiency
        if self.np_tstretch is None or self.np_tstretch.size == 0:
            return 0 # No raw stretch data

        idx = np.searchsorted(self.np_tstretch, max_stretch_to_plot, side='right')

        # Add a small buffer (e.g., 5-10 points) to show slightly beyond the last control point
        # or max raw stretch if they are close
        buffer_points = 10
        limit = min(idx + buffer_points, len(self.np_tstretch))

        # Ensure at least some data is included if possible
        if limit == 0 and len(self.np_tstretch) > 0:
             limit = 1 # Show at least the first point if data exists

        return limit

    def get_compliance(self) -> np.ndarray:
        """
        Compute compliance from reference adjusted extension and load.

        Returns:
        - np.ndarray: Compliance values.
        """
        # Keep this data processing method
        if self.ncontrol == 0:
            raise ValueError("Control points not initialized. Ensure ncontrol > 0.")
        # Add a check to avoid division by zero load if needed
        return self.np_textn_ref / self.np_tload_ref

    def get_xstretch(self, xdisp: np.ndarray) -> np.ndarray:
        """
        Compute stretch in x-direction from displacement.

        Parameters:
        - xdisp (np.ndarray): Displacement in x-direction.

        Returns:
        - np.ndarray: Stretch values in x-direction.
        """
        return 1. + (xdisp / self.lx_r)


class MaterialSetup:
    """
        A class for setting up material parameters including isotropic and anisotropic properties.
    """

    def __init__(self, itype: str, bulk: float, dvol: bool, kappa: bool):
        """
        Initializes a MaterialSetup object with given material characteristics.

        Parameters:
            itype (str):        Type of the isotropic material model ('nh' or 'fung').
            bulk (float):       Bulk modulus of the material.
            dvol (bool):        Flag indicating if volumetric dispersion is considered.
            kappa (bool):       Flag indicating if material dispersion is considered.
        """

        self.itype = itype
        self.bulk = bulk
        self.dvol = dvol
        self.kappa = kappa
        self.opt_keys_iso, self.ini_dsvars_iso, self.bounds_dsvars_iso, self.delta_dsvars_iso = self._ini_iso_params()
        self.opt_keys_ani, self.ini_dsvars_ani, self.bounds_dsvars_ani, self.delta_dsvars_ani = self._ini_ani_params()
        self.df_dsvars = self._combine_params()

        self.df_aorta_seq = None
        self.df_mat_corr = None
        self.df_ese_corr = None

    def _ini_iso_params(self) -> Tuple:
        """Initializes isotropic material properties based on the material type."""

        if self.itype == 'nh':
            return ['mu'], [0.007], [(0.0001, 1.)], [0.05]
        elif self.itype == 'fung':
            return ['a_f', 'b_f'], [5., 0.5], [(0.001, 40.), (0.001, 50.)], [0.1, 0.1]
        else:
            raise NotImplementedError(f" Hyperelastic Model not implemented yet: {self.itype}")

    @staticmethod
    def _ini_ani_params() -> Tuple[List[str], List[float | Any], List[tuple[float, float] |
                                                                      Tuple[Any, Any]], List[float]]:
        """Initializes an anisotropic material properties."""

        # Data from "On the Compressibility of Arterial Tissue"
        return (['k_1', 'k_2', 'alpha'],
                [0.01551, 0.01, np.deg2rad(60.)],
               [(0.01, 100.),
                (0.01, 100.),
                (np.deg2rad(0.), np.deg2rad(89.))],
                [0.1, 0.1, 0.2])

    @staticmethod
    def _generate_aorta_sequence() -> Dict[str, int]:
        aorta_seq = {}
        p = 0
        for key_p in ['Ar', 'Tr', 'Ab']:
            for sec_p in ['A', 'B', 'C']:
                key_sec_p = key_p + '-' + sec_p
                aorta_seq[key_sec_p] = p
                p += 1

        return aorta_seq

    def _populate_aorta_info(self):

        aorta_sequence = self._generate_aorta_sequence()

        # Initialize Data Arrays
        np_aorta_seq = np.array(list(aorta_sequence.values()), dtype=int)
        np_ini_data = np.zeros(len(aorta_sequence), dtype=float)
        np_ini_mlx = np.ones(len(aorta_sequence), dtype=float)
        method_info = ['' for _ in range(np_aorta_seq.shape[0])]

        # Populate the dictionary with aorta information
        dict_aorta_info = {'idx': np_aorta_seq}

        # Isotropic Material Parameters
        if self.itype == 'nh':
            dict_aorta_info['mu'] = np_ini_data.copy()
        elif self.itype == 'fung':
            dict_aorta_info.update({key: np_ini_data.copy() for key in ['a_f', 'b_f']})

        if self.dvol:
            # bulk value
            dict_aorta_info['D'] = np_ini_data.copy()

        # Anisotropic Material Parameters
        dict_aorta_info.update({key: np_ini_data.copy() for key in ['k_1', 'k_2', 'a1', 'alpha']})

        if self.kappa:
            dict_aorta_info['kappa'] = np_ini_data.copy()

        # Strain Energy Info
        dict_aorta_info.update({key: np_ini_data.copy() for key in ['iso', 'vol', 'ani', 'sum']})

        dict_aorta_info['method'] = method_info
        dict_aorta_info['lsq'] = np_ini_data.copy()
        dict_aorta_info['mlx'] = np_ini_mlx

        df_aorta_seq = pd.DataFrame.from_dict(dict_aorta_info, orient='columns')
        df_aorta_seq.index = list(aorta_sequence.keys())

        self.df_aorta_seq = df_aorta_seq

    def _combine_params(self):
        """
        Combines isotropic and anisotropic parameters into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing initial values, bounds, and limits for each design variable.
        """

        opt_keys = self.opt_keys_iso + (['D'] if self.dvol else []) + self.opt_keys_ani
        list_ini_dsvars = self.ini_dsvars_iso + ([self.bulk] if self.dvol else []) + self.ini_dsvars_ani
        list_bnds_dsvars = self.bounds_dsvars_iso + ([(0.0001, 5.)] if self.dvol else []) + self.bounds_dsvars_ani
        list_lim_dsvars = self.delta_dsvars_iso + ([0.1] if self.dvol else []) + self.delta_dsvars_ani

        if self.kappa:
            opt_keys += ['kappa']
            list_ini_dsvars += [0.7 * (1. / 3.)]
            list_bnds_dsvars += [(0., 0.99 * 1. / 3.)]
            list_lim_dsvars += [0.25]

        data = {
            "ini": list_ini_dsvars,
            "values": list_ini_dsvars,
            "lower": [lb for (lb, _) in list_bnds_dsvars],
            "upper": [ub for (_, ub) in list_bnds_dsvars],
            "limit": list_lim_dsvars,
            "variable": [True] * len(list_ini_dsvars),
        }

        return pd.DataFrame(data, index=opt_keys)

    def _prepare_mat_correlation(self):
        opt_keys = self.df_dsvars.index.tolist()
        np_ini_data = np.zeros(len(opt_keys), dtype=float)
        self.df_mat_corr = pd.DataFrame(dict.fromkeys(opt_keys, np_ini_data), index=opt_keys)

    def _prepare_ese_correlation(self):
        list_key_labels = self.df_aorta_seq.index.tolist()
        opt_keys = self.df_dsvars.index.tolist()
        np_ini_data = np.zeros(len(opt_keys), dtype=float)
        self.df_ese_corr = pd.DataFrame(dict.fromkeys(list_key_labels, np_ini_data), index=opt_keys)

    def __call__(self):
        self._populate_aorta_info()
        self._prepare_mat_correlation()
        self._prepare_ese_correlation()
        return self.df_dsvars, self.df_aorta_seq, self.df_mat_corr, self.df_ese_corr
