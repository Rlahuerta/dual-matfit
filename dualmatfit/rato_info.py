# -*- coding: utf-8 -*-
"""
Rato (rat) experimental data information and configuration.

This module provides configuration and data loading utilities for
rat arterial tissue experimental data.
"""
import matplotlib
import pandas as pd
from pathlib import Path
from typing import Dict

from dualmatfit.logging_config import get_logger
logger = get_logger('info')

__all__ = [
    'excel_data',
]

matplotlib.use('Agg')


def load_excel_data() -> None:
    """
    Load Excel Data and Save it into H5 file format.
    
    Reads data from 'final_raw_data.xlsx' and saves to HDF5 format.
    """

    script_dir = Path(__file__).parent.parent.resolve()
    data_path = (script_dir / "instron_data").resolve()

    xfile = 'final_raw_data.xlsx'
    xfile_path = (data_path / xfile).resolve()

    xl_data = pd.ExcelFile(str(xfile_path))

    h5_xfile = (data_path / 'final_data.h5').resolve()
    
    with pd.HDFStore(str(h5_xfile), mode='w') as h5_store:
        for i, st_nm_i in enumerate(xl_data.sheet_names):
            if st_nm_i != "Info":
                logger.info(f'Loading sheet {i+1:02d}: {st_nm_i}')
                pd_xldf_i = xl_data.parse(st_nm_i)

                st_nm_cl_i = st_nm_i.replace(' ', '_')
                st_nm_cl_i = st_nm_cl_i.replace('-', '_')
                h5_store[st_nm_cl_i] = pd_xldf_i

                h5_store.flush()


def excel_data() -> Dict[str, dict]:
    """
    Get experimental data dictionary for all rats.
    
    Returns
    -------
    Dict[str, dict]
        Dictionary mapping rat names (e.g., 'rats-1') to their experimental data.
    """
    list_data = []
    # dist : ccomprimento entre as garras

    ##################################################################
    # RATO - 17
    rato_17 = dict(Ar=dict(A=dict(len=1.5, dist=4.0, tcontrol=[3., 26.5]),
                           B=dict(len=1.2, dist=4.0, tcontrol=[3., 22.]),
                           C=dict(len=1.2, dist=7.0, tcontrol=[2.5, 24.]),
                           dia=3.405833, thick=0.2485),
                   Tr=dict(A=dict(len=1.2, dist=4.0, tcontrol=[3., 15.]),
                           B=dict(len=1.6, dist=3.8, tcontrol=[3., 18.5]),
                           dia=3.202000, thick=0.290667),
                   Ab=dict(A=dict(len=1.7, dist=2.0, tcontrol=[3., 17.]),
                           B=dict(len=1.0, dist=2.0, tcontrol=[4., 13.]),
                           C=dict(len=1.1, dist=2.0, tcontrol=[3., 10.5]),
                           dia=2.973333, thick=0.2600),
                   name='rato-17')

    list_data.append(rato_17)

    ##################################################################
    # RATO - 23
    rato_23 = dict(Ar=dict(A=dict(len=1.2, dist=4.0, tcontrol=[2.5, 23.]),
                           B=dict(len=1.0, dist=5.0, tcontrol=[2.5, 19.5]),
                           C=dict(len=1.0, dist=4.0, tcontrol=[2.5, 19.5]),
                           dia=3.684335, thick=0.192335),
                   Tr=dict(A=dict(len=1.5, dist=4.5, tcontrol=[2.5, 9.]),
                           B=dict(len=1.8, dist=3.6, tcontrol=[2.5, 15.]),
                           C=dict(len=1.7, dist=3.5, tcontrol=[2.5, 15]),
                           dia=3.8695, thick=0.2227),
                   Ab=dict(A=dict(len=1.2, dist=2.0, tcontrol=[3.0, 16.]),
                           B=dict(len=1.2, dist=1.8, tcontrol=[3.0, 18.]),
                           C=dict(len=1.5, dist=2.0, tcontrol=[3.0, 14.]),
                           dia=2.884830, thick=0.212500),
                   name='rato-23')

    list_data.append(rato_23)

    ##################################################################
    # RATO: WT 184085
    rato_id_nw = dict(
        Ar=dict(
            A=dict(len=1.8, dist=3.0, tcontrol=[3., 20.]),
            B=dict(len=1.5, dist=3.5, tcontrol=[3., 14.]),
            C=dict(len=1.5, dist=3.0, tcontrol=[2., 24.]),
            dia=3.43017, thick=0.3115),
        Tr=dict(
            A=dict(len=2.0, dist=2.5, tcontrol=[2.5, 17.5]),
            B=dict(len=2.0, dist=2.8, tcontrol=[2.5, 17.5]),
            C=dict(len=1.8, dist=2.5, tcontrol=[2.5, 16.]),
            dia=3.13234, thick=0.36133),
        Ab=dict(
            A=dict(len=1.8, dist=2.5, tcontrol=[2.5, 13.5]),
            B=dict(len=1.8, dist=2.0, tcontrol=[2.5, 13.5]),
            C=dict(len=1.8, dist=2.0, tcontrol=[2.5, 8.]),
            dia=2.42884, thick=0.17983),
        name='rato-wt-184085')

    list_data.append(rato_id_nw)

    ##################################################################
    # RATO: WT 184012
    rato_id_nw = dict(
        Ar=dict(
            A=dict(len=1.2, dist=3.0, tcontrol=[5., 25.]),
            B=dict(len=1.0, dist=3.1, tcontrol=[5., 24.]),
            C=dict(len=1.2, dist=3.5, tcontrol=[3., 22.]),
            dia=3.24284, thick=0.34917),
        Tr=dict(
            A=dict(len=2.0, dist=2.8, tcontrol=[2.5, 15.]),
            B=dict(len=1.8, dist=2.5, tcontrol=[2.5, 18.]),
            C=dict(len=1.6, dist=2.8, tcontrol=[3., 17.]),
            dia=3.19517, thick=0.34983),
        Ab=dict(
            A=dict(len=1.5, dist=2.0, tcontrol=[4., 12.]),
            B=dict(len=1.5, dist=2.0, tcontrol=[4., 13.]),
            C=dict(len=1.8, dist=2.0, tcontrol=[3., 10.]),
            dia=2.95950, thick=0.2375),
        name='rato-wt-184012')

    list_data.append(rato_id_nw)

    ##################################################################
    # RATO: WT 183997
    rato_id_nw = dict(
        Ar=dict(
            A=dict(len=1.2, dist=3.0, tcontrol=[2.5, 20.]),
            B=dict(len=1.2, dist=3.5, tcontrol=[2.5, 20.]),
            C=dict(len=1.5, dist=3.5, tcontrol=[2.5, 20.]),
            dia=2.92034, thick=0.25067),
        Tr=dict(
            A=dict(len=1.8, dist=2.8, tcontrol=[2.5, 19.]),
            B=dict(len=1.8, dist=2.5, tcontrol=[2.5, 20.]),
            C=dict(len=1.8, dist=2.5, tcontrol=[2.5, 20.]),
            dia=2.46900, thick=0.26833),
        Ab=dict(
            A=dict(len=2., dist=2.2, tcontrol=[2., 13.]),
            B=dict(len=1.8, dist=2.0, tcontrol=[2., 10.5]),
            C=dict(len=2.0, dist=2.0, tcontrol=[2., 11.]),
            dia=2.20917, thick=0.2125),
        name='rato-wt-183997')

    list_data.append(rato_id_nw)

    ##################################################################
    # Full Info Raw Data
    info_data = dict()

    for i, data_i in enumerate(list_data):
        info_data[data_i['name']] = data_i

    return info_data


# Info about samples type (KO or WT)
# def optmat_data():
#
#     samples_type = dict(rato_1='KO', rato_2='WT', rato_3='KO', rato_4='WT', rato_5='WT', rato_6='KO', rato_7='WT',
#                         rato_8='KO', rato_9='KO', rato_10='WT', rato_11='KO', rato_17='WT', rato_18='WT', rato_20='WT',
#                         rato_21='WT', rato_22='WT', rato_23='WT', rato_24='WT', rato_idoso_3='KO', rato_idoso_4='WT')
#
#     return samples_type


if __name__ == "__main__":
    load_excel_data()
