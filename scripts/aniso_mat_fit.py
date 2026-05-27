# -*- coding: utf-8 -*-
# import copy
import warnings

import numpy as np
from datetime import datetime
from typing import Dict

from dualmatfit.fitting.core import AnisoMaterialFit
from dualmatfit.utils.logging_config import get_logger

logger = get_logger('aniso_mat_fit')
warnings.filterwarnings('ignore')

# Lambdify Functions
lbdf_mod = "jax"
# lbdf_mod = 'numpy'
# lbdf_mod = 'scipy'

list_slc_colors = ['red', 'blue', 'chocolate', 'tan', 'gray', 'olive', 'lime', 'darkorange', 'teal', 'purple',
                   'salmon', 'teal', 'black', 'deeppink', 'darkviolet', 'royalblue']


########################################################################

def baseline_matfit_run(selection: Dict[str, Dict],
                        mix: int = 1,
                        itype='nh',
                        otype: str = 'L-BFGS-B',
                        dvol: bool = True,
                        kappa: bool = True,
                        glb: bool = True,
                        plot: bool = True,
                        ncontrol: int = 50,
                        **kwargs,
                        ):

    ani_mat_model = AnisoMaterialFit(selection,
                                     itype=itype,
                                     mtype=mix,
                                     dvol=dvol,
                                     kappa=kappa,
                                     iso_split=False,
                                     ncontrol=ncontrol,
                                     hv=False,
                                     opt_type=otype,
                                     opt_glb=glb,
                                     lambdify=lbdf_mod,
                                     )

    # ani_mat_model.load_results(run=True)
    ani_mat_model.exp_test_eval(plot=plot)

    # Options: ['ln', logcosh', 'huber', 'lsq', 'cauchy']
    ftype = 'cauchy_robust'

    # Cauchy Penalization Parameter (lower cm increase D)
    cm = 40.       # **
    # cm = 20       # *
    # cm = 10
    # cm = 5
    # cm = 1
    # cm = 0.1
    # cm = 0.05
    # cm = 0.01

    # Tikhonov regularization scaling parameter
    # alpha = 10.
    # alpha = 1.
    # alpha = 0.1
    # alpha = 0.05
    # alpha = 0.01
    # alpha = 0.005
    # alpha = 0.001
    # alpha = 0.0005
    # alpha = 0.0002       # *
    alpha = 0.0001            # .. nc10
    # alpha = 0.00001

    # Volume regularization parameter (epsilon)
    # epsilon = 0.00001
    # epsilon = 0.0005
    # epsilon = 0.0002       # *
    # epsilon = 0.0001        # ..**
    # epsilon = 0.0005
    epsilon = 0.001
    # epsilon = 0.01
    # epsilon = 0.1
    # epsilon = 1.

    # rsc_type = "inverse"          # "rsc2"
    # rsc_type = "direct"           # "rsc3"
    # rsc_type = "inverse_nrs"
    rsc_type = None

    logger.info(f"M{mix} Ftype: {ftype}, Cauchy Param: {cm}, L2: {alpha}, L2 type: {rsc_type}")

    ani_mat_model.find_baseline_parameters(ftype=ftype,
                                           # miter=100,
                                           miter=200,
                                           giter=30,
                                           # miter=10,
                                           # giter=2,
                                           c=cm,
                                           alpha=alpha,
                                           epsilon=epsilon,
                                           rescale=rsc_type,
                                           dvol=True,
                                           bh_step="random_displacement",
                                           # bh_step="pareto_displacement",
                                           )
    ani_mat_model.save_data()
    ani_mat_model.plot_fit(global_opt=True)

    logger.info(f"Selection Cases: {list(selection.keys())}, ncontrol: {ncontrol}")
    logger.info(f"M{mix} Ftype: {ftype}, Cauchy Param: {cm}, L2: {alpha}, L2 type: {rsc_type}, VReg (epsilon): {epsilon}")

    # Log the formatted timestamp
    now = datetime.now()                # Get the current local timestamp
    logger.info(f"{now.strftime('%A, %d %B %Y, %I:%M:%S %p %Z')}")


def matfit_run(selection: Dict[str, Dict],
               mix: int = 1,
               itype='nh',
               otype: str = 'L-BFGS-B',
               dvol: bool = True,
               kappa: bool = True,
               glb: bool = True,
               plot: bool = True,
               ncontrol: int = 50,
               **kwargs,
               ):

    ani_mat_model = AnisoMaterialFit(selection,
                                     itype=itype,
                                     mtype=mix,
                                     dvol=dvol,
                                     kappa=kappa,
                                     ncontrol=ncontrol,
                                     hv=False,
                                     opt_type=otype,
                                     opt_glb=glb,
                                     lambdify=lbdf_mod,
                                     )

    ani_mat_model.exp_test_eval(plot=plot)
    ani_mat_model.load_results(run=True)
    # sections = ["Ar-A", "Ar-B", "Ar-C"]
    # sections = ["Ab-A", "Ab-B", "Ab-C"]
    # sections = ["Ab-B", "Ab-C"]
    sections = None

    logger.info(f"Selection Cases for local optimization: {list(selection.keys())}, ncontrol: {ncontrol}, sections: {sections}")

    # epsilon = 1.e-4   # *
    epsilon = 1.e-3      # .**
    # epsilon = 1.e-2

    # Tikhonov regularization scaling parameter
    # alpha = 0.
    # alpha = 1.e-9
    # alpha = 0.00001
    # alpha = 0.0001        # **
    alpha = 0.001           # .*
    # alpha = 0.005
    # alpha = 0.01
    # alpha = 0.05
    # alpha = 0.1
    # alpha = 0.25
    # alpha = 0.5
    # alpha = 1.

    # Tikhonov rescaling parameter
    # beta = 2.
    beta = 1.
    # beta = 0.5

    # Cauchy Penalization Parameter
    # cm = 20.
    cm = 40.       # **
    # cm = 60.
    # cm = 100.

    # number of trials in Basin Hopping Method
    # giter = 1
    # giter = 2
    # giter = 5
    # giter = 6
    # giter = 10
    # giter = 15
    giter = 20
    # giter = 25

    # Options: ['ln', logcosh', 'huber', 'lsq', 'cauchy']
    # ftype = 'sum_lsq'
    ftype = 'cauchy_robust'

    # rsc_type = "inverse"          # "rsc2"
    # rsc_type = "direct"           # "rsc3"
    # rsc_type = "inverse_nrs"
    rsc_type = None

    local_opt_path = f"{ftype}_l2"

    if rsc_type is not None:
        local_opt_path = local_opt_path + f"_{rsc_type}"

    if 'cauchy' in ftype:
        if beta == 1.:
            local_opt_path = local_opt_path + f"_alpha_{alpha}_c_{cm}_nc{ncontrol}"
        elif beta > 1.:
            local_opt_path = local_opt_path + f"_alpha_{alpha}_c_{cm}_beta_{beta}_nc{ncontrol}"
    else:
        local_opt_path = None

    logger.info(f"M{mix} Ftype: {ftype}, Cauchy Param: {cm}, L2: {alpha}, L2 type: {rsc_type}, VReg (epsilon): {epsilon}")

    ani_mat_model.find_optimal_parameters(
        miter=400,
        # miter=200,
        # miter=150,
        # miter=50,
        # miter=20,
        # miter=10,
        giter=giter,
        plot=True,
        ftype=ftype,
        # rho=rho,
        alpha=alpha,
        epsilon=epsilon,
        # beta=beta,
        c=cm,
        dvol=True,
        rescale=rsc_type,
        sections=sections,
        local_path=local_opt_path,
        )

    ani_mat_model.save_data()
    ani_mat_model.plot_fit()

    logger.info(f"Selection Cases: {list(selection.keys())}, ncontrol: {ncontrol}, sections: {sections}")

    logger.info(f"M{mix} Ftype: {ftype}, Cauchy Param: {cm}")
    logger.info(f"Regularization Parameters L2: alpha: {alpha}, beta: {beta}, type: {rsc_type}, VReg (epsilon): {epsilon}")
    logger.info(f"Working path: {local_opt_path}")

    # Log the formatted timestamp
    now = datetime.now()                # Get the current local timestamp
    logger.info(f"{now.strftime('%A, %d %B %Y, %I:%M:%S %p %Z')}")


def main_run():
    """
    Verify material parameter tables using:

    Holzapfel, Gerhard A., et al. "Determination of layer-specific mechanical
    properties of human coronary arteries with nonatherosclerotic intimal
    thickening and related constitutive modeling." American Journal of
    Physiology-Heart and Circulatory Physiology 289.5 (2005): H2048-H2058.
    """

    # Load Excel Data
    # load_excel_data()

    kappa_flg = True
    iso_type = 'nh'

    # VFLG = False
    vflg = True

    # Global Optimization Flag
    glb = True

    # New Rats Selection
    SLC_RATS = {
        'rato_17': dict(Ar=['A', 'B', 'C'], Tr=['A', 'B'], Ab=['A', 'B', 'C']),
        'rato_23': dict(Ar=['A', 'C'], Tr=['A', 'B', 'C'], Ab=['A', 'B', 'C']),    #*
        'rato_wt_184012': dict(Ar=['A', 'B', 'C'], Tr=['A', 'B', 'C'], Ab=['A', 'B', 'C']),
        'rato_wt_184085': {'Ar': ['A', 'B', 'C'], 'Tr': ['A', 'B', 'C'], 'Ab': ['A', 'B', 'C']},
        'rato_wt_183997': dict(Ar=['A', 'B', 'C'], Tr=['A', 'B', 'C'], Ab=['A', 'B', 'C']),
    }

    #                  0,       1,          2,     3,         4,       5,        6,
    # OPT_TYPES = ['ipopt', 'SLSQP', 'L-BFGS-B', 'TNC', 'DIFFEVOL', 'SHGO', 'MPowell']
    optimizers = ['ipopt']

    msolu = [3]
    np_control = np.array([15])

    for ncontrol_i in np_control:
        for msol_k in msolu:
            for otype_l in optimizers:
                # Inverse Problem
                baseline_matfit_run(SLC_RATS, mix=msol_k, kappa=kappa_flg, itype=iso_type, otype=otype_l, dvol=vflg,
                glb=glb, ncontrol=ncontrol_i, plot=False)

                matfit_run(SLC_RATS, mix=msol_k, kappa=kappa_flg, itype=iso_type, otype=otype_l, dvol=vflg, glb=glb,
                           ncontrol=ncontrol_i, plot=False)


if __name__ == "__main__":
    main_run()