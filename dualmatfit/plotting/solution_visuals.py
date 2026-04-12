# -*- coding: utf-8 -*-
"""
Solution visualization for 2D material model results.

This module provides the PlotSolution2D class for visualizing strain energy,
stress, and force results from material model solutions.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult

__all__ = [
    'PlotSolution2D',
]


class PlotSolution2D:
    def __init__(self):
        """
        Initialize the PlotSolution2D class.

        Parameters:
        - title (str): The title of the plot.
        - ltype (dict): Dictionary mapping keys to line styles.
        - post_equations (dict, optional): Dictionary of equations to display.
        """

        # Labels for the plot
        self.ltx_energy = r"$\psi$"
        self.ltx_stress_x = r"$\sigma_x$"
        self.ltx_stress_y = r"$\sigma_y$"
        self.ltx_stress_z = r"$\sigma_z$"

        self.colors = {'iso': 'blue', 'vol': 'orange', 'ani': 'green', 'total': 'red'}

        # Initialize lists for legend handles and labels
        self.all_handles = []
        self.all_labels = []

        # Equation text increment position
        self.eq_inc = 4.5

    def _create_2d_plot(self, fontsize: int = 14):

        fig, ax = plt.subplots(2, 2, figsize=(24, 18), dpi=800)
        fl_ax = ax.ravel()

        ax[0, 0].set_xlabel(r'Stretch Ratio, $l_x$', fontsize=fontsize)
        ax[0, 0].set_ylabel(r'Strain Energy, $\psi (F)$', fontsize=fontsize)

        ax[0, 1].set_xlabel(r'Stretch Ratio, $l_x$', fontsize=fontsize)
        ax[0, 1].set_ylabel(f'Engineering Stress, {self.ltx_stress_x}', fontsize=fontsize)

        ax[1, 0].invert_xaxis()
        ax[1, 0].set_xlabel(r'Stretch Ratio, $l_y$', fontsize=fontsize)
        ax[1, 0].set_ylabel(f'Engineering Stress, {self.ltx_stress_y}', fontsize=fontsize)

        ax[1, 1].invert_xaxis()
        ax[1, 1].set_xlabel(r'Stretch Ratio, $l_z$', fontsize=fontsize)
        ax[1, 1].set_ylabel(f'Engineering Stress, {self.ltx_stress_z}', fontsize=fontsize)

        for ax_i in fl_ax:
            ax_i.axvline(x=1.0, color='k', linestyle=":")
            ax_i.axhline(y=0.0, color='r', linestyle=":")
            ax_i.grid(which='minor', alpha=0.2)
            ax_i.grid(which='major', alpha=0.5)

        dict_ax = {'ese': ax[0, 0], 'x': ax[0, 1], 'y': ax[1, 0], 'z': ax[1, 1]}

        return fig, ax, dict_ax

    @staticmethod
    def _create_force_plot(fontsize: int = 14):

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=400)

        ax.set_xlabel(r'Stretch Ratio, $l_x$', fontsize=fontsize)
        ax.set_ylabel(r'Force in $x$-axis', fontsize=fontsize)

        ax.axvline(x=1.0, color='k', linestyle=":")
        ax.axhline(y=0.0, color='r', linestyle=":")
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        return fig, ax

    def components_plot(self,
                        title: str,
                        results: dict,
                        ltype: dict,
                        fname: str,
                        post_equations: dict = None,
                        ):

        if post_equations is None:
            post_equations = {}

        fig, ax, dict_ax = self._create_2d_plot(fontsize=16)
        fig.suptitle(title, fontsize=20)

        eq_inc = 4.5
        all_handles, all_labels = [], []

        for i, (key_i, post_solu_i) in enumerate(results.items()):
            lx_i = post_solu_i.stretch[:, 0]
            ly_i = post_solu_i.stretch[:, 1]
            lz_i = post_solu_i.stretch[:, 2]
            ltype_i = ltype[key_i]

            for stype_k, color_k in zip(['iso', 'vol', 'ani'], ['blue', 'orange', 'green']):
                # Plot the strain energy contributions for iso, vol, ani in subplot 1
                kwargs_k = {'linestyle': ltype_i, 'color': color_k, 'alpha': 0.7, 'label': f'{key_i} ({stype_k})'}

                line_k, = dict_ax["ese"].plot(lx_i, post_solu_i.ese[stype_k], **kwargs_k)
                all_handles.append(line_k)
                all_labels.append(line_k.get_label())

                # Plot the engineering stress contributions in subplot 2 (sigma_x) and 3 (sigma_y)
                dict_ax["x"].plot(lx_i, post_solu_i.stress[stype_k][:, 0], **kwargs_k)
                dict_ax["y"].plot(ly_i, post_solu_i.stress[stype_k][:, 1], **kwargs_k)
                dict_ax["z"].plot(lz_i, post_solu_i.stress[stype_k][:, 2], **kwargs_k)

            # Add equations if provided
            if post_equations.get(key_i) is not None:
                ax[0].text(1.1, eq_inc, f'{self.ltx_energy} - {key_i} = ${post_equations.get(key_i)}$', fontsize=13,
                           ha='left')
                eq_inc -= 1.

        # Create a single legend outside the plots at the bottom
        fig.legend(handles=all_handles, labels=all_labels, loc="lower center", ncol=6, fontsize=12,
                   bbox_to_anchor=(0.5, 0.01))

        # Save figure
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    def force_plot(self,
                    title: str,
                    results: OptimizeResult,
                    fname: str,
                    ):

        fig, ax = self._create_force_plot(fontsize=16)
        fig.suptitle(title, fontsize=20)

        np_lx = results.stretch[:, 0]
        np_fx_r = results.xforce
        np_fint = results.fint[:, 0]

        ax.plot(np_lx, np_fint, color="red", alpha=0.7, label='model')
        ax.plot(np_lx, np_fx_r, "o", color="black", alpha=0.7, label='experiment')

        # Create a single legend outside the plots at the bottom
        fig.legend(loc="lower center", fontsize=12, bbox_to_anchor=(0.5, 0.01))

        # Save figure
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    def full_plot(self,
                  title: str,
                  results: dict,
                  ltype: dict,
                  fname: str,
                  post_equations: dict = None,
                  ):

        if post_equations is None:
            post_equations = {}

        fig, ax, dict_ax = self._create_2d_plot(fontsize=16)
        fig.suptitle(title, fontsize=20)

        eq_inc = 4.5
        all_handles, all_labels = [], []

        for i, (key_i, post_solu_i) in enumerate(results.items()):
            lx_i = post_solu_i.stretch[:, 0]
            ly_i = post_solu_i.stretch[:, 1]
            lz_i = post_solu_i.stretch[:, 2]
            ltype_i = ltype[key_i]

            for stype_k in ['iso', 'vol', 'ani']:

                if stype_k == 'ani':
                    kwargs_k = {'linestyle': ltype_i, 'color': "green", 'alpha': 0.7, 'label': f'{key_i} ({stype_k})'}

                    line_k, = dict_ax["ese"].plot(lx_i, post_solu_i.ese[stype_k], **kwargs_k)
                    dict_ax["x"].plot(lx_i, post_solu_i.stress[stype_k][:, 0], **kwargs_k)
                    dict_ax["y"].plot(ly_i, post_solu_i.stress[stype_k][:, 1], **kwargs_k)
                    dict_ax["z"].plot(ly_i, post_solu_i.stress[stype_k][:, 2], **kwargs_k)

                    all_handles.append(line_k)
                    all_labels.append(line_k.get_label())

            line_i, = dict_ax["ese"].plot(lx_i, post_solu_i.ese['total'], linestyle=ltype_i, color="red", label=f'{key_i}')

            all_handles.append(line_i)
            all_labels.append(line_i.get_label())

            # Plot the engineering stress contributions in subplot 2 (sigma_x) and 3 (sigma_y)
            kwargs_i = {'linestyle': ltype_i, 'color': "red", 'alpha': 0.7, 'label': f'{key_i}'}

            dict_ax["x"].plot(lx_i, post_solu_i.stress['full'][:, 0], **kwargs_i)
            dict_ax["y"].plot(ly_i, post_solu_i.stress['full'][:, 1], **kwargs_i)
            dict_ax["z"].plot(lz_i, post_solu_i.stress['full'][:, 2], **kwargs_i)

            # Add equations if provided
            if post_equations.get(key_i) is not None:
                ax[0].text(1.1, eq_inc, f'{self.ltx_energy} - {key_i} = ${post_equations.get(key_i)}$', fontsize=13,
                           ha='left')
                eq_inc -= 1.

        # Create a single legend outside the plots at the bottom
        fig.legend(handles=all_handles, labels=all_labels, loc="lower center", ncol=6, fontsize=12,
                   bbox_to_anchor=(0.5, 0.01))

        # Save figure
        fig.savefig(fname, dpi=300)
        plt.close(fig)