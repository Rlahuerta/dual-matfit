# -*- coding: utf-8 -*-
"""
Basin hopping optimization with IPOPT.

This module provides basin hopping global optimization using IPOPT
as the local minimizer, with custom step-taking strategies for
material parameter optimization.
"""
import numpy as np
import math

from scipy.optimize import OptimizeResult, rosen, rosen_der
from scipy._lib._util import check_random_state
from dualmatfit.optimization.ipopt import IpyoptMinimizer

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('optimization')

__all__ = [
    'Storage',
    'BasinHoppingRunner',
    'AdaptiveStepSize',
    'StepDisplacement',
    'Metropolis',
    'ipopt_basinhopping',
]



class Storage:
    """
    Class used to store the lowest energy structure found during basin hopping.

    Attributes
    ----------
    minres : OptimizeResult
        The optimization result object corresponding to the lowest function value found so far.
    """

    def __init__(self, minres):
        self._add(minres)

    def _add(self, minres):
        self.minres = minres
        self.minres.x = np.copy(minres.x)

    def update(self, minres):
        """
        Update the stored minimum if the new result is better.

        Parameters
        ----------
        minres : OptimizeResult
            The result from a local minimization.

        Returns
        -------
        bool
            True if the stored minimum was updated, False otherwise.
        """
        if minres.fun < self.minres.fun:
            self._add(minres)
            return True
        else:
            return False

    def get_lowest(self):
        """
        Return the OptimizeResult object corresponding to the lowest function value.

        Returns
        -------
        OptimizeResult
            The optimization result object for the lowest function value found.
        """
        return self.minres


class BasinHoppingRunner:
    """This class implements the core of the basinhopping algorithm.

    x0 : ndarray
        The starting coordinates.
    minimizer : callable
        The local minimizer, with signature ``result = minimizer(x)``.
        The return value is an `optimize.OptimizeResult` object.
    step_taking : callable
        This function displaces the coordinates randomly. Signature should
        be ``x_new = step_taking(x)``. Note that `x` may be modified in-place.
    accept_tests : list of callables
        Each test is passed the kwargs `f_new`, `x_new`, `f_old` and
        `x_old`. These tests will be used to judge whether to accept
        the step. The acceptable return values are True, False, or ``"force
        accept"``. If any of the tests return False then the step is rejected.
        If ``"force accept"``, then this will override any other tests in
        order to accept the step. This can be used, for example, to forcefully
        escape from a local minimum that ``basinhopping`` is trapped in.
    disp : bool, optional
        Display status messages.

    """
    def __init__(self,
                 x0: np.ndarray,
                 minimizer: IpyoptMinimizer,
                 step_taking,
                 accept_tests,
                 disp=False,
                 ):

        self.x = np.copy(x0)
        self.minimizer = minimizer
        self.step_taking = step_taking
        self.accept_tests = accept_tests
        self.disp = disp

        self.nstep = 0
        self.xtrial = None
        self.energy_trial = None
        self.accept = None

        # initialize return object
        self.res = OptimizeResult()
        self.res.minimization_failures = 0

        self.consecutive_failures = 0           # Failure counter
        self.failure_threshold_restart = 10     # Restart threshold

        # Initial minimization
        min_res = minimizer(self.x)
        if not min_res.success:
            self.res.minimization_failures += 1
            logger.warning("Basin hopping: local minimization failure")

        self.x = np.copy(min_res.x)
        self.energy = min_res.fun
        self.incumbent_minres = min_res  # best minimize result found so far
        logger.info("Basin hopping step %d: f %g", self.nstep, self.energy)

        # initialize storage class
        self.storage = Storage(min_res)

        # Copy relevant counters from initial minimization result
        self._update_counters(min_res)

    def _update_counters(self, results: OptimizeResult):
        for attr in ["nfev", "njev", "nhev"]:
            if hasattr(results, attr):
                setattr(self.res, attr, getattr(results, attr))

    def _monte_carlo_step(self):
        """
        Perform one Monte Carlo step: take a step, minimize, and accept/reject.

        Randomly displace the coordinates, minimize, and decide whether to accept the new coordinates.
        """
        x_after_step = self.x.copy()
        x_after_step = self.step_taking(x_after_step)

        # Initialize
        # min_res = None
        accept = False

        try:
            # do a local minimization
            logger.debug(f"Basin hopping: New initial trial: {x_after_step.round(5)}")
            min_res = self.minimizer(x_after_step)
            # x_after_quench = min_res.x
            # energy_after_quench = min_res.fun

            if min_res.success:
                self.consecutive_failures = 0       # Reset counter on success
                accept = True                       # Proceed with accept tests if minimization successful
            else:
                self.res.minimization_failures += 1
                if self.disp:
                    logger.warning("Basin hopping: local minimization failure")

            # Accumulate function, Jacobian, Hessian evaluations
            self._update_counters(min_res)

        except (np.linalg.LinAlgError, ValueError, RuntimeError, FloatingPointError) as e:
            # LinAlgError: Singular matrix, convergence issues in linear algebra
            # ValueError: Invalid input parameters to optimizer
            # RuntimeError: Iteration limit exceeded or other runtime issues
            # FloatingPointError: NaN/Inf encountered during optimization
            self.consecutive_failures += 1
            self.res.minimization_failures += 1

            if self.disp:
                logger.warning(f"Local minimization failed with exception: {e}, consecutive failures: {self.consecutive_failures}")

            # Create a dummy OptimizeResult indicating failure
            min_res = OptimizeResult(x=x_after_step, fun=np.inf, success=False, message=f"Local minimization failed: {e}")

        if accept:
            # Run through all acceptance tests.
            for test_i in self.accept_tests:
                test_res_i = test_i(res_new=min_res, res_old=self.incumbent_minres) # Simplified signature
                if test_res_i == 'force accept':
                    accept = True
                    break
                elif test_res_i is None:
                    raise ValueError("accept_tests must return True, False, or 'force accept'")
                elif not test_res_i:
                    accept = False

        # Report the result of the acceptance test to the take step class. This is for adaptive step taking
        if hasattr(self.step_taking, "report"):
            self.step_taking.report(accept, res_new=min_res, res_old=self.incumbent_minres)  # Pass result objects

        if self.consecutive_failures > self.failure_threshold_restart:
            if self.disp:
                logger.warning(f"Restarting basin hopping due to excessive failures ({self.consecutive_failures} > {self.failure_threshold_restart})")
            return False, min_res

        return accept, min_res

    def one_cycle(self):
        """
        Perform one full cycle of the basin hopping algorithm.

        Returns
        -------
        new_global_min : bool
            True if a new global minimum was found in this cycle, False otherwise.
        """

        self.nstep += 1
        new_global_min = False
        accept, min_res = self._monte_carlo_step()

        # Ensure min_res is valid and minimization was successful
        if accept and min_res is not None and min_res.success:
            self.energy = min_res.fun
            self.x = np.copy(min_res.x)

            # best minimize result found so far
            self.incumbent_minres = min_res
            new_global_min = self.storage.update(min_res)

        # print some information
        if self.disp:
            # Handle potential None min_res
            self.print_report(min_res.fun if min_res is not None else np.nan, accept)
            if new_global_min:
                logger.info(f"Found new global minimum on step {self.nstep} with function value {self.energy:.4f}")

        # save some variables as BasinHoppingRunner attributes
        self.xtrial = min_res.x if min_res is not None else self.xtrial
        self.energy_trial = min_res.fun
        self.accept = accept

        return new_global_min

    def print_report(self, energy_trial, accept):
        """Print status update."""

        minres = self.storage.get_lowest()
        logger.info(f"Basin hopping step {self.nstep}: f {self.energy:.4f}, trial_f {energy_trial:.4f}, "
                    f"accepted {int(accept)}, lowest_f {minres.fun:.4f}")


class AdaptiveStepSize:
    """
    Class to implement adaptive stepsize adjustment for the step taking method.

    This class wraps a step taking routine and modifies the stepsize to maintain
    a target step acceptance rate.

    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update

    """

    def __init__(self,
                 takestep,
                 accept_rate: float = 0.5,
                 interval: int = 50,
                 factor: float = 0.9,
                 verbose: bool = True,
                 ):

        if not hasattr(takestep, 'stepsize'):
            raise ValueError("The 'takestep' object must have a 'stepsize' attribute for adaptive stepsize to work.")

        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose

        if not 0 < accept_rate < 1:
            raise ValueError("Target acceptance rate must be between 0 and 1.")
        if not 0 < factor < 1:
            raise ValueError("Stepwise factor must be between 0 and 1.")

        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        return self.take_step(x)

    def _adjust_step_size(self, aggressiveness_factor=None):
        """Adjust step size based on acceptance rate."""
        
        # Guard against division by zero when no steps have been taken
        if self.nstep == 0:
            return

        old_stepsize = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            self.takestep.stepsize /= self.factor
        else:
            self.takestep.stepsize *= self.factor

        if self.verbose:
            logger.debug(f"Adaptive stepsize: acceptance rate {accept_rate:.4f} target "
                         f"{self.target_accept_rate:.4f} new stepsize {self.takestep.stepsize:.4g} old stepsize {old_stepsize:.4g}")

        self.nstep = 0
        self.naccept = 0

    def take_step(self, x):
        """Take a step and potentially adjust stepsize."""

        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
        return self.takestep(x)

    def report(self, accept, failure=False, **kwargs):
        """Report step result and adjust stepsize more aggressively on failure."""
        if accept:
            self.naccept += 1
        elif failure:
            self._adjust_step_size(aggressiveness_factor=self.factor * 0.5)
            self.nstep += 1
            self.nstep_tot += 1


class StepDisplacement:
    """
    Randomly displace coordinates with a uniform distribution, ensuring bounds are respected.

    Parameters
    ----------
    step_size : float, optional
        Maximum step size in each dimension. Default is 0.5.
    step_type : str, optional
        Step type, "random_displacement" or "pareto_displacement". Default is "random_displacement".
    x_l : np.ndarray, optional
        Lower bounds for design variables.
    x_u : np.ndarray, optional
        Upper bounds for design variables.
    rng : {None, int, `numpy.random.Generator`}, optional
        Random number generator. If None, a default RNG is used.
    alpha : float, optional
        Alpha parameter for Pareto distribution (used if step_type="pareto_displacement").
    """

    def __init__(self,
                 step_size: float = 0.5,
                 step_type: str = "random_displacement",
                 x_l: np.ndarray = None,
                 x_u: np.ndarray = None,
                 rng: int = None,
                 alpha: float = 1.5,
                 ):
        """Initialize RandomDisplacement with bounds and step type."""

        self._stepsize = step_size
        self._step_type = step_type.lower()         # Ensure lowercase for comparison
        # Guard against division by zero
        self._alpha = alpha if alpha != 0 else 1.5
        self.rng = np.random.default_rng(rng)       # Use default_rng directly

        if x_l is None:
            x_l = -1.e6

        if x_u is None:
            x_u = 1.e6

        self.x_l = x_l
        self.x_u = x_u
        self.dx = x_u - x_l

    def _uniform_displacement(self, x: np.ndarray) -> np.ndarray:
        """
        Compute uniform displacement ensuring bounds are respected.
        """
        # Calculate maximum allowed displacement in each direction
        max_step_down = np.minimum(self._stepsize, x - self.x_l)
        max_step_up = np.minimum(self._stepsize, self.x_u - x)

        # Generate random displacement within allowed range
        displacement = np.zeros_like(x)
        for i in range(len(x)):
            # Random displacement between -max_step_down[i] and +max_step_up[i]
            displacement[i] = self.rng.uniform(-max_step_down[i], max_step_up[i])

        return displacement

    def _pareto_displacement(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Pareto displacement ensuring bounds are respected.
        """
        direction = self.rng.choice([-1, 1], size=x.shape)
        pareto_disp = np.zeros_like(x)

        # Calculate maximum allowed displacement in each direction
        max_step_down = x - self.x_l
        max_step_up = self.x_u - x

        # Sample displacement magnitude from Pareto distribution
        dist = self.rng.pareto(self._alpha, size=x.shape) / self._alpha

        for i in range(len(x)):
            if direction[i] >= 0:
                # Moving up: limit by upper bound and stepsize
                max_allowed = min(float(max_step_up[i]), self._stepsize)
                pareto_disp[i] = max_allowed * dist[i] * direction[i]
            else:
                # Moving down: limit by lower bound and stepsize
                max_allowed = min(float(max_step_down[i]), self._stepsize)
                pareto_disp[i] = -max_allowed * dist[i] * abs(direction[i])

        return pareto_disp

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Displace coordinates randomly while ensuring bounds are never violated.
        """
        x = np.asarray(x)

        # Generate displacement that respects bounds
        if self._step_type == "random_displacement":
            displacement = self._uniform_displacement(x)
        elif self._step_type == "pareto_displacement":
            displacement = self._pareto_displacement(x)
        else:
            raise ValueError(
                f"Not a valid step type: '{self._step_type}'. "
                f"Valid options are: 'random_displacement', 'pareto_displacement'."
            )

        # Apply displacement
        x_new = x + displacement

        # Clamp to bounds as safety measure (should not be needed with correct implementation)
        x_new = np.clip(x_new, self.x_l, self.x_u)

        return x_new

    @property
    def stepsize(self):
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):
        if value <= 0:
            raise ValueError(f"stepsize must be positive, got {value}")
        self._stepsize = value


class Metropolis:
    """
    Metropolis acceptance criterion based on energy difference and temperature.

    Parameters
    ----------
    T : float
        Temperature parameter. Higher T allows for accepting larger increases in function value.
        T=0 corresponds to monotonic basin hopping (only downhill steps accepted).
    rng : {None, int, `numpy.random.Generator`}, optional
        Random number generator. If None, a default RNG is used.
    """

    def __init__(self, T: float, rng=None):
        if T < 0:
            raise ValueError("Temperature T must be non-negative.")

        # Avoid ZeroDivisionError
        self.beta = 1.0 / T if T != 0 else float('inf')
        self.rng = check_random_state(rng)

    def accept_reject(self, res_new, res_old):
        """
        Assuming the local search underlying res_new was successful:
        If new energy is lower than old, it will always be accepted.
        If new is higher than old, there is a chance it will be accepted, less likely for larger differences.
        """
        with np.errstate(invalid='ignore'):
            # The energy values being fed to Metropolis are 1-length arrays, and if
            # they are equal, their difference is 0, which gets multiplied by beta,
            # which is inf, and array([0]) * float('inf') causes
            #
            # RuntimeWarning: invalid value encountered in multiply
            #
            # Ignore this warning so when the algorithm is on a flat plane, it always
            # accepts the step, to try to move off the plane.
            prod = -(res_new.fun - res_old.fun) * self.beta
            w = math.exp(min(0, prod))

        rand = self.rng.uniform()
        return w >= rand and (res_new.success or not res_old.success)

    def __call__(self, *, res_new, res_old):
        """
        f_new and f_old are mandatory in kwargs
        """
        return bool(self.accept_reject(res_new, res_old))


def ipopt_basinhopping(ipopt_func: IpyoptMinimizer,
                       x0: np.ndarray,
                       x_l: np.ndarray = None,
                       x_u: np.ndarray = None,
                       niter: int = 100,
                       T: float = 1.,
                       stepsize: float = 0.5,
                       take_step=None,
                       accept_test=None,
                       callback=None,
                       interval: int = 50,
                       disp: bool = False,
                       niter_success: int = None,
                       rng: int = None,
                       target_accept_rate: float = 0.5,
                       stepwise_factor: float = 0.9,
                       step_taking_method: str = "random_displacement",
                       pareto_alpha: float = 1.5 # Pareto alpha parameter
                       ):
    """
    Global minimization via basin hopping using ipyopt as the local minimizer.

    Parameters:
      ipopt_func: callable IpOpt Class (IpyoptMinimizer)
      x0:       array_like, initial guess
      x_l:      array_like, lower bound
      x_u:      array_like, upper bound
      niter:    int, number of basin hopping iterations
      T:        float, temperature for the Metropolis criterion
      stepsize: float, maximum random step size in each coordinate
      take_step: callable, custom step-taking function (if not provided, a random displacement is used)
      accept_test: callable, extra acceptance test (if any)
      callback: callable, a function called after every local minimization
      interval: int, interval (in iterations) to adjust step size
      disp: bool, if True, display progress messages
      niter_success: int, stop if the global minimum candidate is unchanged for these many iterations
      rng: random state (or seed)
      target_accept_rate:       float, target acceptance rate for adaptive stepsize
      stepwise_factor:          float, factor for adjusting stepsize
      step_taking_method:
      pareto_alpha:

    Returns:
      res: OptimizeResult object with the lowest found minimum.
    """

    if target_accept_rate <= 0. or target_accept_rate >= 1.:
        raise ValueError('target_accept_rate has to be in range (0, 1)')

    if stepwise_factor <= 0. or stepwise_factor >= 1.:
        raise ValueError('stepwise_factor has to be in range (0, 1)')

    x0 = np.array(x0)
    rng = np.random.default_rng(rng)

    # Set up the step-taking function.
    if take_step is None:
        displacement = StepDisplacement(step_size=stepsize,
                                        x_l=x_l,
                                        x_u=x_u,
                                        rng=rng,
                                        step_type=step_taking_method,
                                        alpha=pareto_alpha,
                                        )
        take_step_wrapped = AdaptiveStepSize(displacement,
                                             interval=interval,
                                             accept_rate=target_accept_rate,
                                             factor=stepwise_factor,
                                             verbose=disp,
                                             )
    else:
        if not callable(take_step):
            raise TypeError("take_step must be callable")

        if hasattr(take_step, "stepsize"):
            take_step_wrapped = AdaptiveStepSize(take_step,
                                                 interval=interval,
                                                 accept_rate=target_accept_rate,
                                                 factor=stepwise_factor,
                                                 verbose=disp)
        else:
            take_step_wrapped = take_step

    # Set up acceptance tests. (Start with any user-supplied test, then add the Metropolis test.)
    accept_tests = []
    if accept_test is not None:
        if not callable(accept_test):
            raise TypeError("accept_test must be callable")
        accept_tests.append(accept_test)
    metropolis = Metropolis(T, rng=rng)
    accept_tests.append(metropolis)

    niter_success = niter + 2 if niter_success is None else niter_success

    # Initialize the basin hopping runner.
    bh = BasinHoppingRunner(x0, ipopt_func, take_step_wrapped, accept_tests, disp=disp)

    # Call callback on the initial minimization, if provided.
    if callable(callback):
        callback(bh.storage.minres.x, bh.storage.minres.fun, True, bh.nstep)

    # start main iteration loop
    count, i = 0, 0
    message = "requested number of basinhopping iterations completed successfully"
    for i in range(niter):
        new_global_min_i = bh.one_cycle()

        if callable(callback):
            val = callback(bh.xtrial, bh.energy_trial, bh.accept, bh.nstep) # Added step count to callback
            if val is not None and val:
                message = "callback function requested stop early by returning True"
                break

        count += 1
        if new_global_min_i:
            count = 0
        elif count > niter_success:
            message = "success condition satisfied"
            break

    # Prepare and return the result.
    res = bh.res
    res.lowest_optimization_result = bh.storage.get_lowest()
    res.x = np.copy(res.lowest_optimization_result.x)
    res.fun = res.lowest_optimization_result.fun
    res.message = message
    res.nit = i + 1
    res.success = res.lowest_optimization_result.success
    return res


###############################################################################
# --- Example Usage ---
###############################################################################

def bs_ex1():
    # Here we optimize the Rosenbrock function.
    # Define the objective and its gradient.
    def rosen_fun(x):
        return rosen(x)

    def rosen_grad(x, out: np.ndarray = None):
        return rosen_der(x)

    # Initial guess.
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    x0_lwr = np.zeros_like(x0)
    x0_upp = np.ones_like(x0) * 100.

    # (Optional) ipyopt options – for example, suppress printing:
    ipyopt_opts = {"print_level": 0,
                   "max_iter": 100,
                   "warm_start_init_point": "yes",
                   "warm_start_bound_push": 1.e-8,
                   "warm_start_slack_bound_push": 1.e-8,
                   "warm_start_mult_bound_push": 1.e-8}

    min_ipopt = IpyoptMinimizer(x0_lwr, x0_upp, rosen_fun, obj_grad_fun=rosen_grad, ipyopt_options=ipyopt_opts)

    # Run the basin hopping with ipyopt as the local minimizer.
    result = ipopt_basinhopping(min_ipopt, x0, niter=1000, disp=True)

    logger.info("\nFinal result:")
    logger.info("x = %s", result.x)
    logger.info("f(x) = %s", result.fun)
    logger.info("message: %s", result.message)


def bs_ex2():

    def rosen_fun(x):
        return rosen(x)


    def rosen_grad(x, out: np.ndarray = None):
        dfx = rosen_der(x)
        if out is not None:
            out[:] = dfx.astype(float)

        return dfx.astype(float)


    # Initial guess.
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    x0_lwr = np.zeros_like(x0)
    x0_upp = np.ones_like(x0) * 10.

    x0_bs_lwr = np.ones_like(x0) + 0.2
    x0_bs_upp = np.ones_like(x0) + 0.7

    # (Optional) ipyopt options – for example, suppress printing:
    ipyopt_opts = {"print_level": 5,
                   "max_iter": 1000,
                   "tol": 1e-09,
                   "acceptable_tol": 1e-07,
                   "warm_start_init_point": "yes",
                   "warm_start_bound_push": 1.e-8,
                   "warm_start_slack_bound_push": 1.e-8,
                   "warm_start_mult_bound_push": 1.e-8,
                   }

    min_ipopt = IpyoptMinimizer(x0_lwr, x0_upp, rosen_fun, obj_grad_fun=rosen_grad, ipyopt_options=ipyopt_opts)

    # Run the basin hopping with ipyopt as the local minimizer.
    result = ipopt_basinhopping(min_ipopt,
                                x0,
                                x_l=x0_bs_lwr,
                                x_u=x0_bs_upp,
                                niter=10,
                                disp=True,
                                )

    logger.info("\nFinal result:")
    logger.info("x = %s", result.x)
    logger.info("f(x) = %s", result.fun)
    logger.info("message: %s", result.message)


if __name__ == '__main__':
    bs_ex2()
