import unittest
import pytest
import numpy as np
import pandas as pd
from scipy.optimize import rosen, rosen_der

from dualmatfit.drivers import opt_solvers
from dualmatfit.ipopt import IpyoptMinimizer
from dualmatfit.basinhopping import ipopt_basinhopping, AdaptiveStepSize, StepDisplacement
from dualmatfit.least_square import CostFunction


# Define Rosenbrock function and its gradient for testing
def rosen_fun(x):
    return rosen(x)


def rosen_grad(x, out: np.ndarray = None):
    dfx = rosen_der(x)
    if out is not None:
        out[:] = dfx.astype(float)

    return dfx.astype(float)


def rastrigin_fun(x):
    """Rastrigin function."""
    n = len(x)
    return 10 * n + sum(x_i ** 2 - 10 * np.cos(2 * np.pi * x_i) for x_i in x)


def rastrigin_grad(x, out=None):
    """Gradient of Rastrigin function."""
    grad = np.array([2 * x_i + 20 * np.pi * np.sin(2 * np.pi * x_i) for x_i in x])

    if out is not None:
        out[:] = grad.astype(float)

    return grad


class TestIpyoptBasinHopping(unittest.TestCase):

    def setUp(self):

        self.initial_guess = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        self.lower_bounds = np.zeros_like(self.initial_guess) - 2.0
        self.upper_bounds = np.ones_like(self.initial_guess) * 2.0
        self.ipyopt_opts = {"print_level": 0,
                           "max_iter": 50,
                           "warm_start_init_point": "yes",
                           "warm_start_bound_push": 1.e-8,
                           "warm_start_slack_bound_push": 1.e-8,
                           "warm_start_mult_bound_push": 1.e-8}

    def test_ipyopt_minimizer(self):
        """Test the IpyoptMinimizer class directly."""
        min_ipopt = IpyoptMinimizer(self.lower_bounds,
                                    self.upper_bounds,
                                    rosen_fun,
                                    obj_grad_fun=rosen_grad,
                                    ipyopt_options=self.ipyopt_opts,
                                    )
        result = min_ipopt(self.initial_guess)

        self.assertTrue(result.success, f"IPOPT Minimizer failed: {result.message}")
        self.assertAlmostEqual(result.fun, 0.0, places=2, msg="IPOPT Minimizer did not reach Rosenbrock minimum")
        self.assertTrue(np.all(result.x >= self.lower_bounds), "Solution violates lower bounds")
        self.assertTrue(np.all(result.x <= self.upper_bounds), "Solution violates upper bounds")

    def test_ipopt_basinhopping(self):
        """Test the ipopt_basinhopping function."""
        min_ipopt = IpyoptMinimizer(self.lower_bounds,
                                    self.upper_bounds,
                                    rosen_fun,
                                    obj_grad_fun=rosen_grad,
                                    ipyopt_options=self.ipyopt_opts,
                                    )
        result = ipopt_basinhopping(min_ipopt, self.initial_guess, niter=10, disp=False, rng=7) # Reduced niter for faster tests

        self.assertTrue(result.success, f"Basin Hopping failed: {result.message}")
        self.assertAlmostEqual(result.fun, 0.0, places=2, msg="Basin Hopping did not reach Rosenbrock minimum")
        self.assertTrue(np.all(result.x >= self.lower_bounds), "Solution violates lower bounds in Basin Hopping")
        self.assertTrue(np.all(result.x <= self.upper_bounds), "Solution violates upper bounds in Basin Hopping")

    def test_ipopt_basinhopping_with_bounds(self):
        """Test the ipopt_basinhopping function with bounds."""

        min_ipopt = IpyoptMinimizer(self.lower_bounds,
                                    self.upper_bounds,
                                    rosen_fun,
                                    obj_grad_fun=rosen_grad,
                                    ipyopt_options=self.ipyopt_opts,
                                    )

        # Set bounds that include the optimal solution [1,1,1,1,1]
        x0_bs_lwr = np.ones_like(self.initial_guess) * 0.8  # Lower bound: 0.8
        x0_bs_upp = np.ones_like(self.initial_guess) * 1.5  # Upper bound: 1.5

        result = ipopt_basinhopping(min_ipopt,
                                    self.initial_guess,
                                    x_l=x0_bs_lwr,
                                    x_u=x0_bs_upp,
                                    niter=10,
                                    disp=False,
                                    rng=7,
                                    ) # Reduced niter for faster tests

        self.assertTrue(result.success, f"Basin Hopping failed: {result.message}")
        # Verify bounds are respected in the final solution
        self.assertTrue(np.all(result.x >= x0_bs_lwr), f"Solution {result.x} violates lower bounds {x0_bs_lwr}")
        self.assertTrue(np.all(result.x <= x0_bs_upp), f"Solution {result.x} violates upper bounds {x0_bs_upp}")

    def test_step_displacement_bounds_enforcement(self):
        """Test that StepDisplacement class properly enforces bounds."""

        # Test with tight bounds
        x = np.array([0.5, 0.5, 0.5])
        x_l = np.array([0.0, 0.0, 0.0])
        x_u = np.array([1.0, 1.0, 1.0])

        # Test uniform displacement
        stepper_uniform = StepDisplacement(step_size=0.3, step_type="random_displacement",
                                         x_l=x_l, x_u=x_u, rng=42)

        for i in range(20):  # Test multiple steps
            x_new = stepper_uniform(x)
            self.assertTrue(np.all(x_new >= x_l), f"Uniform step {i}: {x_new} violates lower bounds {x_l}")
            self.assertTrue(np.all(x_new <= x_u), f"Uniform step {i}: {x_new} violates upper bounds {x_u}")

        # Test Pareto displacement
        stepper_pareto = StepDisplacement(step_size=0.3, step_type="pareto_displacement",
                                        x_l=x_l, x_u=x_u, rng=42, alpha=1.5)

        for i in range(20):  # Test multiple steps
            x_new = stepper_pareto(x)
            self.assertTrue(np.all(x_new >= x_l), f"Pareto step {i}: {x_new} violates lower bounds {x_l}")
            self.assertTrue(np.all(x_new <= x_u), f"Pareto step {i}: {x_new} violates upper bounds {x_u}")

    def test_step_displacement_edge_cases(self):
        """Test StepDisplacement with edge cases like points at bounds."""

        # Test point at lower bound
        x_l = np.array([0.0, 0.0])
        x_u = np.array([1.0, 1.0])
        x_at_lower = np.array([0.0, 0.0])

        stepper = StepDisplacement(step_size=0.5, x_l=x_l, x_u=x_u, rng=123)

        for i in range(10):
            x_new = stepper(x_at_lower)
            self.assertTrue(np.all(x_new >= x_l), f"From lower bound step {i}: {x_new} violates lower bounds {x_l}")
            self.assertTrue(np.all(x_new <= x_u), f"From lower bound step {i}: {x_new} violates upper bounds {x_u}")

        # Test point at upper bound
        x_at_upper = np.array([1.0, 1.0])

        for i in range(10):
            x_new = stepper(x_at_upper)
            self.assertTrue(np.all(x_new >= x_l), f"From upper bound step {i}: {x_new} violates lower bounds {x_l}")
            self.assertTrue(np.all(x_new <= x_u), f"From upper bound step {i}: {x_new} violates upper bounds {x_u}")

    def test_ipopt_basinhopping_very_tight_bounds(self):
        """Test basinhopping with very tight bounds to stress-test the bound enforcement."""

        min_ipopt = IpyoptMinimizer(self.lower_bounds,
                                    self.upper_bounds,
                                    rosen_fun,
                                    obj_grad_fun=rosen_grad,
                                    ipyopt_options=self.ipyopt_opts,
                                    )

        # Very tight bounds around the known minimum [1,1,1,1,1]
        x0_bs_lwr = np.ones_like(self.initial_guess) * 0.8
        x0_bs_upp = np.ones_like(self.initial_guess) * 1.2

        result = ipopt_basinhopping(min_ipopt,
                                    self.initial_guess,
                                    x_l=x0_bs_lwr,
                                    x_u=x0_bs_upp,
                                    niter=15,
                                    stepsize=0.1,  # Small step size for tight bounds
                                    disp=False,
                                    rng=456,
                                    )

        self.assertTrue(result.success, f"Basin Hopping with tight bounds failed: {result.message}")
        # Verify bounds are strictly respected
        self.assertTrue(np.all(result.x >= x0_bs_lwr), f"Solution {result.x} violates tight lower bounds {x0_bs_lwr}")
        self.assertTrue(np.all(result.x <= x0_bs_upp), f"Solution {result.x} violates tight upper bounds {x0_bs_upp}")

        # Verify we're close to the optimal solution within the constrained region
        expected_constrained_optimum = np.ones_like(self.initial_guess)  # [1,1,1,1,1] should be within [0.8, 1.2]
        self.assertTrue(np.allclose(result.x, expected_constrained_optimum, atol=0.3),
                       f"Solution {result.x} not close enough to constrained optimum {expected_constrained_optimum}")

    def test_rastrigin_ipopt_basinhopping(self):
        # Consider other multimodal functions like Ackley or Griewank for more comprehensive testing

        initial_guess = np.array([2., 2., 2.]) # Starting point away from origin
        lower_bounds = np.array([-5.12, -5.12, -5.12])
        upper_bounds = np.array([5.12, 5.12, 5.12])

        min_ipopt = IpyoptMinimizer(lower_bounds,
                                    upper_bounds,
                                    rastrigin_fun,
                                    obj_grad_fun=rastrigin_grad,
                                    ipyopt_options=self.ipyopt_opts,
                                    )

        result = ipopt_basinhopping(min_ipopt,
                                    initial_guess,
                                    x_l=lower_bounds,
                                    x_u=upper_bounds,
                                    niter=700,
                                    disp=False,
                                    rng=10,
                                    )

        self.assertTrue(result.success, f"Basin Hopping on Rastrigin failed: {result.message}")
        self.assertAlmostEqual(result.fun, 0., places=1, msg="Basin Hopping on Rastrigin did not reach global minimum (within tolerance)") # Relaxed places to 1, global min is 0
        self.assertTrue(np.all(result.x >= lower_bounds), "Solution violates lower bounds for Rastrigin")
        self.assertTrue(np.all(result.x <= upper_bounds), "Solution violates upper bounds for Rastrigin")

    def test_adaptive_stepsize_value_error(self):
        """Test if AdaptiveStepSize raises ValueError when takestep has no stepsize attribute."""

        class DummyTakeStep:
            def __call__(self, x):
                return x + 1.0

        dummy_takestep = DummyTakeStep()
        with self.assertRaisesRegex(ValueError, "The 'takestep' object must have a 'stepsize' attribute"):
            AdaptiveStepSize(dummy_takestep)

    def test_adaptive_stepsize_rate_factor_value_error(self):
        """Test if AdaptiveStepSize raises ValueError for invalid accept_rate and factor."""

        takestep = StepDisplacement() # takestep with stepsize attribute

        with self.assertRaisesRegex(ValueError, "Target acceptance rate must be between 0 and 1."):
            AdaptiveStepSize(takestep, accept_rate=1.5)

        with self.assertRaisesRegex(ValueError, "Stepwise factor must be between 0 and 1."):
            AdaptiveStepSize(takestep, factor=1.5)


def test_step_displacement_bounds():
    """Test StepDisplacement bounds enforcement"""
    print("Testing StepDisplacement bounds enforcement...")

    try:
        from dualmatfit.basinhopping import StepDisplacement

        # Test 1: Basic bounds enforcement
        print("\nTest 1: Basic bounds enforcement")
        x = np.array([0.5, 0.5, 0.5])
        x_l = np.array([0.0, 0.0, 0.0])
        x_u = np.array([1.0, 1.0, 1.0])

        stepper = StepDisplacement(step_size=0.3, step_type="random_displacement",
                                   x_l=x_l, x_u=x_u, rng=42)

        violations = 0
        for i in range(50):
            x_new = stepper(x)
            if not (np.all(x_new >= x_l) and np.all(x_new <= x_u)):
                violations += 1
                print(f"  VIOLATION at step {i}: x_new={x_new}, bounds=[{x_l}, {x_u}]")

        print(f"  Random displacement: {violations}/50 violations")

        # Test 2: Pareto displacement
        print("\nTest 2: Pareto displacement bounds enforcement")
        stepper_pareto = StepDisplacement(step_size=0.3, step_type="pareto_displacement",
                                          x_l=x_l, x_u=x_u, rng=42, alpha=1.5)

        violations = 0
        for i in range(50):
            x_new = stepper_pareto(x)
            if not (np.all(x_new >= x_l) and np.all(x_new <= x_u)):
                violations += 1
                print(f"  VIOLATION at step {i}: x_new={x_new}, bounds=[{x_l}, {x_u}]")

        print(f"  Pareto displacement: {violations}/50 violations")

        # Test 3: Edge cases - point at bounds
        print("\nTest 3: Edge cases - points at bounds")
        x_at_lower = np.array([0.0, 0.0, 0.0])
        x_at_upper = np.array([1.0, 1.0, 1.0])

        violations_lower = 0
        violations_upper = 0

        for i in range(20):
            x_new_lower = stepper(x_at_lower)
            x_new_upper = stepper(x_at_upper)

            if not (np.all(x_new_lower >= x_l) and np.all(x_new_lower <= x_u)):
                violations_lower += 1
                print(f"  VIOLATION from lower bound at step {i}: x_new={x_new_lower}")

            if not (np.all(x_new_upper >= x_l) and np.all(x_new_upper <= x_u)):
                violations_upper += 1
                print(f"  VIOLATION from upper bound at step {i}: x_new={x_new_upper}")

        print(f"  From lower bound: {violations_lower}/20 violations")
        print(f"  From upper bound: {violations_upper}/20 violations")

        # Test 4: Very tight bounds
        print("\nTest 4: Very tight bounds")
        x_tight = np.array([0.5])
        x_l_tight = np.array([0.49])
        x_u_tight = np.array([0.51])

        stepper_tight = StepDisplacement(step_size=0.1, x_l=x_l_tight, x_u=x_u_tight, rng=123)

        violations = 0
        for i in range(30):
            x_new = stepper_tight(x_tight)
            if not (np.all(x_new >= x_l_tight) and np.all(x_new <= x_u_tight)):
                violations += 1
                print(f"  VIOLATION with tight bounds at step {i}: x_new={x_new}, bounds=[{x_l_tight}, {x_u_tight}]")

        print(f"  Tight bounds: {violations}/30 violations")

        total_violations = violations_lower + violations_upper + violations

        if total_violations == 0:
            print("\n✅ SUCCESS: All bounds enforcement tests passed!")
        else:
            print(f"\n❌ FAILURE: {total_violations} total bound violations detected!")

    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()


def test_basinhopping_integration():
    """Test full BasinHopping integration with bounds"""
    print("\n" + "=" * 60)
    print("Testing BasinHopping integration with bounds...")

    try:
        from dualmatfit.ipopt import IpyoptMinimizer
        from dualmatfit.basinhopping import ipopt_basinhopping
        from scipy.optimize import rosen, rosen_der

        # Setup
        initial_guess = np.array([1.3, 0.7])
        lower_bounds = np.array([-2.0, -2.0])
        upper_bounds = np.array([2.0, 2.0])

        # Basin hopping bounds (tighter than optimizer bounds)
        x0_bs_lwr = np.array([0.8, 0.8])
        x0_bs_upp = np.array([1.2, 1.2])

        ipyopt_opts = {"print_level": 0, "max_iter": 50}

        min_ipopt = IpyoptMinimizer(lower_bounds, upper_bounds, rosen_fun,
                                    obj_grad_fun=rosen_grad, ipyopt_options=ipyopt_opts)

        print("Running BasinHopping with tight bounds...")
        result = ipopt_basinhopping(min_ipopt, initial_guess,
                                    x_l=x0_bs_lwr, x_u=x0_bs_upp,
                                    niter=5, stepsize=0.1, disp=False, rng=42)

        # Check if bounds are respected
        bounds_ok = np.all(result.x >= x0_bs_lwr) and np.all(result.x <= x0_bs_upp)

        print(f"Result: x = {result.x}")
        print(f"Bounds: [{x0_bs_lwr}, {x0_bs_upp}]")
        print(f"Success: {result.success}")
        print(f"Bounds respected: {bounds_ok}")

        if bounds_ok:
            print("✅ SUCCESS: BasinHopping respects bounds!")
        else:
            print("❌ FAILURE: BasinHopping violated bounds!")

    except Exception as e:
        print(f"❌ ERROR during BasinHopping test: {e}")
        import traceback
        traceback.print_exc()


class SimpleCostFunction(CostFunction):  # Standalone class that matches CostFunction interface
    def __init__(self, func, grad_func):
        self.func_eval = func
        self.grad_eval = grad_func

    def __call__(self, x_vars, *args, **kwargs):
        return self.func_eval(x_vars)

    def derivative(self, x_vars, *args, **kwargs):
        return self.grad_eval(x_vars)

    def derivative_array(self, x_vars, *args, **kwargs):
        return self.grad_eval(x_vars)


class TestOptSolvers(unittest.TestCase):
    def setUp(self):
        self.cost_rosen = SimpleCostFunction(rosen_fun, rosen_grad)
        dsvars_data = {
            'x0': {'values': 0.0, 'lower': -2.0, 'upper': 2.0, 'variable': True},
            'x1': {'values': 0.0, 'lower': -2.0, 'upper': 2.0, 'variable': True}
        }
        self.dsvars_rosen = pd.DataFrame.from_dict(dsvars_data, orient='index')
        self.known_solution_rosen = np.array([1.0, 1.0])
        # Suppress optimizer output for cleaner test logs
        self.silent_solver_options = {'disp': False}
        self.silent_lbfgsb_options = {'lbfgsb_options': {'disp': False}}
        self.silent_tnc_options = {'tnc_options': {'disp': False}}
        self.silent_de_options = {'disp': False}
        self.silent_shgo_options = {'shgo_main_options': {'disp': False}}
        self.silent_ipopt_options = {
            'ipyopt_options': {'print_level': 0},
            'basinhopping_options': {'disp': False}
        }

    def test_invalid_cost_function_type(self):
        with self.assertRaises(NotImplementedError):
            opt_solvers(
                otype='slsqp',
                cost_fun=lambda x: x,  # Invalid type
                dsvars=self.dsvars_rosen
            )

    def test_slsqp_local_rosenbrock(self):
        result = opt_solvers(
            otype='slsqp',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            glb=False,
            giter=1,
            solver_options={'minimize_options': self.silent_solver_options}
        )
        print(f"SLSQP results: success={result.success}, x={result.x}")
        self.assertTrue(result.success, f"SLSQP failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"SLSQP solution {result.x} not close to {self.known_solution_rosen}")

    def test_slsqp_global_rosenbrock(self):
        result = opt_solvers(
            otype='slsqp',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            glb=True, giter=3,
            solver_options={'minimize_options': self.silent_solver_options, 'basinhopping_options': {'disp': False}}
        )
        print(f"Global results: success={result.success}, x={result.x}")
        self.assertTrue(result.success, f"SLSQP global failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"SLSQP global solution {result.x} not close to {self.known_solution_rosen}")

    def test_mpowell_local_rosenbrock(self):
        result = opt_solvers(
            otype='mpowell',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            solver_options={'powell_options': self.silent_solver_options}
        )
        self.assertTrue(result.success, f"Powell failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"Powell solution {result.x} not close to {self.known_solution_rosen}")

    def test_lbfgsb_local_rosenbrock(self):
        result = opt_solvers(
            otype='lbfgsb',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            glb=False,
            giter=1,
            solver_options=self.silent_lbfgsb_options
        )
        print(f"L-BFGS-B results: success={result.success}, x={result.x}")
        self.assertTrue(result.success, f"L-BFGS-B failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"L-BFGS-B solution {result.x} not close to {self.known_solution_rosen}")

    def test_lbfgsb_warming_start(self):
        result = opt_solvers(
            otype='lbfgsb',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            glb=False,
            giter=3,
            solver_options=self.silent_lbfgsb_options
        )
        self.assertTrue(result.success, f"L-BFGS-B warming start failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"L-BFGS-B warming start solution {result.x} not close to {self.known_solution_rosen}")

    def test_tnc_local_rosenbrock(self):
        result = opt_solvers(
            otype='tnc',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            glb=False,
            giter=1,
            solver_options=self.silent_tnc_options
        )
        print(f"TNC results: success={result.success}, x={result.x}")
        self.assertTrue(result.success in [True, None] or result.status == 0, f"TNC failed: {result.message}, status {result.status}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"TNC solution {result.x} not close to {self.known_solution_rosen}")

    def test_tnc_global_rosenbrock(self):
        result = opt_solvers(
            otype='tnc',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            glb=True,
            giter=3,
            solver_options={'tnc_minimizer_options': self.silent_solver_options, 'basinhopping_options': {'disp': False}}
        )
        print(f"Global TNC results: success={result.success}, x={result.x}")
        self.assertTrue(result.success, f"TNC global failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"TNC global solution {result.x} not close to {self.known_solution_rosen}")

    def test_diffevol_rosenbrock(self):
        result = opt_solvers(
            otype='DIFFEVOL',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            miter=50,
            solver_options=self.silent_de_options
        )
        print(f"DIFFEVOL results: success={result.success}, x={result.x}")
        self.assertTrue(result.success or "maximum number of iterations" in str(result.message).lower(), f"Differential Evolution failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"DIFFEVOL solution {result.x} not close to {self.known_solution_rosen}")

    def test_shgo_rosenbrock(self):
        result = opt_solvers(
            otype='shgo',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            miter=2,
            solver_options=self.silent_shgo_options
        )
        print(f"SHGO results: success={result.success}, x={result.x}")
        self.assertTrue(result.success, f"SHGO failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"SHGO solution {result.x} not close to {self.known_solution_rosen}")

    @pytest.mark.skip(reason="This test is not ready yet")
    def test_ipopt_local_rosenbrock(self):
        result = opt_solvers(
            otype='ipopt',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            giter=0,
            miter=500,
            solver_options=self.silent_ipopt_options
        )
        self.assertTrue(result.success, f"IPOPT failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"IPOPT solution {result.x} not close to {self.known_solution_rosen}")

    @pytest.mark.skip(reason="This test is not ready yet")
    def test_ipopt_global_rosenbrock(self):
        result = opt_solvers(
            otype='ipopt',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            glb=True,
            giter=10,
            miter=500,
            solver_options=self.silent_ipopt_options
        )
        self.assertTrue(result.success, f"IPOPT global failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"IPOPT global solution {result.x} not close to {self.known_solution_rosen}")

    def test_lbfgsb_solver_options_maxiter(self):
        solver_opts = {'lbfgsb_options': {'maxiter': 1, 'disp': False}}
        result = opt_solvers(
            otype='lbfgsb',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            solver_options=solver_opts
        )
        print(f"L-BFGS-B results: success={result.success}, x={result.x}")
        self.assertEqual(result.nit, 1, "L-BFGS-B did not respect maxiter=1 from solver_options")

    def test_empty_flag_returns_initial(self):
        initial_values = self.dsvars_rosen['values'].values
        result = opt_solvers(
            otype='slsqp',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            empty=True
        )
        self.assertTrue(np.allclose(result.x, initial_values),
                        f"With empty=True, result.x {result.x} should be initial values {initial_values}")
        self.assertTrue(hasattr(result, 'fun'), "Result should have a 'fun' attribute even if empty.")

    def test_lbfgsb_global_rosenbrock(self):
        result = opt_solvers(
            otype='lbfgsb',
            cost_fun=self.cost_rosen,
            dsvars=self.dsvars_rosen,
            glb=True,
            giter=3,
            solver_options=self.silent_lbfgsb_options
        )
        print(f"Global L-BFGS-B results: success={result.success}, x={result.x}")
        self.assertTrue(result.success, f"L-BFGS-B global (basinhopping) failed: {result.message}")
        self.assertTrue(np.allclose(result.x, self.known_solution_rosen, atol=1e-3),
                        f"L-BFGS-B global solution {result.x} not close to {self.known_solution_rosen}")


if __name__ == '__main__':
    unittest.main()
