# -*- coding: utf-8 -*-

import os
# import gc
import pytest
import unittest

import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

from dualmatfit.formulation.material_law import right_cauchy_fun, get_fiber_vector, heaviside, HeavisideFunction
from dualmatfit.formulation.tensor import safe_simplify

current_file_path = os.path.dirname(os.path.abspath(__file__))

# Base directory for tests plots
work_path = os.path.join(current_file_path, "tests_plots", "functions")

# Create the base directory if it doesn't exist
os.makedirs(work_path, exist_ok=True)
LIMIT_TIME = 60


class TestRightCauchyFunction(unittest.TestCase):
    """
    Use the following articles to validate the tensor operations:

    Vasta, Marcello, Alessio Gizzi, and Anna Pandolfi. "A spectral decomposition approach for the mechanical statistical
    characterization of distributed fiber-reinforced tissues."
    International Journal of Non-Linear Mechanics 106 (2018): 258-265.
    """

    @pytest.mark.timeout(LIMIT_TIME)
    def test_numpy(self):
        # Create a numpy array
        def_grad_np = np.array([[1, 2], [3, 4]])

        # Compute the right Cauchy tensor
        right_cauchy_np = right_cauchy_fun(def_grad_np)

        # Expected result
        expected = def_grad_np.T @ def_grad_np

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(right_cauchy_np, expected)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_sympy(self):
        # Create a sympy Matrix
        def_grad_sy = sy.Matrix([[1, 2], [3, 4]])

        # Compute the right Cauchy tensor
        right_cauchy_sy = right_cauchy_fun(def_grad_sy)

        # Expected result
        expected = def_grad_sy.T * def_grad_sy

        # Use the .equals() method for comparison
        self.assertTrue(right_cauchy_sy.equals(expected))

    @pytest.mark.timeout(LIMIT_TIME)
    def test_isochoric_numpy(self):
        # Create a numpy array with determinant not equal to 1
        def_grad_np = np.array([[2, 0], [0, 1]])

        # Compute the isochoric part of the right Cauchy tensor
        right_cauchy_np = right_cauchy_fun(def_grad_np, isochoric=True)

        # Compute expected isochoric right Cauchy tensor
        J = np.linalg.det(def_grad_np)  # Should be 2
        J_neg_third = J ** (-1/3)       # J^(-1/3)
        F_bar = J_neg_third * def_grad_np
        expected = F_bar.T @ F_bar

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(right_cauchy_np, expected)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_isochoric_sympy(self):
        # Create a sympy Matrix with determinant not equal to 1
        def_grad_sy = sy.Matrix([[2, 0], [0, 1]])

        # Compute the isochoric part of the right Cauchy tensor
        right_cauchy_sy = right_cauchy_fun(def_grad_sy, isochoric=True)

        # Compute expected isochoric right Cauchy tensor
        J = def_grad_sy.det()  # Should be 2
        J_neg_third = J ** (-sy.Rational(1, 3))  # J^(-1/3)
        F_bar = J_neg_third * def_grad_sy
        # expected = (F_bar.T * F_bar).applyfunc(sy.simplify)
        expected = safe_simplify(F_bar.T * F_bar)

        # Simplify expressions before comparison
        difference = (right_cauchy_sy - expected).applyfunc(sy.simplify)
        self.assertTrue(difference == sy.zeros(2, 2))

    @pytest.mark.timeout(LIMIT_TIME)
    def test_negative_jacobian_numpy(self):
        # Create a numpy array with negative determinant
        def_grad_np = np.array([[-1, 0], [0, 1]])

        # Expect ValueError when computing isochoric part
        with self.assertRaises(ValueError):
            right_cauchy_fun(def_grad_np, isochoric=True)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_zero_jacobian_numpy(self):
        # Create a numpy array with zero determinant
        def_grad_np = np.array([[1, 0], [0, 0]])

        # Expect ValueError when computing isochoric part
        with self.assertRaises(ValueError):
            right_cauchy_fun(def_grad_np, isochoric=True)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_3d_numpy(self):
        # Create a 3x3 numpy array
        def_grad_np = np.array([[1, 2, 0],
                                [3, 4, 0],
                                [0, 0, 1]])

        # Compute the right Cauchy tensor
        right_cauchy_np = right_cauchy_fun(def_grad_np)

        # Expected result
        expected = def_grad_np.T @ def_grad_np

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(right_cauchy_np, expected)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_3d_isochoric_numpy(self):
        # Create a 3x3 numpy array with determinant not equal to 1
        def_grad_np = np.array([[2, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])

        # Compute the isochoric part of the right Cauchy tensor
        right_cauchy_np = right_cauchy_fun(def_grad_np, isochoric=True)

        # Compute expected isochoric right Cauchy tensor
        J = np.linalg.det(def_grad_np)  # Should be 2
        J_neg_third = J ** (-1/3)       # J^(-1/3)
        F_bar = J_neg_third * def_grad_np
        expected = F_bar.T @ F_bar

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(right_cauchy_np, expected)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_unsupported_type(self):
        # Pass an unsupported type (e.g., list)
        def_grad_list = [[1, 2], [3, 4]]

        # Expect NotImplementedError
        with self.assertRaises(NotImplementedError):
            right_cauchy_fun(def_grad_list)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_symbolic_sympy(self):
        # Define symbolic variables
        a, b, c, d = sy.symbols('a b c d')
        def_grad_sy = sy.Matrix([[a, b], [c, d]])

        # Compute the right Cauchy tensor
        right_cauchy_sy = right_cauchy_fun(def_grad_sy)

        # Expected result
        expected = def_grad_sy.T * def_grad_sy

        # Simplify expressions before comparison
        difference = safe_simplify(right_cauchy_sy - expected)
        self.assertTrue(difference == sy.zeros(2, 2))


class TestHeavisideFunction(unittest.TestCase):
    """
    Unit tests for the HeavisideFunction and heaviside functions.
    """

    @classmethod
    def setUpClass(cls):
        # Define symbolic variables
        cls.x = sy.symbols('x', real=True)
        cls.k = 150.0  # Slope parameter for the Heaviside approximation

        # Define numeric x values for testing
        cls.num_points = 1000
        cls.x_values = np.linspace(0.5, 1.5, cls.num_points)  # Range around the step at x=1

    @pytest.mark.timeout(LIMIT_TIME)
    def test_heaviside_eval_symbolic(self):
        """
        Test the symbolic evaluation of the HeavisideFunction.
        """
        # Define the Heaviside expression symbolically
        heavi_expr = heaviside(self.x, self.k)

        # Define the expected expression
        expected_expr = 1 / (1 + sy.exp(-2.0 * self.k * (self.x - 1)))

        # Assert that the expressions are identical
        self.assertEqual(safe_simplify(heavi_expr - expected_expr), 0,
                         "HeavisideFunction symbolic evaluation does not match expected expression.")

    @pytest.mark.timeout(LIMIT_TIME)
    def test_heaviside_derivative_symbolic(self):
        """
        Test the symbolic derivative of the HeavisideFunction.
        """
        lk = 20.

        # Define the Heaviside expression
        heavi_expr = heaviside(self.x, lk)
        heavi_func = sy.lambdify(self.x, heavi_expr, modules='numpy')

        # Compute its derivative
        derivative = heavi_expr.diff(self.x)
        derivative_func = sy.lambdify(self.x, derivative, modules='numpy')

        # Compute finite difference derivative using central differences
        # x_vals = np.linspace(0.5, 1.5, 100)
        x_vals = self.x_values

        dx = 1e-6
        heavi_plus = heavi_func(x_vals + dx)
        heavi_minus = heavi_func(x_vals - dx)
        np_derivative_fd = (heavi_plus - heavi_minus) / (2 * dx)
        np_derivative_fd /= (np.abs(np_derivative_fd)).max()

        np_derivative = derivative_func(x_vals)
        np_derivative /= (np.abs(np_derivative)).max()

        deriv_diff = np_derivative_fd - np_derivative

        ################################################################################
        fig, ax = plt.subplots(1, figsize=(8, 6), dpi=700)
        fig.suptitle('Heaviside Function with Smooth Approximation')

        ax.plot(x_vals, heavi_plus, label=f'Heaviside(x, k={lk})', color='blue')
        ax.plot(x_vals, np_derivative_fd, linestyle='--', label=f'Heaviside(x, k={lk}) - Derivative FD', color='blue')
        ax.plot(x_vals, np_derivative, linestyle='-.', label=f'Heaviside(x, k={lk}) - Derivative Sympy', color='green')

        ax.axvline(x=1.0, color='red', linestyle='--', label='x=1')
        ax.axhline(y=0.0, color='black', linestyle=':', linewidth=0.5)
        ax.axhline(y=0.5, color='black', linestyle=':', linewidth=0.5)
        ax.axhline(y=1.0, color='black', linestyle=':', linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('Heaviside(x)')
        ax.legend(loc='best')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        plt.tight_layout()

        fig.savefig(f"{work_path}/heaviside_function_k{lk}.png")
        plt.close(fig)
        ################################################################################

        # Assert that the derivative matches the expected expression
        self.assertLessEqual(np.abs(deriv_diff).sum(), 1.e-3,
                         "Derivative of HeavisideFunction does not match expected derivative.")

    @pytest.mark.timeout(LIMIT_TIME)
    def test_heaviside_numeric_scalar(self):
        """
        Test the numerical evaluation of the heaviside function with scalar inputs.
        """
        k = self.k

        # Define test cases with expected outcomes
        test_cases = [
            {'x': 0.5, 'expected': 0.0},
            {'x': 1.0, 'expected': 0.5},
            {'x': 1.5, 'expected': 1.0},
        ]

        for case_i in test_cases:
            with self.subTest(x=case_i['x']):
                x_val = case_i['x']
                expected = case_i['expected']
                result = float(heaviside(x_val, k))

                diff_i = expected - result

                # Due to the smooth approximation, use a tolerance
                self.assertLessEqual(abs(diff_i), 1e-3,
                                       msg=f"Heaviside({x_val}) = {result}, expected approximately {expected}")

    @pytest.mark.timeout(LIMIT_TIME)
    def test_heaviside_numeric_array(self):
        """
        Test the numerical evaluation of the heaviside function with numpy array inputs.
        """
        k = self.k
        x_vals = self.x_values

        # Compute expected values using the same formula
        expected = 1.0 / (1.0 + np.exp(-2.0 * k * (x_vals - 1.0)))

        # Evaluate the heaviside function
        result = heaviside(x_vals, k)

        err_msg = "Heaviside function numerical array evaluation does not match expected values."

        # Assert that the result matches the expected values within a tolerance
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-3, err_msg=err_msg)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_heaviside_derivative_numeric_array(self):
        """
        Test the numerical derivative of the heaviside function.
        """
        k = self.k
        x_vals = self.x_values

        # Compute the Heaviside values
        heavi_vals = heaviside(x_vals, k)

        # Compute numerical derivative using finite differences
        dx = x_vals[1] - x_vals[0]
        numerical_derivative = np.gradient(heavi_vals, dx)

        # Compute expected derivative using the analytical expression
        expected_derivative = (2.0 * k * np.exp(-2.0 * k * (x_vals - 1.0))) / (1.0 + np.exp(-2.0 * k * (x_vals - 1.0)))**2

        # Assert that numerical derivative matches the expected derivative
        np.testing.assert_allclose(numerical_derivative, expected_derivative, rtol=1e-1, atol=1e-2,
                                   err_msg="Numerical derivative of Heaviside function does not match expected values.")

    @pytest.mark.timeout(LIMIT_TIME)
    def test_heaviside_plot(self):
        """
        Generate a plot of the Heaviside function for visual inspection.
        """
        k = self.k
        x_vals = self.x_values
        heavi_vals = heaviside(x_vals, k)

        fig, ax = plt.subplots(1, figsize=(8, 6), dpi=700)
        fig.suptitle('Heaviside Function with Smooth Approximation')

        ax.plot(x_vals, heavi_vals, label=f'Heaviside(x, k={k})', color='blue')
        ax.axvline(x=1.0, color='red', linestyle='--', label='x=1')
        ax.axhline(y=0.0, color='black', linestyle=':', linewidth=0.5)
        ax.axhline(y=0.5, color='black', linestyle=':', linewidth=0.5)
        ax.axhline(y=1.0, color='black', linestyle=':', linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('Heaviside(x)')
        ax.legend()
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        plt.tight_layout()

        fig.savefig(f"{work_path}/heaviside_function_k{self.k}.png")
        plt.close(fig)

    @pytest.mark.timeout(LIMIT_TIME)
    def test_heaviside_sympy_equality(self):
        """
        Test that heaviside(x, k) returns a HeavisideFunction instance when x is symbolic.
        """
        heavi_expr = heaviside(self.x, self.k)
        heavi_class_method = HeavisideFunction.eval(self.x, self.k)
        self.assertIsInstance(heavi_expr, type(heavi_class_method),
                              "heaviside(x, k) should return a HeavisideFunction instance for symbolic x.")

    @pytest.mark.timeout(LIMIT_TIME)
    def test_heaviside_zero_case(self):
        """
        Test the behavior of the heaviside function at x = 1.
        """
        k = self.k
        x_val = 1.0
        expected = 0.5
        result = heaviside(x_val, k)
        self.assertAlmostEqual(result, expected, places=5,
                               msg=f"Heaviside({x_val}) = {result}, expected {expected}")


class TestFiberVector(unittest.TestCase):

    def setUp(self):
        """
        Set up common variables for the tests.
        """
        # Define symbolic variable for alpha
        self.alpha_sym = sy.Symbol('alpha')

        # Define planes to test
        self.planes = {
            'XY': (0, 1),
            'XZ': (0, 2),
            'YZ': (1, 2)
        }

        # Define dimensions to test
        self.dimensions = [2, 3]

        # Define fiber sizes to test
        self.sizes = [1, 2, 3, 4]

        # Define angles to test (in degrees)
        self.angles_deg = [0, 30, 45, 60, 90]

        # Define a directory to save plots (modify as needed)
        self.plot_dir = "fiber_plots"

    @staticmethod
    def arrays_are_equal(array1: sy.Array, array2: sy.Array, tolerance=1e-6) -> bool:
        """
        Helper function to compare two SymPy Arrays element-wise within a specified tolerance.

        Parameters
        ----------
        array1 : sympy.Array
            First array to compare.
        array2 : sympy.Array
            Second array to compare.
        tolerance : float, optional (default=1e-6)
            Numerical tolerance for comparing floating-point numbers.

        Returns
        -------
        bool
            True if all corresponding elements are equal within the tolerance, False otherwise.
        """
        if array1.shape != array2.shape:
            return False

        for a, b in zip(array1, array2):
            # Evaluate numerical expressions if possible
            a_val = a.evalf() if isinstance(a, sy.Expr) else a
            b_val = b.evalf() if isinstance(b, sy.Expr) else b
            if not safe_simplify(a_val - b_val) == 0:
                return False
        return True

    @pytest.mark.timeout(LIMIT_TIME)
    def test_get_fiber_vector_output(self):
        """
        Test the output of get_fiber_vector for correctness.
        Verifies the number of fibers and their symbolic forms.
        """
        # Test for size=2 in 3D, XY plane
        fibers = get_fiber_vector(self.alpha_sym, size=2, dim=3, plane=self.planes['XY'])
        self.assertEqual(len(fibers), 2, "Should generate two fiber vectors.")

        # Check the form of the first fiber
        expected_fiber1 = sy.cos(self.alpha_sym) * sy.Array([1, 0, 0]) - sy.sin(self.alpha_sym) * sy.Array([0, 1, 0])
        self.assertTrue(self.arrays_are_equal(fibers[0], expected_fiber1), "First fiber vector does not match expected form.")

        # Check the form of the second fiber
        expected_fiber2 = sy.cos(self.alpha_sym) * sy.Array([1, 0, 0]) + sy.sin(self.alpha_sym) * sy.Array([0, 1, 0])
        self.assertTrue(self.arrays_are_equal(fibers[1], expected_fiber2), "Second fiber vector does not match expected form.")

    @pytest.mark.timeout(LIMIT_TIME)
    def test_get_fiber_vector_invalid_inputs(self):
        """
        Test that get_fiber_vector raises appropriate exceptions for invalid inputs.
        """

        # Invalid size
        with self.assertRaises(ValueError):
            get_fiber_vector(self.alpha_sym, size=0, dim=3, plane=(0, 1))

        # Invalid dimension
        with self.assertRaises(ValueError):
            get_fiber_vector(self.alpha_sym, size=2, dim=0, plane=(0, 1))

        # Invalid plane type
        with self.assertRaises(TypeError):
            get_fiber_vector(self.alpha_sym, size=2, dim=3, plane='XY')

        # Plane indices out of bounds
        with self.assertRaises(ValueError):
            get_fiber_vector(self.alpha_sym, size=2, dim=3, plane=(0, 3))

    @pytest.mark.timeout(LIMIT_TIME)
    def test_get_fiber_vector_multiple_planes(self):
        """
        Test get_fiber_vector function across multiple planes and dimensions.
        Verifies that fibers are correctly generated in different planes.
        """
        planes = {
            'XY': (0, 1),
            'XZ': (0, 2),
            'YZ': (1, 2)
        }

        for plane_name, plane_axes in planes.items():
            with self.subTest(plane=plane_name):
                fibers = get_fiber_vector(self.alpha_sym, size=2, dim=3, plane=plane_axes)
                self.assertEqual(len(fibers), 2, f"Should generate two fibers in {plane_name} plane.")

                # Check if fibers lie in the correct plane
                for fiber_i in fibers:
                    # The axis not in the plane should be zero
                    non_plane_axis = set(range(3)) - set(plane_axes)
                    for axis in non_plane_axis:
                        # Extract the element at the non-plane axis
                        element = fiber_i[axis]

                        # Simplify the expression to check if it's zero
                        simplified_element = safe_simplify(element)
                        self.assertTrue(simplified_element == 0,
                                        f"Fiber has non-zero component in axis {axis} outside the {plane_name} plane.")

    @pytest.mark.timeout(LIMIT_TIME)
    def test_get_fiber_vector_various_sizes(self):
        """
        Test get_fiber_vector function with various sizes.
        Ensures that the correct number of fibers is generated and they follow expected symmetry.
        """
        sizes = [1, 2, 3, 4]
        for size in sizes:
            with self.subTest(size=size):
                fibers = get_fiber_vector(self.alpha_sym, size=size, dim=3, plane=(0, 1))
                self.assertEqual(len(fibers), size, f"Should generate {size} fibers.")

                # For size=1, only one fiber at +alpha
                if size == 1:
                    expected_fiber = sy.cos(self.alpha_sym) * sy.Array([1, 0, 0]) - sy.sin(self.alpha_sym) * sy.Array(
                        [0, 1, 0])
                    self.assertTrue(self.arrays_are_equal(fibers[0], expected_fiber),
                                    "Single fiber vector does not match expected form.")

                # For size=2, fibers at +alpha and -alpha
                elif size == 2:
                    expected_fiber1 = sy.cos(self.alpha_sym) * sy.Array([1, 0, 0]) - sy.sin(
                        self.alpha_sym) * sy.Array([0, 1, 0])
                    expected_fiber2 = sy.cos(self.alpha_sym) * sy.Array([1, 0, 0]) + sy.sin(
                        self.alpha_sym) * sy.Array([0, 1, 0])
                    self.assertTrue(self.arrays_are_equal(fibers[0], expected_fiber1),
                                    "First fiber vector does not match expected positive angle form.")
                    self.assertTrue(self.arrays_are_equal(fibers[1], expected_fiber2),
                                    "Second fiber vector does not match expected negative angle form.")

                # For size>2, additional fibers can be implemented as needed
                else:
                    # For this example, we assume symmetry continues (pairs at +alpha and -alpha)
                    for i in range(0, size, 2):
                        if i + 1 < size:
                            expected_fiber_pos = sy.cos(self.alpha_sym) * sy.Array([1, 0, 0]) - sy.sin(
                                self.alpha_sym) * sy.Array([0, 1, 0])
                            expected_fiber_neg = sy.cos(self.alpha_sym) * sy.Array([1, 0, 0]) + sy.sin(
                                self.alpha_sym) * sy.Array([0, 1, 0])
                            self.assertTrue(self.arrays_are_equal(fibers[i], expected_fiber_pos),
                                            "Fiber vector does not match expected positive angle form.")
                            self.assertTrue(self.arrays_are_equal(fibers[i + 1], expected_fiber_neg),
                                            "Fiber vector does not match expected negative angle form.")
                        else:
                            # For odd sizes, the last fiber can be at +alpha
                            expected_fiber_pos = sy.cos(self.alpha_sym) * sy.Array([1, 0, 0]) - sy.sin(
                                self.alpha_sym) * sy.Array([0, 1, 0])
                            self.assertTrue(self.arrays_are_equal(fibers[i], expected_fiber_pos),
                                            "Fiber vector does not match expected positive angle form.")

    @pytest.mark.timeout(LIMIT_TIME)
    def test_get_fiber_vector_visual_2d(self):
        """
        Visual unit test for the get_fiber_vector function in 2D.
        Generates fiber vectors at various angles and plots them for visual verification.
        """
        # Define angles in degrees
        angles_deg = [0, 30, 45, 60, 90, 120, 135, 150, 180]

        # Set up the plot
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        origin = np.zeros(2)  # 2D origin

        for angle_deg in angles_deg:
            angle_rad = np.deg2rad(angle_deg)

            # Generate fiber vectors symbolically
            fibers_sym = get_fiber_vector(self.alpha_sym, size=2, dim=2, plane=(0, 1))

            # Substitute alpha with numerical value
            fibers_num = [fiber.subs(self.alpha_sym, angle_rad) for fiber in fibers_sym]

            # Convert SymPy expressions to NumPy arrays for plotting
            fibers_np = [np.array(fiber).astype(np.float64).flatten() for fiber in fibers_num]

            # Define colors for different angles
            color = plt.cm.viridis(angle_deg / 180)

            for fiber in fibers_np:
                ax.quiver(
                    origin[0], origin[1],  # Starting point (origin)
                    fiber[0], fiber[1],  # Vector components
                    angles='xy', scale_units='xy', scale=1, color=color, alpha=0.7
                )

        # Formatting the plot
        ax.set_title('Fiber Vectors in 2D Plane at Various Angles')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)
        ax.set_aspect('equal')

        # Create a ScalarMappable to generate a colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(angles_deg), vmax=max(angles_deg)))
        sm.set_array([])  # We set an empty array to avoid errors when creating the colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Fiber Angle (Degrees)')
        cbar.set_ticks(angles_deg)  # Set ticks to the angle values

        # Save and close the plot
        plt.tight_layout()
        plt.savefig(f"{work_path}/fiber_vectors_2d.png")
        plt.close()

    @pytest.mark.timeout(LIMIT_TIME)
    def test_get_fiber_vector_visual_multiple_planes_3d(self):
        """
        Visual unit test for get_fiber_vector function across multiple planes in 3D.
        Generates fiber vectors at various angles and plots them for visual verification.
        """
        # Define planes to visualize
        planes = {
            'XY': (0, 1),
            'XZ': (0, 2),
            'YZ': (1, 2)
        }

        # Define angles in degrees
        angles_deg = [0, 30, 45, 60, 90, 120, 135, 150, 180]

        # Create subplots for each plane
        fig = plt.figure(figsize=(18, 6))

        for idx, (plane_name, plane_axes) in enumerate(planes.items(), 1):
            ax = fig.add_subplot(1, 3, idx, projection='3d')
            ax.set_title(f'Fiber Vectors in {plane_name} Plane')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            ax.grid(True)
            ax.view_init(elev=20, azim=30)  # Adjust viewing angle

            for angle_deg in angles_deg:
                angle_rad = np.deg2rad(angle_deg)

                # Generate fiber vectors symbolically
                fibers_sym = get_fiber_vector(self.alpha_sym, size=2, dim=3, plane=plane_axes)

                # Substitute alpha with numerical value
                fibers_num = [fiber.subs(self.alpha_sym, angle_rad) for fiber in fibers_sym]

                # Convert SymPy expressions to NumPy arrays for plotting
                fibers_np = [np.array(fiber).astype(np.float64).flatten() for fiber in fibers_num]

                # Define color based on angle
                color = plt.cm.viridis(angle_deg / 180)

                for fiber in fibers_np:
                    ax.quiver(
                        0, 0, 0,  # Origin
                        fiber[0], fiber[1], fiber[2],  # Fiber components
                        length=1, normalize=True, color=color, alpha=0.7
                    )

            # Create colorbar for the last subplot
            if idx == len(planes):
                mappable = plt.cm.ScalarMappable(cmap='viridis',
                                                 norm=plt.Normalize(vmin=min(angles_deg), vmax=max(angles_deg)))
                mappable.set_array([])
                cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)
                cbar.set_label('Fiber Angle (Degrees)')

        # plt.tight_layout()
        plt.savefig(f"{work_path}/fiber_vectors_multiple_planes_3d.png")
        plt.close()

    @pytest.mark.timeout(LIMIT_TIME)
    def test_get_fiber_vector_dimension_variation(self):
        """
        Test get_fiber_vector function across different dimensions (2D and 3D).
        Verifies that fibers are correctly generated and plotted.
        """
        dimensions = [2, 3]
        planes = {
            2: {'XY': (0, 1)},
            3: {'XY': (0, 1), 'XZ': (0, 2), 'YZ': (1, 2)}
        }
        angles_deg = [0, 45, 90, 135, 180]

        for dim in dimensions:
            with self.subTest(dim=dim):
                if dim == 2:
                    # 2D plot
                    plt.figure(figsize=(8, 8))
                    ax = plt.gca()
                    ax.set_title('Fiber Vectors in 2D Plane at Various Angles')
                    ax.set_xlabel('X-axis')
                    ax.set_ylabel('Y-axis')
                    ax.set_xlim(-1.5, 1.5)
                    ax.set_ylim(-1.5, 1.5)
                    ax.grid(True)
                    ax.set_aspect('equal')

                    for angle_deg in angles_deg:
                        angle_rad = np.deg2rad(angle_deg)
                        fibers_sym = get_fiber_vector(self.alpha_sym, size=2, dim=dim, plane=(0, 1))
                        fibers_num = [fiber.subs(self.alpha_sym, angle_rad) for fiber in fibers_sym]
                        fibers_np = [np.array(fiber).astype(np.float64).flatten() for fiber in fibers_num]
                        color = plt.cm.viridis(angle_deg / 180)
                        for fiber in fibers_np:
                            ax.quiver(
                                0, 0,  # Origin
                                fiber[0], fiber[1],  # Fiber components
                                angles='xy', scale_units='xy', scale=1, color=color, alpha=0.7
                            )

                    # Create a ScalarMappable to generate a colorbar
                    sm = plt.cm.ScalarMappable(cmap='viridis',
                                               norm=plt.Normalize(vmin=min(angles_deg), vmax=max(angles_deg)))
                    sm.set_array([])  # We set an empty array to avoid errors when creating the colorbar
                    cbar = plt.colorbar(sm, ax=ax)
                    cbar.set_label('Fiber Angle (Degrees)')
                    cbar.set_ticks(angles_deg)  # Set ticks to the angle values

                    # Save and close the plot
                    plt.tight_layout()
                    plt.savefig(f"{work_path}/fiber_vectors_2d_multiple_angles.png")
                    plt.close()

                elif dim == 3:
                    # 3D plot across multiple planes
                    fig = plt.figure(figsize=(18, 6))
                    for idx, (plane_name, plane_axes) in enumerate(planes[dim].items(), 1):
                        ax = fig.add_subplot(1, len(planes[dim]), idx, projection='3d')
                        ax.set_title(f'Fiber Vectors in {plane_name} Plane')
                        ax.set_xlabel('X-axis')
                        ax.set_ylabel('Y-axis')
                        ax.set_zlabel('Z-axis')
                        ax.set_xlim([-1.5, 1.5])
                        ax.set_ylim([-1.5, 1.5])
                        ax.set_zlim([-1.5, 1.5])
                        ax.grid(True)
                        ax.view_init(elev=20, azim=30)  # Adjust viewing angle

                        for angle_deg in angles_deg:
                            angle_rad = np.deg2rad(angle_deg)
                            fibers_sym = get_fiber_vector(self.alpha_sym, size=2, dim=dim, plane=plane_axes)
                            fibers_num = [fiber.subs(self.alpha_sym, angle_rad) for fiber in fibers_sym]
                            fibers_np = [np.array(fiber).astype(np.float64).flatten() for fiber in fibers_num]
                            color = plt.cm.viridis(angle_deg / 180)
                            for fiber in fibers_np:
                                ax.quiver(
                                    0, 0, 0,  # Origin
                                    fiber[0], fiber[1], fiber[2],  # Fiber components
                                    length=1, normalize=True, color=color, alpha=0.7
                                )

                        # Add colorbar to the last subplot
                        if idx == len(planes[dim]):
                            # Create a ScalarMappable to generate a colorbar
                            sm = plt.cm.ScalarMappable(cmap='viridis',
                                                       norm=plt.Normalize(vmin=min(angles_deg), vmax=max(angles_deg)))
                            sm.set_array([])  # We set an empty array to avoid errors when creating the colorbar
                            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=10)
                            cbar.set_label('Fiber Angle (Degrees)')
                            cbar.set_ticks(angles_deg)  # Set ticks to the angle values

                    plt.tight_layout()
                    plt.savefig(f"{work_path}/fiber_vectors_3d_multiple_planes.png")
                    plt.close()


if __name__ == '__main__':
    unittest.main()
