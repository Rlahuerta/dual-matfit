# -*- coding: utf-8 -*-

import unittest
import sympy as sy
from dualmatfit.tensor import TensorManager, IsotropicMaterialModel


class TestTensorManager(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test. We can set up some common resources here.
        self.mu = sy.symbols('mu')
        self.alpha = sy.symbols('alpha')
        self.a0 = sy.Matrix([sy.cos(self.alpha), sy.sin(self.alpha), 0])
        self.Mr = self.a0 * self.a0.T

        lx, ly, lz = sy.symbols('l_x l_y l_z')
        self.F_expr = sy.Matrix([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])

        self.F = sy.MatrixSymbol('F', 3, 3)
        self.C = sy.MatrixSymbol('C', 3, 3)
        self.B = sy.MatrixSymbol('B', 3, 3)

        self.tensor_index = {'F': self.F, 'C': self.C, 'B': self.B}
        self.tensor_expr = {self.F: self.F, self.C: self.F.T * self.F, self.B: self.F * self.F.T}
        self.manager = TensorManager(self.tensor_index, self.tensor_expr)

    def test_get_by_index(self):
        self.assertEqual(self.manager.get_symbol_by_index('F'), self.F)
        self.assertEqual(self.manager.get_symbol_by_index('C'), self.C)
        self.assertIsNone(self.manager.get_symbol_by_index('NonExistent'))

    def test_get_expression_by_symbol(self):
        self.assertEqual(self.manager.get_expression_by_symbol(self.C), self.F.T * self.F)

        # Test getting a non-existing symbol
        self.assertIsNone(self.manager.get_expression_by_symbol(sy.MatrixSymbol("G", 3, 3)))

    def test_get_expression_by_index(self):
        self.assertEqual(self.manager.get_expression_by_index('C'), self.F.T * self.F)
        self.assertIsNone(self.manager.get_expression_by_index('NonExistent'))

    def test_add(self):
        I_1 = sy.symbols('I_1')
        expr = sy.Trace(self.F.T * self.F)
        self.manager.add('I_1', I_1, expr)
        self.assertIn('I_1', self.manager.index_to_symbol)
        self.assertIn(I_1, self.manager.symbol_to_abstract_expression)

    def test_concrete_symbol_form(self):

        Ic_1 = sy.symbols('Ic_1')
        expr_c = sy.Trace(self.C)
        expr_ref_c = sy.Trace(self.F.T * self.F)
        self.manager.add('Ic_1', Ic_1, expr_c)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(Ic_1), expr_ref_c)

        Ib_1 = sy.symbols('Ib_1')
        expr_b = sy.Trace(self.B)
        expr_ref_b = sy.Trace(self.F * self.F.T)
        self.manager.add('Ib_1', Ib_1, expr_b)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(Ib_1), expr_ref_b)

        nh = sy.symbols('psi_iso')
        expr_nh = 0.5 * self.mu * sy.Trace(self.C)
        expr_ref_nh = 0.5 * self.mu * sy.Trace(self.F.T * self.F)
        self.manager.add('nh', nh, expr_nh)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(nh), expr_ref_nh)

        # Adding/Change de def. grad
        self.manager.add('F', self.F, self.F_expr)

        # Now comparing it
        expr_ref_nh = 0.5 * self.mu * sy.Trace(self.F_expr.T * self.F_expr)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(nh), expr_ref_nh.doit())

    def test_anisotropic_symbol_form(self):
        Iv_4 = sy.symbols('Iv_4')
        C_ref = self.F_expr.T * self.F_expr

        Iv4_abs_def = (sy.Trace(self.C * self.Mr)).doit()
        Iv4_abs_ref = (sy.Trace(C_ref * self.Mr)).doit()

        self.manager.add('F', self.F, self.F_expr)
        self.manager.add('Iv_4', Iv_4, Iv4_abs_def)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(Iv_4), Iv4_abs_ref)

    def test_find_symbol_by_expression(self):
        self.assertEqual(self.manager.find_symbol_by_expression(self.F.T * self.F), self.C)
        self.assertIsNone(self.manager.find_symbol_by_expression(sy.MatrixSymbol('G', 3, 3)))
        self.assertIsNone(self.manager.find_symbol_by_expression(sy.MatrixSymbol('G', 3, 3), abstract=False))

    def test_iterator(self):
        """Test iterating over the tensors."""

        for index, symbol, expression in self.manager:
            self.assertIn(index, self.tensor_index)
            self.assertIn(symbol, self.tensor_expr)

    def test_print_all(self):
        """Test the print_all method."""
        import logging

        # Add a more complex expression to test concrete form
        I_1 = sy.symbols('I_1')
        expr = sy.Trace(self.C)  # C is F.T * F
        self.manager.add('I_1', I_1, expr)

        # Add a concrete value for F
        lx, ly, lz = sy.symbols('l_x l_y l_z')
        F_expr = sy.Matrix([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
        self.manager.add('F', self.F, F_expr)

        # Capture logger output
        from dualmatfit.tensor import logger as tensor_logger
        
        log_capture = []
        class LogCapture(logging.Handler):
            def emit(self, record):
                log_capture.append(self.format(record))
        
        handler = LogCapture()
        handler.setLevel(logging.INFO)
        tensor_logger.addHandler(handler)
        original_level = tensor_logger.level
        tensor_logger.setLevel(logging.INFO)
        
        try:
            self.manager.print_all()
        except Exception as e:
            self.fail(f"print_all() raised {type(e)} unexpectedly!")
        finally:
            tensor_logger.removeHandler(handler)
            tensor_logger.setLevel(original_level)

        output = "\n".join(log_capture)

        # Basic validation of header
        self.assertIn("Index", output)
        self.assertIn("Symbol", output)
        self.assertIn("Abstract Expression", output)
        self.assertIn("Concrete Expression", output)

        # Check for content
        self.assertIn("F", output)
        self.assertIn("C", output)
        self.assertIn("B", output)
        self.assertIn("I_1", output)

        # Check for abstract expression of C
        self.assertIn("F.T*F", output)
        # Check for concrete expression of I_1
        self.assertIn("l_x**2 + l_y**2 + l_z**2", output)

    def test_print_all_with_non_sympy_expr(self):
        """Test print_all with a non-sympy expression (list)."""
        import logging

        # Add a non-sympy expression
        list_sym = sy.symbols('list_sym')
        list_expr = [1, 2, 3]
        self.manager.add('list_sym', list_sym, list_expr)

        # Capture logger output
        from dualmatfit.tensor import logger as tensor_logger
        
        log_capture = []
        class LogCapture(logging.Handler):
            def emit(self, record):
                log_capture.append(self.format(record))
        
        handler = LogCapture()
        handler.setLevel(logging.INFO)
        tensor_logger.addHandler(handler)
        original_level = tensor_logger.level
        tensor_logger.setLevel(logging.INFO)
        
        try:
            self.manager.print_all()
        except Exception as e:
            self.fail(f"print_all() raised {type(e)} unexpectedly with non-SymPy expression!")
        finally:
            tensor_logger.removeHandler(handler)
            tensor_logger.setLevel(original_level)

        output = "\n".join(log_capture)

        # Check that the list is printed correctly
        self.assertIn("list_sym", output)
        self.assertIn("[1, 2, 3]", output)
        self.assertIn("(equiv. to abstract)", output)

    def test_init_with_tensors_index_and_expressions(self):
        # Test initialization with both tensors_index and expressions
        F_sym, C_sym = sy.symbols('F C')
        F_expr = sy.MatrixSymbol('F', 3, 3)
        C_expr = F_expr.T * F_expr
        t_idx = {'F_key': F_sym, 'C_key': C_sym}
        exprs = {F_sym: F_expr, C_sym: C_expr}
        manager = TensorManager(tensors_index=t_idx, expressions=exprs)

        self.assertEqual(manager.get_symbol_by_index('F_key'), F_sym)
        self.assertEqual(manager.get_expression_by_symbol(F_sym), F_expr)
        self.assertEqual(manager.get_concrete_expression_by_symbol(F_sym), F_expr)

        self.assertEqual(manager.get_symbol_by_index('C_key'), C_sym)
        self.assertEqual(manager.get_expression_by_symbol(C_sym), C_expr)
        self.assertEqual(manager.get_concrete_expression_by_symbol(C_sym), C_expr)

    def test_init_with_only_expressions(self):
        # Test initialization with only expressions
        A, B = sy.symbols('A B')
        exprs = {A: sy.Integer(1), B: A + 2}
        manager = TensorManager(expressions=exprs)

        self.assertEqual(manager.get_symbol_by_index('A'), A)
        self.assertEqual(manager.get_expression_by_symbol(A), sy.Integer(1))
        self.assertEqual(manager.get_concrete_expression_by_symbol(A), sy.Integer(1))

        self.assertEqual(manager.get_symbol_by_index('B'), B)
        self.assertEqual(manager.get_expression_by_symbol(B), A + 2)
        self.assertEqual(manager.get_concrete_expression_by_symbol(B), sy.Integer(3))

    def test_init_empty(self):
        # Test initialization with no inputs
        manager = TensorManager()
        self.assertEqual(len(manager.index_to_symbol), 0)
        self.assertEqual(len(manager.symbol_to_abstract_expression), 0)

    def test_add_existing_index_different_symbol_raises_error(self):
        # Test adding a symbol with an existing index but a different symbol
        with self.assertRaises(ValueError):
            self.manager.add('F', sy.MatrixSymbol('G', 3, 3), sy.MatrixSymbol('G', 3, 3))

    def test_add_existing_index_same_symbol_updates(self):
        # Test adding a symbol with an existing index and the same symbol (should update)
        new_F_expr = sy.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.manager.add('F', self.F, new_F_expr)
        self.assertEqual(self.manager.get_expression_by_symbol(self.F), new_F_expr)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(self.F), new_F_expr)

    def test_get_concrete_expression_by_symbol_unresolved(self):
        # Test getting concrete expression for a symbol whose dependencies are not yet resolved
        X = sy.symbols('X')
        Y = sy.symbols('Y')
        self.manager.add('X', X, Y + 1)
        # Y is not in the manager, so X's concrete form should remain Y + 1
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(X), Y + 1)

    def test_get_concrete_expression_by_index_unresolved(self):
        # Test getting concrete expression by index for an unresolved symbol
        X = sy.symbols('X')
        Y = sy.symbols('Y')
        self.manager.add('X_idx', X, Y + 1)
        self.assertEqual(self.manager.get_concrete_expression_by_index('X_idx'), Y + 1)

    def test_get_concrete_expression_by_index_non_existent(self):
        # Test getting concrete expression by index for a non-existent index
        self.assertIsNone(self.manager.get_concrete_expression_by_index('NonExistent'))

    def test_get_direct_abstract_dependencies_no_deps(self):
        # Test getting dependencies for a symbol with no free symbols
        const_sym = sy.symbols('const_sym')
        self.manager.add('const', const_sym, sy.Integer(5))
        self.assertEqual(self.manager.get_direct_abstract_dependencies(const_sym), [])

    def test_get_direct_abstract_dependencies_with_deps(self):
        # Test getting dependencies for a symbol with dependencies
        A, B, C = sy.symbols('A B C')
        self.manager.add('A', A, B + C)
        self.assertEqual(set(self.manager.get_direct_abstract_dependencies(A)), {B, C})

    def test_resolve_single_concrete_form_literal(self):
        # Test _resolve_single_concrete_form with a literal (no free symbols)
        A = sy.symbols('A')
        self.manager.add('A', A, sy.Integer(10))
        # Should return False as no change is made (already concrete)
        self.assertFalse(self.manager._resolve_single_concrete_form(A))
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(A), sy.Integer(10))

    def test_set_simplify_concrete_flag(self):
        # Test setting the simplify flag and its effect on resolution
        x, y = sy.symbols('x y')
        expr_sym = sy.symbols('expr_sym')
        self.manager.add('expr_key', expr_sym, x + x + y)
        self.manager.set_simplify_concrete_flag(False)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(expr_sym), x + x + y)
        self.manager.set_simplify_concrete_flag(True)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(expr_sym), 2 * x + y)

    def test_resolve_all_concrete_forms_override_simplify(self):
        # Test resolve_all_concrete_forms with simplify_concrete_override
        a, b, c = sy.symbols('a b c')
        self.manager = TensorManager()
        self.manager.add('a', a, b + b)
        self.manager.add('b', b, c + c)
        self.manager.add('c', c, sy.Integer(1))

        # Initially, with simplify_concrete_expressions=True (default)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(a), sy.Integer(4))

        # Override to False for this call
        self.manager.resolve_all_concrete_forms(simplify_concrete_override=False)
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(a), 2*(2*sy.Integer(1)))

        # Should revert to original flag (True)
        self.manager.resolve_all_concrete_forms()
        self.assertEqual(self.manager.get_concrete_expression_by_symbol(a), sy.Integer(4))

    def test_find_symbol_by_expression_concrete(self):
        # Test finding symbol by concrete expression
        Ic_1 = sy.symbols('Ic_1')
        expr_c = sy.Trace(self.C)
        self.manager.add('Ic_1', Ic_1, expr_c)
        # After adding, C is F.T * F, so Ic_1's concrete form is Trace(F.T * F)
        expected_concrete = sy.Trace(self.F.T * self.F)
        found_symbol = self.manager.find_symbol_by_expression(expected_concrete, abstract=False)
        self.assertEqual(found_symbol, Ic_1)

    def test_find_symbol_by_expression_zero_matrix(self):
        # Test finding symbol when expression simplifies to ZeroMatrix
        Z = sy.symbols('Z')
        zero_mat = sy.Matrix([[0, 0], [0, 0]])
        expr_z = sy.zeros(2, 2)
        self.manager.add('Z', Z, expr_z)
        found_symbol = self.manager.find_symbol_by_expression(zero_mat)
        self.assertEqual(found_symbol, Z)

    def test_find_symbol_by_expression_mathematical_equivalence(self):
        # Test finding symbol when expressions are mathematically equivalent but not identical
        X, Y = sy.symbols('X Y')
        expr1 = (X + Y)**2
        expr2 = X**2 + 2*X*Y + Y**2
        self.manager.add('X_sq', X, expr1)
        found_symbol = self.manager.find_symbol_by_expression(expr2)
        self.assertEqual(found_symbol, X)

    def test_find_index_by_symbol_existing(self):
        # Test finding index for an existing symbol
        self.assertEqual(self.manager.find_index_by_symbol(self.F), 'F')

    def test_find_index_by_symbol_non_existing(self):
        # Test finding index for a non-existing symbol
        non_existent_sym = sy.symbols('NonExistent')
        self.assertIsNone(self.manager.find_index_by_symbol(non_existent_sym))

    def test_get_concrete_expression_for_non_existent_symbol(self):
        # Test getting concrete expression for a symbol not in the manager
        non_existent_sym = sy.symbols('NonExistent')
        # Should return None as the symbol is not in the manager
        self.assertIsNone(self.manager.get_concrete_expression_by_symbol(non_existent_sym))

    def test_repr(self):
        # Test the __repr__ method
        self.assertEqual(repr(self.manager), 'TensorManager with 3 symbols')


class TestIsotropicMaterialModel(unittest.TestCase):

    def setUp(self):
        self.F = sy.MatrixSymbol('F', 3, 3)
        self.tensor_manager = TensorManager(tensors_index={'F': self.F})

    def test_neo_hookean_model(self):
        mu = sy.symbols('mu')
        material = IsotropicMaterialModel(
            material_type='neo_hookean',
            tensor_manager=self.tensor_manager,
            mu=mu
        )
        psi = material.get_strain_energy()
        self.assertIsNotNone(psi)
        self.assertIn(mu, psi.free_symbols)

    def test_fung_model(self):
        a_f, b_f = sy.symbols('a_f b_f')
        material = IsotropicMaterialModel(
            material_type='fung',
            tensor_manager=self.tensor_manager,
            a_f=a_f,
            b_f=b_f
        )
        psi = material.get_strain_energy()
        self.assertIsNotNone(psi)
        self.assertIn(a_f, psi.free_symbols)
        self.assertIn(b_f, psi.free_symbols)

    def test_fung_general_model(self):
        a_f, q_f = sy.symbols('a_f q_f')
        material = IsotropicMaterialModel(
            material_type='fung_general',
            tensor_manager=self.tensor_manager,
            a_f=a_f,
            q_f=q_f
        )
        psi = material.get_strain_energy()
        self.assertIsNotNone(psi)
        self.assertIn(a_f, psi.free_symbols)
        self.assertIn(q_f, psi.free_symbols)

    def test_unsupported_material_model(self):
        with self.assertRaises(ValueError):
            IsotropicMaterialModel(
                material_type='unsupported',
                tensor_manager=self.tensor_manager
            )

    def test_missing_material_parameters(self):
        with self.assertRaises(ValueError):
            IsotropicMaterialModel(
                material_type='neo_hookean',
                tensor_manager=self.tensor_manager
            )

    def test_compute_pk1(self):
        mu = sy.symbols('mu')
        material = IsotropicMaterialModel(
            material_type='neo_hookean',
            tensor_manager=self.tensor_manager,
            mu=mu
        )
        pk1 = material.compute_pk1()
        self.assertIsNotNone(pk1)

    def test_compute_hessian(self):
        mu = sy.symbols('mu')
        material = IsotropicMaterialModel(
            material_type='neo_hookean',
            tensor_manager=self.tensor_manager,
            mu=mu
        )
        hessian = material.compute_hessian()
        self.assertIsNotNone(hessian)


if __name__ == '__main__':
    unittest.main()

