# -*- coding: utf-8 -*-
"""
Symbolic tensor management for material models.

This module provides the TensorManager class for handling symbolic
tensor expressions in SymPy for constructing strain energy functions
and their derivatives.
"""
# import warnings
import sympy as sy

from sympy.matrices.expressions.special import ZeroMatrix
from sympy.matrices.expressions.matexpr import MatrixExpr

from typing import Union, List, Dict, Optional
from dualmatfit.material_law import neo_hookean, fung, fung_general
from dualmatfit.simplify import safe_simplify

from dualmatfit.logging_config import get_logger
logger = get_logger('tensor')

__all__ = [
    'TensorManager',
    'safe_simplify',
]




class TensorManager:
    def __init__(self,
                 tensors_index: Optional[Dict[str, sy.Basic]] = None,
                 expressions: Optional[Dict[sy.Basic, sy.Basic]] = None,
                 simplify_concrete_expressions: bool = False,
                 simplify_intermediate_steps: bool = False,
                 simplify_timeout: int = 10,
                 ):
        """
        Initialize the TensorManager.

        This class manages SymPy symbols and their associated abstract and concrete expressions.
        It handles dependencies between expressions and iteratively resolves concrete forms.

        Performance Considerations
        --------------------------
        Symbolic simplification is expensive and often unnecessary during initialization.
        For best performance:
        - Use `simplify_concrete_expressions=False` (default) for fast initialization
        - Set `simplify_intermediate_steps=False` to avoid simplifying intermediate resolution steps
        - Only enable simplification when simplified expressions are required for downstream operations
          (e.g., code generation, LaTeX output)

        Example - Fast mode (recommended for tests and initialization):
            >>> manager = TensorManager(simplify_concrete_expressions=False)

        Example - Thorough mode (for production with code generation):
            >>> manager = TensorManager(
            ...     simplify_concrete_expressions=True,
            ...     simplify_intermediate_steps=False,
            ...     simplify_timeout=10
            ... )

        :param tensors_index: Optional dictionary mapping string keys (indices) to SymPy symbols.
                              These symbols represent the "names" or identifiers for the tensors.
        :param expressions: Optional dictionary mapping SymPy symbols to their abstract expressions.
        :param simplify_concrete_expressions: If True, final concrete expressions will be simplified.
                                               Defaults to False for performance.
        :param simplify_intermediate_steps: If True, simplifies after each resolution step during
                                             iterative resolution. Usually unnecessary and expensive.
                                             Defaults to False.
        :param simplify_timeout: Maximum time (seconds) for simplification attempts. Defaults to 10.
        """
        self.index_to_symbol: Dict[str, sy.Basic] = {}
        self.symbol_to_abstract_expression: Dict[sy.Basic, sy.Basic] = {}
        self.symbol_to_concrete_expression: Dict[sy.Basic, Optional[sy.Basic]] = {}
        self.symbol_to_direct_abstract_dependencies: Dict[sy.Basic, List[sy.Basic]] = {}

        self._simplify_concrete_flag: bool = simplify_concrete_expressions
        self._simplify_intermediate_steps: bool = simplify_intermediate_steps
        self._simplify_timeout: int = simplify_timeout

        # Process initial tensors_index if provided
        if tensors_index:
            for index, symbol in tensors_index.items():
                # Initialize with symbol itself as abstract expression, will be updated by 'expressions' if present
                self.index_to_symbol[index] = symbol
                self.symbol_to_abstract_expression[symbol] = symbol
                self.symbol_to_concrete_expression[symbol] = symbol # Initialize concrete with abstract
                try:
                    self.symbol_to_direct_abstract_dependencies[symbol] = list(symbol.free_symbols)
                except (AttributeError, TypeError) as e:
                    # AttributeError: Object doesn't have free_symbols attribute
                    # TypeError: free_symbols is not iterable
                    logger.debug(f"Warning! Found the following error: {e}")
                    self.symbol_to_direct_abstract_dependencies[symbol] = []

        # Process expressions, updating existing or adding new ones
        if expressions:
            for symbol, expr in expressions.items():
                index = self.find_index_by_symbol(symbol) or str(symbol)
                self.add(index, symbol, expr)

        # After initial population, resolve all concrete forms
        self._iterative_resolve_all()

    def __repr__(self):
        """
        Return a string representation of the TensorManager instance.
        """
        return f"TensorManager with {len(self.index_to_symbol)} symbols"

    def __iter__(self):
        """
        Iterator for (index, symbol, abstract_expression).
        """
        for index, symbol in self.index_to_symbol.items():
            abstract_expression = self.symbol_to_abstract_expression.get(symbol)
            if abstract_expression is not None:
                yield index, symbol, abstract_expression

    def get_symbol_by_index(self, index: str) -> Optional[sy.Basic]:
        """Get the symbol by its index key."""
        return self.index_to_symbol.get(index)

    def get_expression_by_symbol(self, symbol: sy.Basic) -> Optional[sy.Basic]:  # Gets abstract
        """Get the abstract expression for a given symbol."""
        return self.symbol_to_abstract_expression.get(symbol)

    def get_expression_by_index(self, index: str) -> Optional[sy.Basic]:  # Gets abstract
        """Get the abstract expression by its index key."""
        symbol = self.get_symbol_by_index(index)
        if symbol is not None:
            return self.get_expression_by_symbol(symbol)
        return None

    def get_concrete_expression_by_symbol(self, symbol: sy.Basic) -> Optional[sy.Basic]:
        """Get the stored concrete expression for a symbol."""
        return self.symbol_to_concrete_expression.get(symbol)

    def get_concrete_expression_by_index(self, index: str) -> Optional[sy.Basic]:
        """Get the stored concrete expression by its index key."""
        symbol = self.get_symbol_by_index(index)
        if symbol is not None:
            return self.get_concrete_expression_by_symbol(symbol)
        return None

    def get_direct_abstract_dependencies(self, symbol: sy.Basic) -> Optional[List[sy.Basic]]:
        """Get the list of direct free symbols of the abstract expression of a symbol."""
        return self.symbol_to_direct_abstract_dependencies.get(symbol)

    def set_simplify_concrete_flag(self, simplify_flag: bool, 
                                    simplify_intermediate: Optional[bool] = None):
        """
        Sets whether to simplify concrete expressions during computation.
        
        :param simplify_flag: Enable/disable final simplification
        :param simplify_intermediate: Enable/disable intermediate step simplification
        """
        self._simplify_concrete_flag = simplify_flag
        if simplify_intermediate is not None:
            self._simplify_intermediate_steps = simplify_intermediate
        self.resolve_all_concrete_forms()  # Re-resolve if flag changes

    @staticmethod
    def _should_simplify(expr: sy.Basic) -> bool:
        """
        Heuristic to determine if expression would benefit from simplification.
        
        Returns False for:
        - Non-SymPy objects
        - Atomic expressions (no operations)
        - Small expressions (< 10 operations)
        
        :param expr: Expression to evaluate
        :return: True if simplification is recommended
        """
        if not isinstance(expr, sy.Basic):
            return False
        if expr.is_Atom or expr.is_Symbol or expr.is_Number:
            return False
        # Simple size heuristic: count_ops returns operation count
        if hasattr(expr, 'count_ops'):
            try:
                if expr.count_ops() < 10:
                    return False
            except (AttributeError, TypeError):
                pass
        return True

    def _resolve_single_concrete_form(self, symbol_to_update: sy.Basic) -> bool:
        """
        Private helper to compute/update the concrete form for a single specified symbol.

        This method attempts to substitute the free symbols within `symbol_to_update`'s
        current expression (either its abstract form or its partially resolved concrete form)
        with their *already computed* concrete forms available in the manager.
        It performs one step of substitution. If the `_simplify_concrete_flag` is True,
        the resulting concrete expression is simplified.

        :param symbol_to_update: The SymPy symbol for which to compute/update the concrete form.
        :return: True if the concrete form of `symbol_to_update` was changed in this step, False otherwise.
        """
        abstract_expr = self.symbol_to_abstract_expression.get(symbol_to_update)
        if abstract_expr is None:
            self.symbol_to_concrete_expression[symbol_to_update] = None
            return False

        # Start with the current concrete form, or abstract if no concrete form yet
        current_expr_for_resolution = self.symbol_to_concrete_expression.get(symbol_to_update)
        if current_expr_for_resolution is None:
            current_expr_for_resolution = abstract_expr

        # Build substitution map from *currently available concrete expressions* of its free symbols
        subs_map_for_this_step = {}

        # Ensure the expression is evaluated to its basic form before checking free_symbols
        evaluated_expr = current_expr_for_resolution.doit() if hasattr(current_expr_for_resolution, 'doit') else current_expr_for_resolution

        if not isinstance(evaluated_expr, sy.Basic) or not hasattr(evaluated_expr, 'free_symbols') or not evaluated_expr.free_symbols:
            # Already a concrete literal or non-SymPy object, or has no free symbols, no further SymPy substitution.
            if self.symbol_to_concrete_expression.get(symbol_to_update) != evaluated_expr:
                self.symbol_to_concrete_expression[symbol_to_update] = evaluated_expr
                return True
            return False

        for free_sym in list(evaluated_expr.free_symbols):
            if free_sym == symbol_to_update: continue  # Avoid direct self-substitution

            concrete_form_of_free_sym = self.symbol_to_concrete_expression.get(free_sym)
            # Only use the dependency's concrete form if it's available and not the abstract symbol itself
            if concrete_form_of_free_sym is not None:
                subs_map_for_this_step[free_sym] = concrete_form_of_free_sym

        if not subs_map_for_this_step:
            # If no substitutions are possible, and the concrete form is just the abstract one,
            # it means it's either already concrete or its dependencies are not yet resolved.
            # If it was None and now it's abstract_expr, it's a change.
            if self.symbol_to_concrete_expression.get(
                    symbol_to_update) is None and current_expr_for_resolution == abstract_expr:
                self.symbol_to_concrete_expression[symbol_to_update] = abstract_expr
                return True  # Changed from None to abstract_expr
            return False  # No further substitutions to make in this step

        try:
            new_concrete_expr = current_expr_for_resolution.subs(subs_map_for_this_step).doit()
            
            # Simplification logic with multiple conditions:
            # 1. Final simplification must be enabled
            # 2. Expression must be complex enough to benefit
            # 3. Either intermediate simplification is enabled OR this is final (no more substitutions possible)
            if self._simplify_concrete_flag and self._should_simplify(new_concrete_expr):
                # Check if this is likely a final resolution (no more free symbols that we manage)
                is_likely_final = not any(
                    fs in self.symbol_to_abstract_expression 
                    for fs in (new_concrete_expr.free_symbols if hasattr(new_concrete_expr, 'free_symbols') else [])
                )
                
                if self._simplify_intermediate_steps or is_likely_final:
                    new_concrete_expr = safe_simplify(new_concrete_expr, timeout=self._simplify_timeout)

            if self.symbol_to_concrete_expression.get(symbol_to_update) != new_concrete_expr:
                self.symbol_to_concrete_expression[symbol_to_update] = new_concrete_expr
                return True  # Concrete form changed
            return False  # No change
        except AttributeError as e:
            index_key = self.find_index_by_symbol(symbol_to_update)
            logger.debug(
                f"Warning: Could not compute concrete form for symbol '{symbol_to_update}' (index '{index_key}'): {e}")
            # Keep the last known good concrete form or the abstract one if concrete computation fails
            if self.symbol_to_concrete_expression.get(symbol_to_update) is None:
                self.symbol_to_concrete_expression[symbol_to_update] = abstract_expr
            return False

    def _iterative_resolve_all(self):
        """
        Iteratively computes and updates all concrete forms until convergence.

        This private method performs multiple passes over all registered symbols.
        In each pass, it attempts to substitute abstract symbols with their already
        resolved concrete forms. The process continues until a full pass occurs
        where no concrete form changes, indicating convergence.
        A heuristic maximum number of passes is set to prevent infinite loops
        in cases of circular dependencies or non-converging expressions, though
        such cases should ideally be avoided in the input definitions.
        This iterative approach is robust for acyclic dependency graphs, guaranteeing
        resolution. For very complex graphs, consider external topological sorting
        for single-pass resolution if performance is critical.
        """
        max_passes = len(self.symbol_to_abstract_expression) + 3  # Heuristic based on number of symbols
        # logger.debug(f"Starting iterative resolution (max_passes={max_passes})...")
        for pass_num in range(max_passes):
            changed_in_this_pass = False
            # Iterate in the order of addition, or a sorted order if preferred, for some determinism
            # For now, iterating over keys should be fine for convergence.
            symbols_to_process = list(self.symbol_to_abstract_expression.keys())

            for symbol_key in symbols_to_process:
                if self._resolve_single_concrete_form(symbol_key):
                    changed_in_this_pass = True

            if not changed_in_this_pass:
                # logger.debug(f"Concrete forms converged after {pass_num + 1} passes.")
                return  # Converged

        logger.warning(" Concrete form resolution might not have fully converged after max passes.")

    def add(self,
            index: str,
            symbol: sy.Basic,
            abstract_expression: Union[sy.Basic, sy.Matrix, MatrixExpr, sy.MutableDenseNDimArray, List[sy.Basic]],
            ) -> None:
        """
        Add a new tensor definition (symbol and its abstract expression).
        
        This method registers a new symbol and its abstract expression with the manager.
        It automatically identifies and stores the direct free symbols (dependencies)
        of the abstract expression. After adding, it triggers a full iterative
        resolution of all concrete forms to ensure all dependent expressions are updated.
        
        If the `index` already exists but is mapped to a different `symbol`,
        a `ValueError` is raised. If the `index` exists and maps to the same `symbol`,
        or if the `symbol` already exists with a different `abstract_expression`,
        the existing entry is updated. This allows for re-definition of a symbol's
        abstract expression while maintaining its index-to-symbol mapping.

        :param index: A unique string key to identify the tensor.
        :param symbol: The SymPy symbol representing the tensor.
        :param abstract_expression: The SymPy expression defining the tensor's abstract form.
                                    Can be a basic SymPy expression, Matrix, MatrixExpr, or MutableDenseNDimArray.
        """
        if index in self.index_to_symbol and self.index_to_symbol[index] != symbol:
            raise ValueError(
                f"Index '{index}' already exists with a different symbol ('{self.index_to_symbol[index]}'). Cannot add new symbol '{symbol}'.")

        self.index_to_symbol[index] = symbol
        self.symbol_to_abstract_expression[symbol] = abstract_expression

        # Store direct dependencies of the abstract expression
        try:
            self.symbol_to_direct_abstract_dependencies[symbol] = list(abstract_expression.free_symbols)
        except (AttributeError, TypeError) as e:
            # AttributeError: Object doesn't have free_symbols attribute
            # TypeError: free_symbols is not iterable
            logger.debug(f"Warning: Could not get free symbols for abstract form of '{symbol}' (index '{index}'): {e}")
            self.symbol_to_direct_abstract_dependencies[symbol] = []

        # Initialize concrete expression with the abstract one
        self.symbol_to_concrete_expression[symbol] = abstract_expression

        # Trigger full iterative resolution for all symbols
        self._iterative_resolve_all()

    def resolve_all_concrete_forms(self, simplify_concrete_override: Optional[bool] = None) -> None:
        """
        Public method to explicitly trigger iterative recomputation of all concrete forms.
        Useful if simplification flags change or for ensuring resolution.

        :param simplify_concrete_override: Optionally override the instance's
                                           _simplify_concrete_flag for this resolution pass.
        """
        original_simplify_flag = self._simplify_concrete_flag
        if simplify_concrete_override is not None:
            self.set_simplify_concrete_flag(simplify_concrete_override)  # This will trigger a resolve pass

        self._iterative_resolve_all()  # Call the internal iterative resolver

        if simplify_concrete_override is not None:  # Restore original flag if it was overridden
            self._simplify_concrete_flag = original_simplify_flag

    def find_symbol_by_expression(self,
                                  target_expression: sy.Basic,
                                  abstract: bool = True,
                                  ) -> Optional[sy.Basic]:
        source_dict = self.symbol_to_abstract_expression if abstract else self.symbol_to_concrete_expression
        for symbol_i, expression_i in source_dict.items():
            if expression_i is not None:
                if isinstance(expression_i, sy.Basic) and isinstance(target_expression, sy.Basic):
                    # Both are SymPy objects, use SymPy's robust comparison methods
                    # Prioritize .equals() for exact structural and mathematical equivalence
                    if expression_i.equals(target_expression):
                        return symbol_i

                    # Evaluate expression_i to its basic form before comparison for robustness
                    evaluated_expression_i = expression_i.doit() if hasattr(expression_i, 'doit') else expression_i

                    # Special handling for ZeroMatrix comparisons
                    if isinstance(evaluated_expression_i, ZeroMatrix) and isinstance(target_expression, sy.Matrix):
                        if evaluated_expression_i.shape == target_expression.shape and target_expression.is_zero_matrix:
                            return symbol_i
                    elif isinstance(target_expression, ZeroMatrix) and isinstance(evaluated_expression_i, sy.Matrix):
                        if target_expression.shape == evaluated_expression_i.shape and evaluated_expression_i.is_zero_matrix:
                            return symbol_i

                    # Determine if expressions are matrix-like
                    is_expr_matrix = isinstance(evaluated_expression_i, (sy.Matrix, MatrixExpr))
                    is_target_matrix = isinstance(target_expression, (sy.Matrix, MatrixExpr))

                    # If one is a matrix and the other is a scalar, they are not mathematically equivalent
                    if is_expr_matrix != is_target_matrix:
                        continue  # Skip to next symbol_i

                    # If both are matrices, ensure shapes are compatible before attempting subtraction
                    if is_expr_matrix and evaluated_expression_i.shape != target_expression.shape:
                        continue  # Skip if matrix shapes are incompatible

                    # Fallback for cases where .equals() might not catch mathematical equivalence
                    # This part is only reached if types are compatible (both matrix or both scalar)
                    diff = safe_simplify(evaluated_expression_i - target_expression, timeout=self._simplify_timeout)
                    if diff == sy.S(0):
                        return symbol_i
                    if isinstance(diff, ZeroMatrix) and diff.is_ZeroMatrix:
                        return symbol_i
                else:
                    # One or both are not SymPy objects, use direct comparison
                    if expression_i == target_expression:
                        return symbol_i
        return None

    def find_index_by_symbol(self, target_symbol: sy.Basic) -> Optional[str]:
        for index, symbol in self.index_to_symbol.items():
            if symbol == target_symbol:
                return index
        return None

    def print_all(self, max_str_len: int = 60) -> None:
        """
        Prints a formatted table of all managed tensors, including their index, symbol,
        direct abstract dependencies, abstract expression, and concrete expression.

        The output is truncated if expressions or dependencies exceed `max_str_len`.
        Concrete expressions are marked as "(equiv. to abstract)" if they simplify to the same
        expression as their abstract counterpart, or "<not computed>" if unresolved.
        
        Output is sent to the logger at INFO level.

        :param max_str_len: Maximum string length for displaying expressions and dependencies.
                            Longer strings will be truncated with "...". Defaults to 60.
        """
        col_width_index = 15
        col_width_symbol = 20
        col_width_deps = 30
        col_width_expr = max_str_len

        header = (
            f'{"Index":<{col_width_index}} '
            f'{"Symbol":<{col_width_symbol}} '
            f'{"Direct Abs Deps":<{col_width_deps}} '
            f'{"Abstract Expression":<{col_width_expr}} '
            f'{"Concrete Expression":<{col_width_expr}}'
        )
        logger.info(header)
        logger.info("=" * len(header))

        for index, symbol, abstract_expression in self:
            abstract_expr_str = str(abstract_expression)

            dependencies = self.get_direct_abstract_dependencies(symbol)
            deps_str = ", ".join(sorted(str(s) for s in dependencies)) if dependencies else "-"

            concrete_expr = self.get_concrete_expression_by_symbol(symbol)
            concrete_expr_str = ""
            if concrete_expr is not None:
                try:
                    # Check if both expressions are SymPy objects before simplifying
                    if isinstance(abstract_expression, sy.Basic) and isinstance(concrete_expr, sy.Basic):
                        # Attempt robust SymPy comparison
                        simplified_abstract_for_comp = safe_simplify(
                            abstract_expression.doit() if hasattr(abstract_expression, 'doit') else abstract_expression, timeout=self._simplify_timeout)
                        simplified_concrete_for_comp = safe_simplify(
                            concrete_expr.doit() if hasattr(concrete_expr, 'doit') else concrete_expr, timeout=self._simplify_timeout)

                        if simplified_concrete_for_comp.equals(simplified_abstract_for_comp):
                            concrete_expr_str = "(equiv. to abstract)"
                        else:
                            concrete_expr_str = str(concrete_expr)
                    else:
                        # For non-SymPy objects, simple string comparison
                        if str(concrete_expr) == str(abstract_expression):
                            concrete_expr_str = "(equiv. to abstract)"
                        else:
                            concrete_expr_str = str(concrete_expr)
                except (AttributeError, TypeError):
                    # Fallback for any errors
                    if str(concrete_expr) == str(abstract_expression):
                        concrete_expr_str = "(equiv. to abstract)"
                    else:
                        concrete_expr_str = str(concrete_expr)
            else:
                concrete_expr_str = "<not computed>"

            if len(deps_str) > col_width_deps: deps_str = deps_str[:col_width_deps - 3] + "..."
            if len(abstract_expr_str) > col_width_expr: abstract_expr_str = abstract_expr_str[
                                                                            :col_width_expr - 3] + "..."
            if len(concrete_expr_str) > col_width_expr: concrete_expr_str = concrete_expr_str[
                                                                            :col_width_expr - 3] + "..."

            row = (
                f'{index:<{col_width_index}} '
                f'{str(symbol):<{col_width_symbol}} '
                f'{deps_str:<{col_width_deps}} '
                f'{abstract_expr_str:<{col_width_expr}} '
                f'{concrete_expr_str:<{col_width_expr}}'
            )
            logger.info(row)


class IsotropicMaterialModel:

    # Define constants for isotropic material types
    NEO_HOOKEAN = 'nh'
    FUNG = 'fung'

    def __init__(self,
                 material_type: str,
                 tensor_manager: TensorManager,
                 isochoric: bool = False,
                 volumetric: bool = True,
                 **material_params,
                 ):
        """
        Initialize the IsotropicMaterialModel.

        :param material_type:       Type of isotropic material model (e.g., 'neo_hookean', 'fung', 'fung_general').
        :param tensor_manager:      Instance of TensorManager to store tensors.
        :param isochoric:           Flag for isochoric split on the deformation gradient.
        :param volumetric:          Flag for volumetric stabilization (only for Neo-Hookean).
        :param material_params:     Material parameters specific to the chosen model.
        """
        self.material_type = material_type.lower()
        self.tensor_manager = tensor_manager
        self.isochoric = isochoric
        self.volumetric = volumetric
        self.material_params = material_params

        # Initialize strain energy and PK1 tensor expressions
        self._psi_iso = None
        self._pk1_iso = None

        # Validate and initialize the material model
        self._initialize_material_model()

    def _initialize_material_model(self):
        """
        Set up the isotropic material model based on the selected type.
        """
        # Retrieve deformation gradient tensor from the TensorManager
        F = self.tensor_manager.get_symbol_by_index('F')

        if F is None:
            raise ValueError("Deformation gradient 'F' must be defined in TensorManager.")

        if self.material_type in [self.NEO_HOOKEAN, 'neo_hookean']:
            mu = self.material_params.get('mu')
            if mu is None:
                raise ValueError("Parameter 'mu' is required for Neo-Hookean model.")
            self._psi_iso = neo_hookean(mu, F, isochoric=self.isochoric, volumetric=self.volumetric)

        elif self.material_type == 'fung':
            a_f = self.material_params.get('a_f')
            b_f = self.material_params.get('b_f')
            if a_f is None or b_f is None:
                raise ValueError("Parameters 'a_f' and 'b_f' are required for Fung model.")
            self._psi_iso = fung(a_f, b_f, F, isochoric=self.isochoric)

        elif self.material_type == 'fung_general':
            a_f = self.material_params.get('a_f')
            q_f = self.material_params.get('q_f')
            if a_f is None or q_f is None:
                raise ValueError("Parameters 'a_f' and 'q_f' are required for Fung-General model.")
            self._psi_iso = fung_general(a_f, q_f, F, isochoric=self.isochoric)

        else:
            raise ValueError(f"Unsupported material type: {self.material_type}")

    def get_strain_energy(self) -> sy.Expr:
        """
        Return the isotropic strain energy expression.
        """
        return self._psi_iso

    def compute_pk1(self, variables: sy.Array = None) -> sy.MutableDenseNDimArray:
        """
        Compute the first Piola-Kirchhoff stress tensor (PK1) from the strain energy.

        :return: First Piola-Kirchhoff stress tensor.
        """
        if self._psi_iso is None:
            raise ValueError(
                f"Strain energy not initialized for material model '{self.material_type}'. "
                "Call _initialize_material_model() first."
            )

        if variables is None:
            F = self.tensor_manager.get_symbol_by_index('F')
            self._pk1_iso = sy.derive_by_array(self._psi_iso, F)
        elif isinstance(variables, sy.Array):
            self._pk1_iso = sy.derive_by_array(self._psi_iso, variables)
        else:
            raise ValueError(f"Unsupported variables type: {type(self.material_type)}")

        return self._pk1_iso

    def compute_hessian(self, variables: sy.Array = None) -> sy.MutableDenseNDimArray:
        """
        Compute the Hessian matrix (second derivative of the strain energy).

        :return: Hessian matrix of the strain energy function.
        """
        if self._psi_iso is None:
            raise ValueError(
                f"Strain energy not initialized for material model '{self.material_type}'. "
                "Call _initialize_material_model() first."
            )

        if variables is None:
            F = self.tensor_manager.get_symbol_by_index('F')
            hessian = sy.derive_by_array(self._psi_iso, F)
        elif isinstance(variables, sy.Array):
            hessian = sy.derive_by_array(self._psi_iso, variables)
        else:
            raise ValueError(f"Unsupported variables type: {type(self.material_type)}")

        return hessian
