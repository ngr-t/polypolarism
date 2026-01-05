"""AST analysis and data flow tracking."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from polypolarism.types import DataType, FrameType, Nullable
from polypolarism.dsl import parse_schema, ParseError
from polypolarism.ops.join import infer_join, JoinError
from polypolarism.ops.groupby import (
    infer_groupby_result,
    infer_agg_result_type,
    AggExpr,
    AggFunction,
    GroupByTypeError,
)
from polypolarism.expr_infer import infer_col, ColumnNotFoundError


class AnalysisError(Exception):
    """Error during analysis."""

    pass


# =============================================================================
# Data structures for function registry
# =============================================================================


@dataclass
class FunctionSignature:
    """Type signature for a DF-annotated function."""

    name: str
    parameters: dict[str, tuple[int, FrameType]]  # param_name -> (position, type)
    return_type: Optional[FrameType]
    lineno: int

    def get_param_by_position(self, position: int) -> Optional[tuple[str, FrameType]]:
        """Get parameter info by position."""
        for name, (idx, frame_type) in self.parameters.items():
            if idx == position:
                return (name, frame_type)
        return None


@dataclass
class FunctionInfo:
    """Information about a function (typed or untyped)."""

    name: str
    node: ast.FunctionDef  # AST node for body analysis
    signature: Optional[FunctionSignature]  # None if untyped
    inferred_returns: dict[tuple, FrameType] = field(default_factory=dict)


@dataclass
class FunctionRegistry:
    """Registry of all functions in a file."""

    functions: dict[str, FunctionInfo] = field(default_factory=dict)

    def register(self, info: FunctionInfo) -> None:
        """Register a function."""
        self.functions[info.name] = info

    def get(self, name: str) -> Optional[FunctionInfo]:
        """Get function info by name."""
        return self.functions.get(name)

    def has_signature(self, name: str) -> bool:
        """Check if function has a type signature."""
        info = self.functions.get(name)
        return info is not None and info.signature is not None


# =============================================================================
# Type compatibility checking
# =============================================================================


def _is_column_subtype(actual: DataType, expected: DataType) -> bool:
    """Check if actual is a subtype of expected.

    Rules:
    - T is subtype of T
    - T is subtype of Nullable[T]
    - Nullable[T] is NOT subtype of T
    """
    if actual == expected:
        return True

    # Non-nullable is subtype of nullable with same base
    if isinstance(expected, Nullable) and not isinstance(actual, Nullable):
        return actual == expected.inner

    return False


def _is_frame_subtype(actual: FrameType, expected: FrameType) -> bool:
    """Check if actual FrameType is subtype of expected.

    Rules:
    - actual must have all columns that expected has
    - Each column type must be a subtype
    - actual may have extra columns (structural subtyping)
    """
    for col_name, expected_type in expected.columns.items():
        if col_name not in actual.columns:
            return False
        actual_type = actual.columns[col_name]
        if not _is_column_subtype(actual_type, expected_type):
            return False
    return True


# =============================================================================
# Analysis result
# =============================================================================


@dataclass
class FunctionAnalysis:
    """Result of analyzing a single function."""

    name: str
    input_types: dict[str, FrameType]
    declared_return_type: Optional[FrameType]
    inferred_return_type: Optional[FrameType]
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True if any errors were found during analysis."""
        return len(self.errors) > 0


def _extract_df_schema(annotation: ast.expr) -> Optional[str]:
    """Extract schema string from DF["{...}"] annotation."""
    # Handle DF["{...}"]
    if isinstance(annotation, ast.Subscript):
        # Check if it's DF[...]
        if isinstance(annotation.value, ast.Name) and annotation.value.id == "DF":
            # Extract the string inside
            if isinstance(annotation.slice, ast.Constant) and isinstance(
                annotation.slice.value, str
            ):
                return annotation.slice.value
    return None


def _parse_frame_type(schema_str: str) -> Optional[FrameType]:
    """Parse a schema string into FrameType."""
    try:
        return parse_schema(schema_str)
    except ParseError:
        return None


def _parse_frame_type_with_error(
    schema_str: str,
) -> tuple[Optional[FrameType], Optional[str]]:
    """Parse a schema string into FrameType, returning error message if failed."""
    try:
        return parse_schema(schema_str), None
    except ParseError as e:
        return None, str(e)


class ExpressionAnalyzer(ast.NodeVisitor):
    """Analyze expressions to infer their types and output column names."""

    def __init__(self, current_frame: FrameType):
        self.current_frame = current_frame
        self.errors: list[str] = []

    def analyze_agg_expr(self, node: ast.expr) -> Optional[AggExpr]:
        """Analyze an aggregation expression like pl.col("x").sum().alias("total")."""
        # Pattern: pl.col("col").agg_func().alias("name") or pl.col("col").agg_func()
        alias = None
        agg_node = node

        # Check for .alias("name") at the end
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "alias":
                if node.args and isinstance(node.args[0], ast.Constant):
                    alias = node.args[0].value
                    agg_node = node.func.value

        # Now look for the aggregation function call
        if isinstance(agg_node, ast.Call):
            if isinstance(agg_node.func, ast.Attribute):
                agg_func_name = agg_node.func.attr
                col_expr = agg_node.func.value

                # Map function names to AggFunction enum
                func_map = {
                    "sum": AggFunction.SUM,
                    "mean": AggFunction.MEAN,
                    "count": AggFunction.COUNT,
                    "n_unique": AggFunction.N_UNIQUE,
                    "list": AggFunction.LIST,
                    "first": AggFunction.FIRST,
                    "last": AggFunction.LAST,
                    "min": AggFunction.MIN,
                    "max": AggFunction.MAX,
                }

                if agg_func_name in func_map:
                    # Extract column name from pl.col("col")
                    col_name = self._extract_col_name(col_expr)
                    if col_name:
                        return AggExpr(
                            column=col_name,
                            function=func_map[agg_func_name],
                            alias=alias,
                        )

        return None

    def _extract_col_name(self, node: ast.expr) -> Optional[str]:
        """Extract column name from pl.col("name") expression."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "col":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "pl":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            return node.args[0].value
        return None

    def analyze_select_expr(self, node: ast.expr) -> tuple[Optional[str], Optional[DataType]]:
        """Analyze a select expression, return (output_name, type)."""
        # Check for .alias() wrapper
        alias = None
        inner_node = node

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "alias":
                if node.args and isinstance(node.args[0], ast.Constant):
                    alias = node.args[0].value
                    inner_node = node.func.value

        # Check for binary operations like pl.col("x") * 2
        if isinstance(inner_node, ast.BinOp):
            # For binary ops, try to infer the type from the left operand
            left_name, left_type = self.analyze_select_expr(inner_node.left)
            if left_type:
                output_name = alias if alias else left_name
                return output_name, left_type

        # Check for pl.col("name")
        col_name = self._extract_col_name(inner_node)
        if col_name:
            try:
                col_type = infer_col(col_name, self.current_frame)
                output_name = alias if alias else col_name
                return output_name, col_type
            except ColumnNotFoundError as e:
                self.errors.append(str(e))
                return None, None

        # Check for pl.lit(value)
        lit_type = self._extract_lit_type(inner_node)
        if lit_type:
            return alias, lit_type

        return None, None

    def _extract_lit_type(self, node: ast.expr) -> Optional[DataType]:
        """Extract type from pl.lit(value) expression."""
        from polypolarism.types import Int64, Float64, Utf8, Boolean, Null

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "lit":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "pl":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            value = node.args[0].value
                            if value is None:
                                return Null()
                            elif isinstance(value, bool):
                                return Boolean()
                            elif isinstance(value, int):
                                return Int64()
                            elif isinstance(value, float):
                                return Float64()
                            elif isinstance(value, str):
                                return Utf8()
        return None


class FunctionBodyAnalyzer(ast.NodeVisitor):
    """Analyze a function body to track DataFrame types."""

    def __init__(
        self,
        input_types: dict[str, FrameType],
        errors: list[str],
        registry: Optional[FunctionRegistry] = None,
    ):
        self.input_types = input_types
        self.errors = errors
        self.registry = registry or FunctionRegistry()
        # Track variable -> FrameType mapping
        self.var_types: dict[str, FrameType] = dict(input_types)
        self.return_type: Optional[FrameType] = None

    def visit_Return(self, node: ast.Return) -> None:
        """Handle return statements."""
        if node.value:
            self.return_type = self._infer_expr_type(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle variable assignments."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            inferred = self._infer_expr_type(node.value)
            if inferred:
                self.var_types[var_name] = inferred
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments like: df: DF["{...}"] = expr."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # Try to get type from annotation
            schema_str = _extract_df_schema(node.annotation)
            if schema_str:
                frame_type = _parse_frame_type(schema_str)
                if frame_type:
                    self.var_types[var_name] = frame_type
                    return
            # Fall back to inference from value
            if node.value:
                inferred = self._infer_expr_type(node.value)
                if inferred:
                    self.var_types[var_name] = inferred
        self.generic_visit(node)

    def _infer_expr_type(self, node: ast.expr) -> Optional[FrameType]:
        """Infer the FrameType of an expression."""
        # Variable reference
        if isinstance(node, ast.Name):
            return self.var_types.get(node.id)

        # Method call chain or function call
        if isinstance(node, ast.Call):
            return self._infer_call_type(node)

        return None

    def _infer_call_type(self, node: ast.Call) -> Optional[FrameType]:
        """Infer the type of a method or function call."""
        # Function call: func_name(args)
        if isinstance(node.func, ast.Name):
            return self._infer_function_call_type(node)

        # Method call: obj.method(args)
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            receiver = node.func.value

            # Handle .agg() call (comes after .group_by())
            if method_name == "agg":
                return self._infer_agg_call(receiver, node)

            # Handle other DataFrame methods
            receiver_type = self._infer_expr_type(receiver)
            if receiver_type:
                if method_name == "join":
                    return self._infer_join_call(receiver_type, node)
                elif method_name == "group_by":
                    # Return a marker or the receiver type
                    # The actual result comes from .agg()
                    return None  # Will be handled by .agg()
                elif method_name == "select":
                    return self._infer_select_call(receiver_type, node)
                elif method_name == "with_columns":
                    return self._infer_with_columns_call(receiver_type, node)

        return None

    def _infer_function_call_type(self, node: ast.Call) -> Optional[FrameType]:
        """Infer type of a function call like helper(df)."""
        if not isinstance(node.func, ast.Name):
            return None

        func_name = node.func.id
        func_info = self.registry.get(func_name)

        if func_info is None:
            # Unknown function - cannot infer
            return None

        # Infer argument types
        arg_types: list[Optional[FrameType]] = []
        for arg in node.args:
            arg_type = self._infer_expr_type(arg)
            arg_types.append(arg_type)

        # If function has a signature, use declared return type and check args
        if func_info.signature is not None:
            sig = func_info.signature
            # Check argument types against parameters
            for idx, arg_type in enumerate(arg_types):
                if arg_type is None:
                    continue
                param_info = sig.get_param_by_position(idx)
                if param_info is None:
                    continue
                param_name, expected_type = param_info
                if not _is_frame_subtype(arg_type, expected_type):
                    # Generate detailed error
                    for col_name, expected_col_type in expected_type.columns.items():
                        if col_name not in arg_type.columns:
                            self.errors.append(
                                f"Argument '{param_name}' is missing column '{col_name}'"
                            )
                        elif not _is_column_subtype(
                            arg_type.columns[col_name], expected_col_type
                        ):
                            self.errors.append(
                                f"Argument '{param_name}' column '{col_name}' has type "
                                f"{arg_type.columns[col_name]} but expected {expected_col_type}"
                            )
            return sig.return_type

        # Untyped function - analyze body with propagated argument types
        return self._analyze_untyped_function(func_info, arg_types)

    def _analyze_untyped_function(
        self, func_info: FunctionInfo, arg_types: list[Optional[FrameType]]
    ) -> Optional[FrameType]:
        """Analyze an untyped function body with propagated argument types."""
        # Create cache key from argument types
        cache_key = tuple(
            tuple(sorted(t.columns.items())) if t else None for t in arg_types
        )
        if cache_key in func_info.inferred_returns:
            return func_info.inferred_returns[cache_key]

        # Build input types from function parameters and provided arg types
        input_types: dict[str, FrameType] = {}
        func_node = func_info.node
        for idx, arg in enumerate(func_node.args.args):
            if idx < len(arg_types) and arg_types[idx] is not None:
                input_types[arg.arg] = arg_types[idx]

        # Analyze the function body
        errors: list[str] = []
        body_analyzer = FunctionBodyAnalyzer(input_types, errors, self.registry)
        for stmt in func_node.body:
            body_analyzer.visit(stmt)

        # Cache and return the result
        result = body_analyzer.return_type
        if result is not None:
            func_info.inferred_returns[cache_key] = result
        return result

    def _infer_join_call(
        self, left_type: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        """Infer type of .join() call."""
        # Extract right frame
        if not node.args:
            return None
        right_expr = node.args[0]
        right_type = self._infer_expr_type(right_expr)
        if not right_type:
            return None

        # Extract keyword arguments
        on = None
        left_on = None
        right_on = None
        how = "inner"

        for kw in node.keywords:
            if kw.arg == "on" and isinstance(kw.value, ast.Constant):
                on = kw.value.value
            elif kw.arg == "left_on" and isinstance(kw.value, ast.Constant):
                left_on = kw.value.value
            elif kw.arg == "right_on" and isinstance(kw.value, ast.Constant):
                right_on = kw.value.value
            elif kw.arg == "how" and isinstance(kw.value, ast.Constant):
                how = kw.value.value

        try:
            return infer_join(
                left_type, right_type, on=on, left_on=left_on, right_on=right_on, how=how
            )
        except JoinError as e:
            self.errors.append(str(e))
            return None

    def _infer_agg_call(
        self, groupby_receiver: ast.expr, node: ast.Call
    ) -> Optional[FrameType]:
        """Infer type of .group_by(...).agg(...) call."""
        # groupby_receiver should be a Call to .group_by()
        if not isinstance(groupby_receiver, ast.Call):
            return None
        if not isinstance(groupby_receiver.func, ast.Attribute):
            return None
        if groupby_receiver.func.attr != "group_by":
            return None

        # Get the DataFrame being grouped
        df_expr = groupby_receiver.func.value
        input_frame = self._infer_expr_type(df_expr)
        if not input_frame:
            return None

        # Extract group keys
        keys: list[str] = []
        for arg in groupby_receiver.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                keys.append(arg.value)

        # Extract aggregation expressions
        expr_analyzer = ExpressionAnalyzer(input_frame)
        agg_exprs: list[AggExpr] = []
        for arg in node.args:
            agg_expr = expr_analyzer.analyze_agg_expr(arg)
            if agg_expr:
                agg_exprs.append(agg_expr)

        self.errors.extend(expr_analyzer.errors)

        try:
            return infer_groupby_result(input_frame, keys, agg_exprs)
        except GroupByTypeError as e:
            self.errors.append(str(e))
            return None

    def _infer_select_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        """Infer type of .select() call."""
        expr_analyzer = ExpressionAnalyzer(input_frame)
        result_columns: dict[str, DataType] = {}

        for arg in node.args:
            name, dtype = expr_analyzer.analyze_select_expr(arg)
            if name and dtype:
                result_columns[name] = dtype

        self.errors.extend(expr_analyzer.errors)

        if result_columns:
            return FrameType(columns=result_columns)
        return None

    def _infer_with_columns_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        """Infer type of .with_columns() call."""
        # Start with all existing columns
        result_columns = dict(input_frame.columns)

        expr_analyzer = ExpressionAnalyzer(input_frame)

        for arg in node.args:
            name, dtype = expr_analyzer.analyze_select_expr(arg)
            if name and dtype:
                result_columns[name] = dtype

        self.errors.extend(expr_analyzer.errors)

        return FrameType(columns=result_columns)


def _extract_function_signature(func_node: ast.FunctionDef) -> Optional[FunctionSignature]:
    """Extract type signature from a function definition."""
    parameters: dict[str, tuple[int, FrameType]] = {}
    return_type: Optional[FrameType] = None

    # Extract parameter types
    for idx, arg in enumerate(func_node.args.args):
        if arg.annotation:
            schema_str = _extract_df_schema(arg.annotation)
            if schema_str:
                frame_type = _parse_frame_type(schema_str)
                if frame_type:
                    parameters[arg.arg] = (idx, frame_type)

    # Extract return type
    if func_node.returns:
        schema_str = _extract_df_schema(func_node.returns)
        if schema_str:
            return_type = _parse_frame_type(schema_str)

    # Return None if no DF annotations found
    if not parameters and return_type is None:
        return None

    return FunctionSignature(
        name=func_node.name,
        parameters=parameters,
        return_type=return_type,
        lineno=func_node.lineno,
    )


def analyze_function(
    func_node: ast.FunctionDef,
    registry: Optional[FunctionRegistry] = None,
) -> Optional[FunctionAnalysis]:
    """Analyze a single function definition."""
    input_types: dict[str, FrameType] = {}
    declared_return: Optional[FrameType] = None
    errors: list[str] = []
    has_df_annotation = False

    # Extract input parameter types
    for arg in func_node.args.args:
        if arg.annotation:
            schema_str = _extract_df_schema(arg.annotation)
            if schema_str:
                has_df_annotation = True
                frame_type, parse_error = _parse_frame_type_with_error(schema_str)
                if frame_type:
                    input_types[arg.arg] = frame_type
                elif parse_error:
                    errors.append(f"Parameter '{arg.arg}': {parse_error}")

    # Extract return type
    if func_node.returns:
        schema_str = _extract_df_schema(func_node.returns)
        if schema_str:
            has_df_annotation = True
            declared_return, parse_error = _parse_frame_type_with_error(schema_str)
            if parse_error:
                errors.append(f"Return type: {parse_error}")

    # If no DF annotations found, skip this function
    if not has_df_annotation:
        return None

    # Analyze function body with registry
    body_analyzer = FunctionBodyAnalyzer(input_types, errors, registry)
    for stmt in func_node.body:
        body_analyzer.visit(stmt)

    return FunctionAnalysis(
        name=func_node.name,
        input_types=input_types,
        declared_return_type=declared_return,
        inferred_return_type=body_analyzer.return_type,
        errors=body_analyzer.errors,
    )


def analyze_source(source: str) -> list[FunctionAnalysis]:
    """
    Analyze Python source code for DataFrame type annotations.

    Uses a 3-pass approach:
    1. Collect all function AST nodes
    2. Build registry with signatures (typed) and nodes (all)
    3. Analyze function bodies with registry for call resolution

    Args:
        source: Python source code as a string

    Returns:
        List of FunctionAnalysis results for functions with DF annotations
    """
    tree = ast.parse(source)

    # Pass 1: Collect all function nodes
    func_nodes: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_nodes.append(node)

    # Pass 2: Build registry with all functions
    registry = FunctionRegistry()
    for func_node in func_nodes:
        signature = _extract_function_signature(func_node)
        info = FunctionInfo(
            name=func_node.name,
            node=func_node,
            signature=signature,
            inferred_returns={},
        )
        registry.register(info)

    # Pass 3: Analyze functions with registry
    results: list[FunctionAnalysis] = []
    for func_node in func_nodes:
        analysis = analyze_function(func_node, registry)
        if analysis:
            results.append(analysis)

    return results
