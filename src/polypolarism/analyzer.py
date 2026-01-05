"""AST analysis and data flow tracking."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

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

        return None, None


class FunctionBodyAnalyzer(ast.NodeVisitor):
    """Analyze a function body to track DataFrame types."""

    def __init__(
        self, input_types: dict[str, FrameType], errors: list[str]
    ):
        self.input_types = input_types
        self.errors = errors
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

    def _infer_expr_type(self, node: ast.expr) -> Optional[FrameType]:
        """Infer the FrameType of an expression."""
        # Variable reference
        if isinstance(node, ast.Name):
            return self.var_types.get(node.id)

        # Method call chain
        if isinstance(node, ast.Call):
            return self._infer_call_type(node)

        return None

    def _infer_call_type(self, node: ast.Call) -> Optional[FrameType]:
        """Infer the type of a method call."""
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


def analyze_function(func_node: ast.FunctionDef) -> Optional[FunctionAnalysis]:
    """Analyze a single function definition."""
    input_types: dict[str, FrameType] = {}
    declared_return: Optional[FrameType] = None
    errors: list[str] = []

    # Extract input parameter types
    for arg in func_node.args.args:
        if arg.annotation:
            schema_str = _extract_df_schema(arg.annotation)
            if schema_str:
                frame_type = _parse_frame_type(schema_str)
                if frame_type:
                    input_types[arg.arg] = frame_type

    # Extract return type
    if func_node.returns:
        schema_str = _extract_df_schema(func_node.returns)
        if schema_str:
            declared_return = _parse_frame_type(schema_str)

    # If no DF annotations found, skip this function
    if not input_types and not declared_return:
        return None

    # Analyze function body
    body_analyzer = FunctionBodyAnalyzer(input_types, errors)
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

    Args:
        source: Python source code as a string

    Returns:
        List of FunctionAnalysis results for functions with DF annotations
    """
    tree = ast.parse(source)
    results: list[FunctionAnalysis] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            analysis = analyze_function(node)
            if analysis:
                results.append(analysis)

    return results
