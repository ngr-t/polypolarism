"""AST analysis and data flow tracking."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from polypolarism.types import (
    DataType,
    FrameType,
    Nullable,
    ColumnSpec,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Utf8,
    Boolean,
    Date,
    Datetime,
    Duration,
    List as ListT,
)
from polypolarism.pandera_annotation import extract_dataframe_annotation
from polypolarism.pandera_schema import SchemaRegistry, collect_schemas
from polypolarism.ops.join import infer_join, JoinError
from polypolarism.ops.groupby import (
    infer_groupby_result,
    infer_agg_result_type,
    AggExpr,
    AggFunction,
    GroupByTypeError,
)
from polypolarism.ops.reshape import (
    ReshapeError,
    concat_vertical,
    concat_horizontal,
    concat_diagonal,
    unpivot as infer_unpivot,
)
from polypolarism.expr_infer import infer_col, ColumnNotFoundError


class AnalysisError(Exception):
    """Error during analysis."""

    pass


# Methods whose return frame has the same schema as the receiver.
# `lazy()` and `collect()` cross between DataFrame/LazyFrame, but in our
# static type system both are identity. `set_sorted` mutates only metadata.
_IDENTITY_FRAME_METHODS: frozenset[str] = frozenset({
    "sort",
    "head",
    "tail",
    "limit",
    "slice",
    "reverse",
    "sample",
    "unique",
    "clone",
    "lazy",
    "set_sorted",
    "shrink_to_fit",
    "rechunk",
})


# Map ``pl.<Name>`` attribute references and call expressions to our DataType.
# Datetime/Duration with parameters are handled via Call form.
_PL_DTYPE_NAME_MAP: dict[str, DataType] = {
    "Int8": Int8(),
    "Int16": Int16(),
    "Int32": Int32(),
    "Int64": Int64(),
    "UInt8": UInt8(),
    "UInt16": UInt16(),
    "UInt32": UInt32(),
    "UInt64": UInt64(),
    "Float32": Float32(),
    "Float64": Float64(),
    "Utf8": Utf8(),
    "String": Utf8(),  # polars 1.x alias
    "Boolean": Boolean(),
    "Date": Date(),
    "Datetime": Datetime(),
    "Duration": Duration(),
}


def _resolve_pl_dtype(node: ast.expr) -> Optional[DataType]:
    """Resolve an AST node referring to a Polars dtype literal (``pl.Int32`` etc.)."""
    # Bare ``pl.Int32``
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "pl":
            return _PL_DTYPE_NAME_MAP.get(node.attr)
    # Parametric form like ``pl.Datetime("us", "UTC")`` — keep simple.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "pl":
            base = _PL_DTYPE_NAME_MAP.get(node.func.attr)
            if isinstance(base, Datetime):
                tz: Optional[str] = None
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    if isinstance(node.args[1].value, str):
                        tz = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "time_zone" and isinstance(kw.value, ast.Constant):
                        if isinstance(kw.value.value, str):
                            tz = kw.value.value
                return Datetime(tz=tz)
            return base
    return None


def _wrap_like(receiver: DataType, new_inner: DataType) -> DataType:
    """Preserve the receiver's outer ``Nullable`` wrapper around a new inner dtype."""
    if isinstance(receiver, Nullable):
        return Nullable(new_inner)
    return new_inner


def _str_constant(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _str_list_or_tuple(node: ast.expr) -> Optional[list[str]]:
    if isinstance(node, (ast.List, ast.Tuple)):
        out: list[str] = []
        for elt in node.elts:
            s = _str_constant(elt)
            if s is None:
                return None
            out.append(s)
        return out
    return None


# =============================================================================
# Data structures for function registry
# =============================================================================


@dataclass
class FunctionSignature:
    """Type signature for a function with ``DataFrame[Schema]`` annotations."""

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
    - actual must contain every required column expected has
    - For columns present on both sides, the actual dtype must be a subtype
      and an actual optional column cannot satisfy a required expected column
    - actual may have extra columns unless ``expected.strict`` is True
    """
    for col_name, expected_spec in expected.columns.items():
        actual_spec = actual.columns.get(col_name)
        if actual_spec is None:
            if expected_spec.required:
                return False
            continue
        if expected_spec.required and not actual_spec.required:
            return False
        if not _is_column_subtype(actual_spec.dtype, expected_spec.dtype):
            return False
    if expected.strict:
        for col_name in actual.columns:
            if col_name not in expected.columns:
                return False
    return True


# =============================================================================
# Analysis result
# =============================================================================


@dataclass
class FunctionAnalysis:
    """Result of analyzing a single function."""

    name: str
    lineno: int  # Line number of function definition (1-indexed)
    end_lineno: int  # End line number of function definition (1-indexed)
    input_types: dict[str, FrameType]
    declared_return_type: Optional[FrameType]
    inferred_return_type: Optional[FrameType]
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True if any errors were found during analysis."""
        return len(self.errors) > 0


def _resolve_declared_type(
    annotation: ast.expr,
    schema_registry: SchemaRegistry,
) -> tuple[Optional[FrameType], Optional[str]]:
    """Resolve a declared FrameType from a Pandera ``DataFrame[Schema]`` annotation.

    Returns ``(frame_type, error)``. Both are ``None`` when the annotation
    doesn't declare a Pandera-backed frame type. ``error`` is reserved for
    future schema-resolution errors; currently always ``None``.
    """
    pandera_ft = extract_dataframe_annotation(annotation, schema_registry)
    if pandera_ft is not None:
        return pandera_ft, None
    return None, None


def _annotation_declares_frame(
    annotation: ast.expr,
    schema_registry: SchemaRegistry,
) -> bool:
    """Return True if the annotation is ``DataFrame[Schema]`` / ``LazyFrame[Schema]``."""
    return extract_dataframe_annotation(annotation, schema_registry) is not None


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
                    "std": AggFunction.STD,
                    "var": AggFunction.VAR,
                    "median": AggFunction.MEDIAN,
                    "quantile": AggFunction.QUANTILE,
                    "product": AggFunction.PRODUCT,
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

    # Methods on a column expression that always produce Boolean.
    _BOOLEAN_PREDICATE_METHODS = frozenset({
        "is_null",
        "is_not_null",
        "is_nan",
        "is_not_nan",
        "is_finite",
        "is_infinite",
        "is_unique",
        "is_duplicated",
        "is_first_distinct",
        "is_last_distinct",
        "is_in",
        "is_between",
        "has_nulls",
        "not_",
    })

    # Methods that return Float64 from any numeric receiver.
    _FLOAT_RETURN_METHODS = frozenset({
        "log",
        "log10",
        "log1p",
        "exp",
        "sqrt",
        "cbrt",
        "entropy",
    })

    # Methods that preserve the receiver's dtype (numeric mostly).
    _DTYPE_PRESERVING_METHODS = frozenset({
        "abs",
        "round",
        "clip",
        "floor",
        "ceil",
        "sign",
        "neg",
        "shrink_dtype",
        "rechunk",
    })

    # ---- sub-namespace return type tables ----------------------------------

    # ``pl.col("x").str.<method>(...)``. Values are either a fixed DataType or
    # a callable that takes the receiver dtype.
    _STR_RETURN: dict[str, DataType] = {
        # Boolean predicates
        "contains": Boolean(),
        "contains_any": Boolean(),
        "starts_with": Boolean(),
        "ends_with": Boolean(),
        "is_empty": Boolean(),
        # Utf8-returning transformations
        "lower": Utf8(),
        "upper": Utf8(),
        "to_lowercase": Utf8(),
        "to_uppercase": Utf8(),
        "to_titlecase": Utf8(),
        "strip": Utf8(),
        "strip_chars": Utf8(),
        "strip_chars_start": Utf8(),
        "strip_chars_end": Utf8(),
        "lstrip": Utf8(),
        "rstrip": Utf8(),
        "replace": Utf8(),
        "replace_all": Utf8(),
        "replace_many": Utf8(),
        "pad_start": Utf8(),
        "pad_end": Utf8(),
        "zfill": Utf8(),
        "slice": Utf8(),
        "head": Utf8(),
        "tail": Utf8(),
        "reverse": Utf8(),
        "concat": Utf8(),
        "join": Utf8(),
        # Length / counts
        "len_chars": UInt32(),
        "len_bytes": UInt32(),
        "count_matches": UInt32(),
        # Splitting
        "split": ListT(Utf8()),
        # Parsing
        "to_date": Date(),
        "to_datetime": Datetime(),
    }

    # ``pl.col("ts").dt.<method>()``. Datetime → various integer parts; some
    # methods preserve the receiver dtype (truncate / round / offset_by /
    # replace_time_zone / convert_time_zone).
    _DT_RETURN: dict[str, DataType] = {
        "year": Int32(),
        "iso_year": Int32(),
        "month": Int8(),
        "day": Int8(),
        "hour": Int8(),
        "minute": Int8(),
        "second": Int8(),
        "millisecond": Int32(),
        "microsecond": Int32(),
        "nanosecond": Int32(),
        "weekday": Int8(),
        "quarter": Int8(),
        "week": Int8(),
        "ordinal_day": Int16(),
        "date": Date(),
        "epoch": Int64(),
        "timestamp": Int64(),
        "total_days": Int64(),
        "total_hours": Int64(),
        "total_minutes": Int64(),
        "total_seconds": Int64(),
        "total_milliseconds": Int64(),
        "total_microseconds": Int64(),
        "total_nanoseconds": Int64(),
    }

    # Methods on ``pl.col("ts").dt`` that preserve the receiver dtype.
    _DT_PRESERVING: frozenset[str] = frozenset({
        "truncate",
        "round",
        "offset_by",
        "replace_time_zone",
        "convert_time_zone",
        "month_start",
        "month_end",
    })

    # Methods on ``pl.col("xs").list`` that preserve the receiver dtype.
    _LIST_PRESERVING: frozenset[str] = frozenset({
        "unique",
        "sort",
        "reverse",
        "head",
        "tail",
        "slice",
        "drop_nulls",
        "sample",
        "shift",
    })

    # Methods on ``pl.col("xs").list`` that return the element dtype.
    _LIST_ELEMENT_RETURN: frozenset[str] = frozenset({
        "get",
        "first",
        "last",
        "sum",
        "mean",
        "min",
        "max",
        "median",
    })

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

        # Comparison expressions (==, !=, <, <=, >, >=) -> Boolean
        if isinstance(inner_node, ast.Compare):
            self._validate_subexpr(inner_node.left)
            for cmp in inner_node.comparators:
                self._validate_subexpr(cmp)
            return alias, Boolean()

        # Logical operators expressed as bitwise: a & b, a | b, a ^ b -> Boolean
        if isinstance(inner_node, ast.BinOp) and isinstance(
            inner_node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            self._validate_subexpr(inner_node.left)
            self._validate_subexpr(inner_node.right)
            return alias, Boolean()

        # Logical NOT: ~expr or `not expr` -> Boolean (when receiver is boolean-like)
        if isinstance(inner_node, ast.UnaryOp) and isinstance(
            inner_node.op, (ast.Invert, ast.Not)
        ):
            self._validate_subexpr(inner_node.operand)
            return alias, Boolean()

        # Arithmetic binary operations like pl.col("x") * 2
        if isinstance(inner_node, ast.BinOp):
            # For binary ops, try to infer the type from the left operand
            left_name, left_type = self.analyze_select_expr(inner_node.left)
            if left_type:
                output_name = alias if alias else left_name
                return output_name, left_type

        # Method-chain on a column expression (is_null, fill_null, std, abs, ...)
        chain_result = self._analyze_method_chain(inner_node)
        if chain_result is not None:
            chain_name, chain_type = chain_result
            output_name = alias if alias else chain_name
            return output_name, chain_type

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

    def _dispatch_namespace_method(
        self,
        namespace: str,
        method: str,
        receiver_type: Optional[DataType],
    ) -> Optional[DataType]:
        """Resolve ``<col_expr>.<namespace>.<method>(...)`` to a DataType.

        ``receiver_type`` is the dtype of the column the namespace was attached
        to (``None`` if it couldn't be resolved). The receiver's nullability
        is preserved on the result for almost all of these methods.
        """
        receiver_inner = receiver_type
        receiver_is_nullable = False
        if isinstance(receiver_type, Nullable):
            receiver_inner = receiver_type.inner
            receiver_is_nullable = True

        result: Optional[DataType] = None

        if namespace == "str":
            result = self._STR_RETURN.get(method)
        elif namespace == "dt":
            if method in self._DT_RETURN:
                result = self._DT_RETURN[method]
            elif method in self._DT_PRESERVING and receiver_inner is not None:
                result = receiver_inner
        elif namespace in ("list", "arr"):
            if method == "len":
                result = UInt32()
            elif method in self._LIST_PRESERVING and receiver_inner is not None:
                result = receiver_inner
            elif method in self._LIST_ELEMENT_RETURN and isinstance(
                receiver_inner, ListT
            ):
                result = receiver_inner.inner

        if result is None:
            return None
        if receiver_is_nullable and not isinstance(result, Nullable):
            return Nullable(result)
        return result

    def _validate_subexpr(self, node: ast.expr) -> None:
        """Run a sub-expression through analyze_select_expr to surface column errors.

        We discard the type/name; the only side-effect of interest is appending
        to ``self.errors`` when ``pl.col("missing")`` shows up.
        """
        self.analyze_select_expr(node)

    def _analyze_method_chain(
        self, node: ast.expr
    ) -> Optional[tuple[Optional[str], DataType]]:
        """Analyze ``pl.col("x").<method>(...)`` style chains.

        Returns ``(default_name, dtype)`` or ``None`` if the node isn't a
        recognised method chain.
        """
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, ast.Attribute):
            return None
        method = node.func.attr
        receiver = node.func.value

        # Sub-namespace: ``pl.col("x").str.contains(...)``,
        # ``pl.col("ts").dt.year()``, ``pl.col("xs").list.get(0)`` etc.
        if isinstance(receiver, ast.Attribute) and receiver.attr in (
            "str",
            "dt",
            "list",
            "arr",
        ):
            ns = receiver.attr
            col_name, col_type = self.analyze_select_expr(receiver.value)
            ns_result = self._dispatch_namespace_method(ns, method, col_type)
            if ns_result is None:
                return None
            return col_name, ns_result

        receiver_result = self.analyze_select_expr(receiver)
        receiver_name, receiver_type = receiver_result
        if receiver_type is None and receiver_name is None:
            # Receiver wasn't recognised — bail out.
            return None

        # Boolean predicates always produce Boolean; column name carried through.
        if method in self._BOOLEAN_PREDICATE_METHODS:
            return receiver_name, Boolean()

        # fill_null / fill_nan strip the Nullable wrapper.
        if method in ("fill_null", "fill_nan"):
            inner_dtype = receiver_type
            if isinstance(receiver_type, Nullable):
                inner_dtype = receiver_type.inner
            return receiver_name, inner_dtype if inner_dtype is not None else Boolean()

        # Float-returning numeric methods.
        if method in self._FLOAT_RETURN_METHODS:
            base = (
                receiver_type.inner
                if isinstance(receiver_type, Nullable)
                else receiver_type
            )
            result: DataType = Float64()
            if isinstance(receiver_type, Nullable):
                result = Nullable(Float64())
            return receiver_name, result

        # Dtype-preserving methods.
        if method in self._DTYPE_PRESERVING_METHODS and receiver_type is not None:
            return receiver_name, receiver_type

        # Aggregation-style methods used outside of group_by — return reduction dtype.
        # Reuses the same map as analyze_agg_expr.
        agg_map: dict[str, AggFunction] = {
            "sum": AggFunction.SUM,
            "mean": AggFunction.MEAN,
            "count": AggFunction.COUNT,
            "n_unique": AggFunction.N_UNIQUE,
            "first": AggFunction.FIRST,
            "last": AggFunction.LAST,
            "min": AggFunction.MIN,
            "max": AggFunction.MAX,
            "std": AggFunction.STD,
            "var": AggFunction.VAR,
            "median": AggFunction.MEDIAN,
            "quantile": AggFunction.QUANTILE,
            "product": AggFunction.PRODUCT,
        }
        if method in agg_map and receiver_type is not None:
            try:
                return receiver_name, infer_agg_result_type(
                    agg_map[method], receiver_type
                )
            except GroupByTypeError as e:
                self.errors.append(str(e))
                return receiver_name, receiver_type

        # ``cast(pl.<dtype>)`` chained directly on column.
        if method == "cast" and node.args:
            target = _resolve_pl_dtype(node.args[0])
            if target is not None and receiver_type is not None:
                return receiver_name, _wrap_like(receiver_type, target)

        return None

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
        schema_registry: Optional[SchemaRegistry] = None,
    ):
        self.input_types = input_types
        self.errors = errors
        self.registry = registry or FunctionRegistry()
        self.schema_registry = schema_registry or SchemaRegistry()
        # Track variable -> FrameType mapping
        self.var_types: dict[str, FrameType] = dict(input_types)
        self.return_type: Optional[FrameType] = None
        # Bare ``Schema.validate(df)`` narrowing only fires at the function's
        # top level. We toggle this off when descending into if/for/while/try.
        self._narrowing_enabled = True

    def _visit_with_narrowing_disabled(self, node: ast.AST) -> None:
        prev = self._narrowing_enabled
        self._narrowing_enabled = False
        try:
            self.generic_visit(node)
        finally:
            self._narrowing_enabled = prev

    def visit_If(self, node: ast.If) -> None:
        self._visit_with_narrowing_disabled(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_with_narrowing_disabled(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_with_narrowing_disabled(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_with_narrowing_disabled(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_with_narrowing_disabled(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with_narrowing_disabled(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with_narrowing_disabled(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Handle return statements."""
        if node.value:
            self.return_type = self._infer_expr_type(node.value)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Bare expression statement; recognise ``Schema.validate(df)`` as narrowing.

        Narrowing only fires at the function body's top level; visits inside
        if/for/while/try/with disable it via the ``_narrowing_enabled`` flag.
        """
        if not self._narrowing_enabled:
            return
        if not isinstance(node.value, ast.Call):
            return
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            return
        if call.func.attr != "validate":
            return
        schema_node = call.func.value
        if not isinstance(schema_node, ast.Name):
            return
        schema_ft = self.schema_registry.to_frame_type(schema_node.id)
        if schema_ft is None or not call.args:
            return
        arg = call.args[0]
        if isinstance(arg, ast.Name) and arg.id in self.var_types:
            self.var_types[arg.id] = schema_ft

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle variable assignments."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            inferred = self._infer_expr_type(node.value)
            if inferred:
                self.var_types[var_name] = inferred
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments like: df: DataFrame[Schema] = expr."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # Try to get type from a Pandera DataFrame[Schema] annotation
            frame_type, _ = _resolve_declared_type(node.annotation, self.schema_registry)
            if frame_type is not None:
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

            # ``pl.concat([...], how=...)`` — top-level pl function, not a frame method.
            if (
                isinstance(receiver, ast.Name)
                and receiver.id == "pl"
                and method_name == "concat"
            ):
                return self._infer_concat_call(node)

            # Pandera narrowing: Schema.validate(df) -> Schema's FrameType
            if method_name == "validate":
                schema_ft = self._infer_validate_call(node)
                if schema_ft is not None:
                    return schema_ft

            # df.pipe(Schema.validate) -> Schema's FrameType; otherwise pipe is identity-typed
            if method_name == "pipe":
                piped = self._infer_pipe_call(node)
                if piped is not None:
                    return piped
                return self._infer_expr_type(receiver)

            # LazyFrame.collect() is identity for our static type system
            if method_name == "collect":
                return self._infer_expr_type(receiver)

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
                elif method_name == "drop":
                    return self._infer_drop_call(receiver_type, node)
                elif method_name == "rename":
                    return self._infer_rename_call(receiver_type, node)
                elif method_name == "cast":
                    return self._infer_cast_call(receiver_type, node)
                elif method_name == "drop_nulls":
                    return self._infer_drop_nulls_call(receiver_type, node)
                elif method_name == "with_row_index":
                    return self._infer_with_row_index_call(receiver_type, node)
                elif method_name == "filter":
                    return self._infer_filter_call(receiver_type, node)
                elif method_name == "explode":
                    return self._infer_explode_call(receiver_type, node)
                elif method_name == "vstack":
                    return self._infer_vstack_call(receiver_type, node)
                elif method_name in ("hstack", "extend"):
                    return self._infer_hstack_call(receiver_type, node)
                elif method_name in ("unpivot", "melt"):
                    return self._infer_unpivot_call(receiver_type, node)
                elif method_name in _IDENTITY_FRAME_METHODS:
                    return receiver_type

        return None

    def _infer_validate_call(self, node: ast.Call) -> Optional[FrameType]:
        """Resolve ``Schema.validate(df)`` to the schema's FrameType."""
        if not isinstance(node.func, ast.Attribute):
            return None
        schema_node = node.func.value
        if not isinstance(schema_node, ast.Name):
            return None
        return self.schema_registry.to_frame_type(schema_node.id)

    def _infer_pipe_call(self, node: ast.Call) -> Optional[FrameType]:
        """Resolve ``df.pipe(Schema.validate)`` to the schema's FrameType."""
        if not node.args:
            return None
        arg = node.args[0]
        if isinstance(arg, ast.Attribute) and arg.attr == "validate":
            if isinstance(arg.value, ast.Name):
                return self.schema_registry.to_frame_type(arg.value.id)
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
                    for col_name, expected_col_spec in expected_type.columns.items():
                        if col_name not in arg_type.columns:
                            self.errors.append(
                                f"Argument '{param_name}' is missing column '{col_name}'"
                            )
                        else:
                            actual_col_dtype = arg_type.columns[col_name].dtype
                            expected_col_dtype = expected_col_spec.dtype
                            if not _is_column_subtype(actual_col_dtype, expected_col_dtype):
                                self.errors.append(
                                    f"Argument '{param_name}' column '{col_name}' has type "
                                    f"{actual_col_dtype} but expected {expected_col_dtype}"
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
        body_analyzer = FunctionBodyAnalyzer(
            input_types, errors, self.registry, self.schema_registry
        )
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

    # -- M1 frame methods --------------------------------------------------

    def _collect_drop_targets(self, node: ast.Call) -> list[str]:
        """Resolve column-name args for ``drop`` / ``drop_nulls(subset=...)``.

        Supports both ``drop("a", "b")`` and ``drop(["a", "b"])``.
        Returns column names in argument order (subset list flattened in).
        """
        names: list[str] = []
        for arg in node.args:
            s = _str_constant(arg)
            if s is not None:
                names.append(s)
                continue
            lst = _str_list_or_tuple(arg)
            if lst is not None:
                names.extend(lst)
        return names

    def _infer_drop_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        targets = self._collect_drop_targets(node)
        result_columns = dict(input_frame.columns)
        for name in targets:
            if name not in result_columns:
                self.errors.append(f"drop: column '{name}' not found")
                continue
            del result_columns[name]
        return FrameType(
            columns=result_columns, strict=input_frame.strict, rest=input_frame.rest
        )

    def _infer_rename_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        if not node.args or not isinstance(node.args[0], ast.Dict):
            return input_frame
        mapping_node = node.args[0]
        mapping: dict[str, str] = {}
        for key_node, val_node in zip(mapping_node.keys, mapping_node.values):
            if key_node is None:
                continue
            old = _str_constant(key_node)
            new = _str_constant(val_node)
            if old is None or new is None:
                continue
            mapping[old] = new

        result_columns: dict[str, ColumnSpec] = {}
        for col_name, spec in input_frame.columns.items():
            new_name = mapping.get(col_name, col_name)
            result_columns[new_name] = spec
        for old in mapping:
            if old not in input_frame.columns:
                self.errors.append(f"rename: column '{old}' not found")
        return FrameType(
            columns=result_columns, strict=input_frame.strict, rest=input_frame.rest
        )

    def _infer_cast_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        if not node.args:
            return input_frame
        first = node.args[0]
        if not isinstance(first, ast.Dict):
            # ``cast(pl.Int64)`` whole-frame form not handled in M1 — fall back to identity.
            return input_frame
        result_columns: dict[str, ColumnSpec] = dict(input_frame.columns)
        for key_node, val_node in zip(first.keys, first.values):
            if key_node is None:
                continue
            col = _str_constant(key_node)
            if col is None:
                continue
            target = _resolve_pl_dtype(val_node)
            if target is None:
                continue
            spec = result_columns.get(col)
            if spec is None:
                self.errors.append(f"cast: column '{col}' not found")
                continue
            result_columns[col] = ColumnSpec(
                dtype=_wrap_like(spec.dtype, target),
                required=spec.required,
            )
        return FrameType(
            columns=result_columns, strict=input_frame.strict, rest=input_frame.rest
        )

    def _infer_drop_nulls_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        # subset can be passed positionally or as keyword
        subset: Optional[list[str]] = None
        if node.args:
            cand = _str_list_or_tuple(node.args[0]) or (
                [_str_constant(node.args[0])]
                if _str_constant(node.args[0]) is not None
                else None
            )
            if cand is not None:
                subset = cand
        for kw in node.keywords:
            if kw.arg == "subset":
                cand2 = _str_list_or_tuple(kw.value) or (
                    [_str_constant(kw.value)]
                    if _str_constant(kw.value) is not None
                    else None
                )
                if cand2 is not None:
                    subset = cand2

        targets = subset if subset is not None else list(input_frame.columns.keys())
        result_columns: dict[str, ColumnSpec] = {}
        for col_name, spec in input_frame.columns.items():
            if col_name in targets:
                if col_name not in input_frame.columns and subset is not None:
                    self.errors.append(
                        f"drop_nulls: column '{col_name}' not found"
                    )
                inner = (
                    spec.dtype.inner if isinstance(spec.dtype, Nullable) else spec.dtype
                )
                result_columns[col_name] = ColumnSpec(dtype=inner, required=spec.required)
            else:
                result_columns[col_name] = spec
        if subset is not None:
            for s in subset:
                if s not in input_frame.columns:
                    self.errors.append(f"drop_nulls: column '{s}' not found")
        return FrameType(
            columns=result_columns, strict=input_frame.strict, rest=input_frame.rest
        )

    def _collect_concat_frames(
        self, list_node: ast.expr
    ) -> Optional[list[FrameType]]:
        """Resolve a list/tuple-of-frames argument used by ``pl.concat([...])``."""
        if not isinstance(list_node, (ast.List, ast.Tuple)):
            return None
        out: list[FrameType] = []
        for elt in list_node.elts:
            ft = self._infer_expr_type(elt)
            if ft is None:
                return None
            out.append(ft)
        return out

    def _infer_concat_call(self, node: ast.Call) -> Optional[FrameType]:
        if not node.args:
            return None
        frames = self._collect_concat_frames(node.args[0])
        if frames is None:
            return None
        how = "vertical"
        for kw in node.keywords:
            if kw.arg == "how" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    how = kw.value.value
        try:
            if how == "vertical":
                return concat_vertical(frames)
            if how == "horizontal":
                return concat_horizontal(frames)
            if how in ("diagonal", "diagonal_relaxed"):
                return concat_diagonal(frames)
        except ReshapeError as e:
            self.errors.append(str(e))
            return None
        # Unsupported how — treat as vertical with a warning.
        try:
            return concat_vertical(frames)
        except ReshapeError as e:
            self.errors.append(str(e))
            return None

    def _infer_vstack_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        if not node.args:
            return input_frame
        other = self._infer_expr_type(node.args[0])
        if other is None:
            return input_frame
        try:
            return concat_vertical([input_frame, other])
        except ReshapeError as e:
            self.errors.append(str(e))
            return None

    def _infer_hstack_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        if not node.args:
            return input_frame
        other = self._infer_expr_type(node.args[0])
        if other is None:
            return input_frame
        try:
            return concat_horizontal([input_frame, other])
        except ReshapeError as e:
            self.errors.append(str(e))
            return None

    def _infer_explode_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        targets: list[str] = []
        for arg in node.args:
            s = _str_constant(arg)
            if s is not None:
                targets.append(s)
                continue
            lst = _str_list_or_tuple(arg)
            if lst is not None:
                targets.extend(lst)
        if not targets:
            return input_frame

        result_columns: dict[str, ColumnSpec] = dict(input_frame.columns)
        for col in targets:
            spec = result_columns.get(col)
            if spec is None:
                self.errors.append(f"explode: column '{col}' not found")
                continue
            inner = spec.dtype
            outer_nullable = isinstance(inner, Nullable)
            if outer_nullable:
                inner = inner.inner  # type: ignore[union-attr]
            if not isinstance(inner, ListT):
                self.errors.append(
                    f"explode: column '{col}' is {spec.dtype}, not List[T]"
                )
                continue
            elem_dtype: DataType = inner.inner
            if outer_nullable:
                elem_dtype = Nullable(elem_dtype)
            result_columns[col] = ColumnSpec(dtype=elem_dtype, required=spec.required)
        return FrameType(
            columns=result_columns, strict=input_frame.strict, rest=input_frame.rest
        )

    def _infer_unpivot_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        index: list[str] = []
        on: list[str] = []
        variable_name = "variable"
        value_name = "value"
        for kw in node.keywords:
            if kw.arg == "index":
                lst = _str_list_or_tuple(kw.value)
                single = _str_constant(kw.value)
                if lst is not None:
                    index = lst
                elif single is not None:
                    index = [single]
            elif kw.arg == "on":
                lst = _str_list_or_tuple(kw.value)
                single = _str_constant(kw.value)
                if lst is not None:
                    on = lst
                elif single is not None:
                    on = [single]
            elif kw.arg == "variable_name":
                cand = _str_constant(kw.value)
                if cand is not None:
                    variable_name = cand
            elif kw.arg == "value_name":
                cand = _str_constant(kw.value)
                if cand is not None:
                    value_name = cand
        try:
            return infer_unpivot(
                input_frame,
                index=index,
                on=on,
                variable_name=variable_name,
                value_name=value_name,
            )
        except ReshapeError as e:
            self.errors.append(str(e))
            return None

    def _infer_filter_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        """Identity-typed, but walk every predicate sub-expression to validate columns."""
        expr_analyzer = ExpressionAnalyzer(input_frame)
        for arg in node.args:
            expr_analyzer.analyze_select_expr(arg)
        for kw in node.keywords:
            expr_analyzer.analyze_select_expr(kw.value)
        self.errors.extend(expr_analyzer.errors)
        return input_frame

    def _infer_with_row_index_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> Optional[FrameType]:
        name = "index"
        if node.args:
            cand = _str_constant(node.args[0])
            if cand is not None:
                name = cand
        for kw in node.keywords:
            if kw.arg == "name":
                cand2 = _str_constant(kw.value)
                if cand2 is not None:
                    name = cand2

        result_columns: dict[str, ColumnSpec] = {name: ColumnSpec(dtype=UInt32())}
        for col_name, spec in input_frame.columns.items():
            if col_name == name:
                self.errors.append(
                    f"with_row_index: column '{name}' already exists"
                )
                continue
            result_columns[col_name] = spec
        return FrameType(
            columns=result_columns, strict=input_frame.strict, rest=input_frame.rest
        )


def _extract_function_signature(
    func_node: ast.FunctionDef,
    schema_registry: SchemaRegistry,
) -> Optional[FunctionSignature]:
    """Extract type signature from a function definition."""
    parameters: dict[str, tuple[int, FrameType]] = {}
    return_type: Optional[FrameType] = None

    # Extract parameter types
    for idx, arg in enumerate(func_node.args.args):
        if arg.annotation:
            frame_type, _ = _resolve_declared_type(arg.annotation, schema_registry)
            if frame_type is not None:
                parameters[arg.arg] = (idx, frame_type)

    # Extract return type
    if func_node.returns:
        frame_type, _ = _resolve_declared_type(func_node.returns, schema_registry)
        if frame_type is not None:
            return_type = frame_type

    # Return None if no DataFrame annotations found
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
    schema_registry: Optional[SchemaRegistry] = None,
) -> Optional[FunctionAnalysis]:
    """Analyze a single function definition."""
    schema_registry = schema_registry or SchemaRegistry()
    input_types: dict[str, FrameType] = {}
    declared_return: Optional[FrameType] = None
    errors: list[str] = []
    has_df_annotation = False

    # Extract input parameter types
    for arg in func_node.args.args:
        if arg.annotation:
            if _annotation_declares_frame(arg.annotation, schema_registry):
                has_df_annotation = True
                frame_type, parse_error = _resolve_declared_type(
                    arg.annotation, schema_registry
                )
                if frame_type is not None:
                    input_types[arg.arg] = frame_type
                elif parse_error:
                    errors.append(f"Parameter '{arg.arg}': {parse_error}")

    # Extract return type
    if func_node.returns:
        if _annotation_declares_frame(func_node.returns, schema_registry):
            has_df_annotation = True
            declared_return, parse_error = _resolve_declared_type(
                func_node.returns, schema_registry
            )
            if parse_error:
                errors.append(f"Return type: {parse_error}")

    # If no DataFrame annotations found, skip this function
    if not has_df_annotation:
        return None

    # Analyze function body with registry
    body_analyzer = FunctionBodyAnalyzer(input_types, errors, registry, schema_registry)
    for stmt in func_node.body:
        body_analyzer.visit(stmt)

    return FunctionAnalysis(
        name=func_node.name,
        lineno=func_node.lineno,
        end_lineno=func_node.end_lineno or func_node.lineno,
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
        List of FunctionAnalysis results for functions with
        ``DataFrame[Schema]`` / ``LazyFrame[Schema]`` annotations
    """
    tree = ast.parse(source)

    # Pass 0: Collect Pandera DataFrameModel schemas at top level
    schema_registry = collect_schemas(tree)

    # Pass 1: Collect all function nodes
    func_nodes: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_nodes.append(node)

    # Pass 2: Build registry with all functions
    registry = FunctionRegistry()
    for func_node in func_nodes:
        signature = _extract_function_signature(func_node, schema_registry)
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
        analysis = analyze_function(func_node, registry, schema_registry)
        if analysis:
            results.append(analysis)

    return results
