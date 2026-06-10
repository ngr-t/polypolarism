"""AST analysis and data flow tracking."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from polypolarism.compat.polars_api import (
    AGG_SHORTHAND_NAMES,
    DT_NAMESPACE_PRESERVING,
    DT_NAMESPACE_RETURN,
    DTYPE_NAME_MAP,
    EAGER_ONLY_METHODS,
    IDENTITY_FRAME_METHODS,
    LAZY_ONLY_METHODS,
    LIST_NAMESPACE_ELEMENT_RETURN,
    LIST_NAMESPACE_PRESERVING,
    STR_NAMESPACE_RETURN,
    agg_function_for,
    canonicalize_method,
)
from polypolarism.diagnostics import (
    PLW001,
    PLW002,
    PLW003,
    PLW004,
    PLW005,
    PLW006,
    PLY001,
    PLY002,
    PLY003,
    PLY004,
    PLY005,
    PLY006,
    PLY010,
    PLY011,
    PLY020,
    PLY021,
    PLY022,
    PLY030,
    PLY031,
    PLY032,
    tag,
)
from polypolarism.expr_infer import (
    ColumnNotFoundError,
    TypeUnificationError,
    infer_col,
    unify_types,
)
from polypolarism.ops.groupby import (
    AggExpr,
    AggFunction,
    GroupByTypeError,
    infer_agg_result_type,
    infer_groupby_result,
)
from polypolarism.ops.join import JoinError, JoinHow, infer_join
from polypolarism.ops.reshape import (
    ReshapeError,
    concat_diagonal,
    concat_horizontal,
    concat_vertical,
)
from polypolarism.ops.reshape import (
    unpivot as infer_unpivot,
)
from polypolarism.pandera_annotation import (
    extract_dataframe_annotation,
    frame_annotation_schema_name,
)
from polypolarism.pandera_schema import (
    SchemaRegistry,
    collect_schemas,
    collect_schemas_with_imports,
)
from polypolarism.types import (
    Boolean,
    ColumnSpec,
    DataType,
    Date,
    Datetime,
    Duration,
    Float16,
    Float32,
    Float64,
    FrameType,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Nullable,
    RowVar,
    Struct,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
    Utf8,
)
from polypolarism.types import (
    List as ListT,
)


class AnalysisError(Exception):
    """Error during analysis."""

    pass


# Frame method classification — the polars surface knowledge ("which
# methods exist on which frame type, and which preserve shape") lives in
# ``compat.polars_api``. These local aliases keep the existing call sites
# legible.
_IDENTITY_FRAME_METHODS = IDENTITY_FRAME_METHODS
_LAZY_ONLY_METHODS = LAZY_ONLY_METHODS
_EAGER_ONLY_METHODS = EAGER_ONLY_METHODS


# Map ``pl.<Name>`` attribute references to our DataType. Single source of
# truth lives in ``compat.polars_api``; aliased here so existing call sites
# keep their familiar name. Parametrized dtypes (``Datetime``, ``Duration``,
# ``Decimal``) are still handled via Call form below.
_PL_DTYPE_NAME_MAP: dict[str, DataType] = DTYPE_NAME_MAP


# ``pl.<name>(col)`` top-level aggregation shorthand — equivalent to
# ``pl.col(col).<name>()``. Lookup goes through ``compat.polars_api`` so
# the analyzer and the dispatch tables share one source of truth. The
# strict subset of names polars exposes as ``pl.<name>("col")`` (i.e.
# everything in AGG_NAME_MAP minus ``list``, ``quantile``, ``product``)
# lives in ``AGG_SHORTHAND_NAMES``.
def _pl_agg_shorthand(name: str) -> AggFunction | None:
    if name not in AGG_SHORTHAND_NAMES:
        return None
    return agg_function_for(name)


def _resolve_pl_dtype(node: ast.expr) -> DataType | None:
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
                tz: str | None = None
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


def _lazy_like(result: FrameType | None, source: FrameType | None) -> FrameType | None:
    """Stamp ``source.is_lazy`` onto a freshly-built ``result`` FrameType.

    ``ops/{join,groupby,reshape}.py`` build new FrameTypes without knowing
    about laziness — the analyser post-processes their output through this
    helper so the eager/lazy distinction is preserved across the operation.
    """
    if result is None or source is None:
        return result
    result.is_lazy = source.is_lazy
    return result


def _set_lazy(result: FrameType | None, lazy: bool) -> FrameType | None:
    """Force ``is_lazy`` to a specific value (for ``df.lazy()`` / ``lf.collect()``)."""
    if result is None:
        return None
    result.is_lazy = lazy
    return result


def _is_cast_func(node: ast.expr) -> bool:
    """Recognise ``cast`` (from ``typing import cast``) and ``typing.cast``.

    Also accepts ``t.cast`` or any ``<alias>.cast`` form — at AST time we
    can't always resolve the import alias, and ``cast`` is unambiguous
    enough as a name that treating it as the typing helper is safe.
    """
    if isinstance(node, ast.Name) and node.id == "cast":
        return True
    return isinstance(node, ast.Attribute) and node.attr == "cast"


def _unwrap_cast(node: ast.expr) -> ast.expr:
    """If ``node`` is ``cast(T, value)`` / ``typing.cast(T, value)``, return
    the inner ``value`` expression — recursively, in case of nested casts.

    ``typing.cast`` is a no-op at runtime; users add it so that mypy/pyright
    see the call's static type as ``T``. polypolarism verifies the user's
    claim by inferring the inner expression and checking it against the
    surrounding context (function return type, narrowed variable, etc.).
    Otherwise the cast would just hide ``Schema.validate(...)`` from us.
    """
    while isinstance(node, ast.Call) and _is_cast_func(node.func) and len(node.args) >= 2:
        node = node.args[1]
    return node


def _str_constant(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _str_list_or_tuple(node: ast.expr) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        out: list[str] = []
        for elt in node.elts:
            s = _str_constant(elt)
            if s is None:
                return None
            out.append(s)
        return out
    return None


# polars.selectors return-type predicates by selector name.
_SELECTOR_NUMERIC = (
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Float16,
    Float32,
    Float64,
)
_SELECTOR_INTEGER = (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
_SELECTOR_FLOAT = (Float16, Float32, Float64)
_SELECTOR_TEMPORAL = (Date, Datetime, Duration)


def _selector_dtype_filter(name: str):
    """Return a predicate ``(DataType) -> bool`` for selector ``cs.<name>()``."""
    base_map = {
        "numeric": _SELECTOR_NUMERIC,
        "integer": _SELECTOR_INTEGER,
        "float": _SELECTOR_FLOAT,
        "string": (Utf8,),
        "boolean": (Boolean,),
        "temporal": _SELECTOR_TEMPORAL,
    }
    target = base_map.get(name)
    if target is None:
        return None

    def _match(dtype: DataType) -> bool:
        inner = dtype.inner if isinstance(dtype, Nullable) else dtype
        return isinstance(inner, target)

    return _match


def _resolve_selector(node: ast.expr, frame: FrameType) -> list[str] | None:
    """Resolve a ``polars.selectors`` (``cs.*``) call to a list of column names.

    Returns ``None`` if the node isn't a recognised selector. Also handles
    selector algebra:
    - ``cs.a | cs.b``: union
    - ``cs.a & cs.b``: intersection
    - ``cs.a - cs.b``: difference (left minus right)
    - ``~cs.a``: complement (every column not matched by ``cs.a``)
    - ``cs.exclude(names_or_selector)``: shorthand for ``~`` against a name
      list or another selector.
    """
    # Selector algebra — recurse into operands first so ``cs.numeric() - cs.by_name(...)``
    # works regardless of how deeply nested the operators are.
    if isinstance(node, ast.BinOp):
        left = _resolve_selector(node.left, frame)
        right = _resolve_selector(node.right, frame)
        if left is None or right is None:
            return None
        right_set = set(right)
        if isinstance(node.op, ast.BitOr):
            seen: set[str] = set()
            out: list[str] = []
            for c in [*left, *right]:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            return out
        if isinstance(node.op, ast.BitAnd):
            return [c for c in left if c in right_set]
        if isinstance(node.op, ast.Sub):
            return [c for c in left if c not in right_set]
        return None

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        sub = _resolve_selector(node.operand, frame)
        if sub is None:
            return None
        excluded = set(sub)
        return [c for c in frame.columns if c not in excluded]

    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute):
        return None
    if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "cs"):
        return None
    name = node.func.attr

    if name == "all":
        return list(frame.columns.keys())

    pred = _selector_dtype_filter(name)
    if pred is not None:
        return [c for c, spec in frame.columns.items() if pred(spec.dtype)]

    if name == "by_name":
        out: list[str] = []
        for arg in node.args:
            single = _str_constant(arg)
            multi = _str_list_or_tuple(arg)
            if single is not None:
                out.append(single)
            elif multi is not None:
                out.extend(multi)
        return out

    if name == "by_dtype":
        targets: list[DataType] = []
        for arg in node.args:
            multi_args: list[ast.expr] = (
                list(arg.elts) if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
            )
            for inner_arg in multi_args:
                resolved = _resolve_pl_dtype(inner_arg)
                if resolved is not None:
                    targets.append(resolved)
        if not targets:
            return []

        def _matches(dtype: DataType) -> bool:
            inner = dtype.inner if isinstance(dtype, Nullable) else dtype
            return any(inner == t for t in targets)

        return [c for c, spec in frame.columns.items() if _matches(spec.dtype)]

    if name in ("starts_with", "ends_with", "contains"):
        if not node.args:
            return []
        needle = _str_constant(node.args[0])
        if needle is None:
            return []
        if name == "starts_with":
            return [c for c in frame.columns if c.startswith(needle)]
        if name == "ends_with":
            return [c for c in frame.columns if c.endswith(needle)]
        return [c for c in frame.columns if needle in c]

    if name == "exclude":
        # ``cs.exclude("a", "b")``, ``cs.exclude(["a", "b"])``, or
        # ``cs.exclude(cs.<other_selector>())``. The result is every column
        # that does *not* match the inner specification.
        excluded: set[str] = set()
        for arg in node.args:
            single = _str_constant(arg)
            multi = _str_list_or_tuple(arg)
            if single is not None:
                excluded.add(single)
            elif multi is not None:
                excluded.update(multi)
            else:
                inner = _resolve_selector(arg, frame)
                if inner is not None:
                    excluded.update(inner)
        return [c for c in frame.columns if c not in excluded]

    if name in ("first", "last"):
        if not frame.columns:
            return []
        cols = list(frame.columns.keys())
        return [cols[0]] if name == "first" else [cols[-1]]

    return None


# =============================================================================
# Data structures for function registry
# =============================================================================


@dataclass
class FunctionSignature:
    """Type signature for a function with ``DataFrame[Schema]`` annotations."""

    name: str
    parameters: dict[str, tuple[int, FrameType]]  # param_name -> (position, type)
    return_type: FrameType | None
    lineno: int

    def get_param_by_position(self, position: int) -> tuple[str, FrameType] | None:
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
    signature: FunctionSignature | None  # None if untyped
    inferred_returns: dict[tuple, FrameType] = field(default_factory=dict)


@dataclass
class FunctionRegistry:
    """Registry of all functions in a file."""

    functions: dict[str, FunctionInfo] = field(default_factory=dict)

    def register(self, info: FunctionInfo) -> None:
        """Register a function."""
        self.functions[info.name] = info

    def get(self, name: str) -> FunctionInfo | None:
        """Get function info by name."""
        return self.functions.get(name)

    def has_signature(self, name: str) -> bool:
        """Check if function has a type signature."""
        info = self.functions.get(name)
        return info is not None and info.signature is not None


@dataclass
class ClassRegistry:
    """Registry of class methods, keyed by ``(class_name, method_name)``.

    Classes share a flat name → method map at the module scope; nested
    classes aren't supported (rare in DataFrame pipeline code, and the
    extra plumbing would obscure what this registry exists for: looking
    up ``self.foo()`` / ``ClassName().foo()`` to recover an annotated
    return type).
    """

    classes: dict[str, dict[str, FunctionInfo]] = field(default_factory=dict)

    def register_method(self, class_name: str, info: FunctionInfo) -> None:
        self.classes.setdefault(class_name, {})[info.name] = info

    def get_method(self, class_name: str, method_name: str) -> FunctionInfo | None:
        methods = self.classes.get(class_name)
        if methods is None:
            return None
        return methods.get(method_name)

    def __contains__(self, class_name: str) -> bool:
        return class_name in self.classes


# =============================================================================
# Type compatibility checking
# =============================================================================


def _is_column_subtype(actual: DataType, expected: DataType) -> bool:
    """Check if actual is a subtype of expected.

    Rules:
    - T is subtype of T
    - T is subtype of Nullable[T]
    - Nullable[T] is NOT subtype of T
    - Unknown is compatible with everything in both directions (gradual
      typing: uncertainty must not error), even ``Nullable[Unknown]`` vs a
      non-nullable expected type.
    """
    actual_base = actual.inner if isinstance(actual, Nullable) else actual
    expected_base = expected.inner if isinstance(expected, Nullable) else expected
    if isinstance(actual_base, Unknown) or isinstance(expected_base, Unknown):
        return True

    if actual == expected:
        return True

    # Non-nullable is subtype of nullable with same base
    if isinstance(expected, Nullable) and not isinstance(actual, Nullable):
        return actual == expected.inner

    return False


def _is_frame_subtype(actual: FrameType, expected: FrameType) -> bool:
    """Check if actual FrameType is subtype of expected.

    Rules:
    - actual must contain every required column expected has — unless
      ``actual`` is an open frame (``rest`` is not None), whose unknown
      extra columns may satisfy the requirement
    - For columns present on both sides, the actual dtype must be a subtype
      and an actual optional column cannot satisfy a required expected column
    - actual may have extra columns unless ``expected.strict`` is True
    """
    for col_name, expected_spec in expected.columns.items():
        actual_spec = actual.columns.get(col_name)
        if actual_spec is None:
            if expected_spec.required and actual.rest is None:
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
    declared_return_type: FrameType | None
    inferred_return_type: FrameType | None
    errors: list[str] = field(default_factory=list)
    # Non-fatal advisories: situations where polypolarism can't precisely
    # check the code and the user could fix that by adding an annotation
    # or an explicit dtype. Does not affect ``has_errors``.
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True if any errors were found during analysis."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Return True if any warnings were emitted."""
        return len(self.warnings) > 0


def _resolve_declared_type(
    annotation: ast.expr,
    schema_registry: SchemaRegistry,
) -> tuple[FrameType | None, str | None]:
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

    def __init__(
        self,
        current_frame: FrameType,
        warnings: list[str] | None = None,
        registry: FunctionRegistry | None = None,
    ):
        self.current_frame = current_frame
        self.errors: list[str] = []
        # Shared advisory channel (passed in by the body analyzer so warnings
        # bubble up to FunctionAnalysis). New list when used standalone.
        self.warnings: list[str] = warnings if warnings is not None else []
        self.registry = registry or FunctionRegistry()

    def analyze_agg_expr(self, node: ast.expr) -> AggExpr | None:
        """Analyze an aggregation expression like pl.col("x").sum().alias("total")."""
        # Pattern: pl.col("col").agg_func().alias("name") or pl.col("col").agg_func()
        alias = None
        agg_node = node

        # Check for .alias("name") at the end
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "alias":
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    alias = node.args[0].value
                    agg_node = node.func.value

        # Now look for the aggregation function call
        if isinstance(agg_node, ast.Call):
            if isinstance(agg_node.func, ast.Attribute):
                agg_func_name = agg_node.func.attr
                col_expr = agg_node.func.value

                agg_func_value = agg_function_for(agg_func_name)
                if agg_func_value is not None:
                    # Form 1: ``pl.col("X").<agg>()`` — receiver is the col.
                    col_name = self._extract_col_name(col_expr)

                    # Form 2: ``pl.<agg>("X")`` top-level shorthand — the
                    # column name is the first positional arg of the call,
                    # not buried inside a separate ``pl.col(...)`` receiver.
                    if (
                        col_name is None
                        and isinstance(col_expr, ast.Name)
                        and col_expr.id == "pl"
                        and agg_node.args
                    ):
                        col_name = _str_constant(agg_node.args[0])

                    if col_name:
                        return AggExpr(
                            column=col_name,
                            function=agg_func_value,
                            alias=alias,
                        )

        # Chain fallback: anything more elaborate (post-aggregation method
        # chains like ``pl.col("ts").max().dt.year()``, arithmetic on the
        # aggregated value, sub-namespace methods on the aggregated value,
        # etc.) is handled by reusing the expression analyser. We delegate to
        # ``analyze_select_expr`` on the *original* node (it strips the
        # ``.alias(...)`` itself), and turn its (name, dtype) into an
        # AggExpr with a pre-resolved ``dtype`` override.
        chain_name, chain_dtype = self.analyze_select_expr(node)
        if chain_dtype is not None:
            # Anchor the AggExpr to the deepest pl.col so the column-existence
            # check elsewhere has a sensible source attribution; if there's no
            # bare pl.col (e.g. a literal-driven expression) we fall back to
            # the alias / inferred name.
            anchor_col = self._find_deep_col(node) or chain_name or ""
            return AggExpr(
                column=anchor_col,
                function=None,
                alias=chain_name,
                dtype=chain_dtype,
            )

        return None

    def _find_deep_col(self, node: ast.expr) -> str | None:
        """Walk down a method chain to find the innermost ``pl.col("X")`` reference."""
        if isinstance(node, ast.Call):
            direct = self._extract_col_name(node)
            if direct is not None:
                return direct
            if isinstance(node.func, ast.Attribute):
                return self._find_deep_col(node.func.value)
        if isinstance(node, ast.Attribute):
            return self._find_deep_col(node.value)
        if isinstance(node, ast.BinOp):
            return self._find_deep_col(node.left) or self._find_deep_col(node.right)
        return None

    def _extract_col_name(self, node: ast.expr) -> str | None:
        """Extract column name from pl.col("name") expression."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "col":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "pl":
                        if (
                            node.args
                            and isinstance(node.args[0], ast.Constant)
                            and isinstance(node.args[0].value, str)
                        ):
                            return node.args[0].value
        return None

    # Methods on a column expression that always produce Boolean.
    _BOOLEAN_PREDICATE_METHODS = frozenset(
        {
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
        }
    )

    # Methods that return Float64 from any numeric receiver.
    _FLOAT_RETURN_METHODS = frozenset(
        {
            "log",
            "log10",
            "log1p",
            "exp",
            "sqrt",
            "cbrt",
            "entropy",
        }
    )

    # Methods that preserve the receiver's dtype (numeric mostly).
    _DTYPE_PRESERVING_METHODS = frozenset(
        {
            "abs",
            "round",
            "clip",
            "floor",
            "ceil",
            "sign",
            "neg",
            "shrink_dtype",
            "rechunk",
            # Cumulative / window — preserve receiver dtype.
            "cum_sum",
            "cum_max",
            "cum_min",
            "cum_prod",
            "over",
            "rolling_sum",
            "rolling_min",
            "rolling_max",
            "set_sorted",
            "reverse",
        }
    )

    # Shift-like methods: receiver dtype, but head positions become NULL.
    _SHIFT_LIKE_METHODS = frozenset({"shift", "diff", "pct_change"})

    # Rolling reductions to Float64.
    _ROLLING_FLOAT_METHODS = frozenset(
        {
            "rolling_mean",
            "rolling_std",
            "rolling_var",
            "rolling_median",
            "rolling_quantile",
        }
    )

    # ---- sub-namespace return type tables ----------------------------------
    # All five tables are re-exports from compat.polars_api so the polars
    # surface knowledge stays in one place.

    _STR_RETURN = STR_NAMESPACE_RETURN
    _DT_RETURN = DT_NAMESPACE_RETURN
    _DT_PRESERVING = DT_NAMESPACE_PRESERVING
    _LIST_PRESERVING = LIST_NAMESPACE_PRESERVING
    _LIST_ELEMENT_RETURN = LIST_NAMESPACE_ELEMENT_RETURN

    def analyze_select_expr(self, node: ast.expr) -> tuple[str | None, DataType | None]:
        """Analyze a select expression, return (output_name, type)."""
        # Check for .alias() wrapper
        alias = None
        inner_node = node

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "alias":
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
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
        if isinstance(inner_node, ast.UnaryOp) and isinstance(inner_node.op, (ast.Invert, ast.Not)):
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

        # Top-level pl.<func>(...) constructors (pl.struct / concat_str / format / coalesce)
        pl_result = self._analyze_pl_func(inner_node)
        if pl_result is not None:
            pl_name, pl_type = pl_result
            output_name = alias if alias else pl_name
            return output_name, pl_type

        # Check for pl.col("name")
        col_name = self._extract_col_name(inner_node)
        if col_name:
            try:
                col_type = infer_col(col_name, self.current_frame)
                output_name = alias if alias else col_name
                return output_name, col_type
            except ColumnNotFoundError as e:
                self.errors.append(tag(PLY001, str(e)))
                return None, None

        # Check for pl.lit(value)
        lit_type = self._extract_lit_type(inner_node)
        if lit_type:
            return alias, lit_type

        return None, None

    def _analyze_pl_func(self, node: ast.expr) -> tuple[str | None, DataType] | None:
        """Recognise ``pl.struct(...)`` / ``pl.concat_str(...)`` / ``pl.format(...)`` /
        ``pl.coalesce(...)`` top-level constructor calls."""
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, ast.Attribute):
            return None
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "pl"):
            return None
        name = node.func.attr

        if name == "concat_str" or name == "format":
            for arg in node.args:
                self._validate_subexpr(arg)
            return None, Utf8()

        if name == "struct":
            fields: dict[str, DataType] = {}
            for arg in node.args:
                col = self._extract_col_name(arg)
                if col is None:
                    continue
                try:
                    fields[col] = infer_col(col, self.current_frame)
                except ColumnNotFoundError as e:
                    self.errors.append(tag(PLY001, str(e)))
            return None, Struct(fields)

        # ``pl.<agg>("col")`` top-level shorthand — equivalent to
        # ``pl.col("col").<agg>()``. Recognised in ``select`` /
        # ``with_columns`` so the column survives downstream lookups.
        agg_func = _pl_agg_shorthand(name)
        if agg_func is not None and node.args:
            col = _str_constant(node.args[0])
            if col is not None:
                try:
                    col_type = infer_col(col, self.current_frame)
                except ColumnNotFoundError as e:
                    self.errors.append(tag(PLY001, str(e)))
                    return col, None  # type: ignore[return-value]
                try:
                    result_type = infer_agg_result_type(agg_func, col_type)
                except GroupByTypeError as e:
                    self.errors.append(tag(PLY011, str(e)))
                    return col, None  # type: ignore[return-value]
                return col, result_type

        if name == "coalesce":
            inferred: list[DataType] = []
            any_non_nullable = False
            for arg in node.args:
                _, t = self.analyze_select_expr(arg)
                if t is None:
                    continue
                inferred.append(t)
                if not isinstance(t, Nullable):
                    any_non_nullable = True
            if not inferred:
                return None, None  # type: ignore[return-value]
            unified: DataType = inferred[0]
            for t in inferred[1:]:
                try:
                    unified = unify_types(unified, t)
                except TypeUnificationError:
                    return None, unified
            # If any operand is non-Nullable, the coalesced result is non-Nullable.
            if any_non_nullable and isinstance(unified, Nullable):
                unified = unified.inner
            # Preserve the first-arg column name as the default output name.
            first_name = self._extract_col_name(node.args[0]) if node.args else None
            return first_name, unified

        return None

    def _dispatch_namespace_method(
        self,
        namespace: str,
        method: str,
        receiver_type: DataType | None,
        call_node: ast.Call | None = None,
    ) -> DataType | None:
        """Resolve ``<col_expr>.<namespace>.<method>(...)`` to a DataType.

        ``receiver_type`` is the dtype of the column the namespace was attached
        to (``None`` if it couldn't be resolved). The receiver's nullability
        is preserved on the result for almost all of these methods.

        ``call_node`` is the full ``ast.Call`` (e.g. ``...struct.field("x")``);
        passed in so per-namespace handlers that need positional arguments
        (currently just ``struct.field``) can read them.
        """
        receiver_inner = receiver_type
        receiver_is_nullable = False
        if isinstance(receiver_type, Nullable):
            receiver_inner = receiver_type.inner
            receiver_is_nullable = True

        result: DataType | None = None

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
            elif method in self._LIST_ELEMENT_RETURN and isinstance(receiver_inner, ListT):
                result = receiver_inner.inner
        elif namespace == "struct":
            if method == "field" and call_node is not None and call_node.args:
                field_name = _str_constant(call_node.args[0])
                if field_name is not None and isinstance(receiver_inner, Struct):
                    field_dtype = receiver_inner.fields.get(field_name)
                    if field_dtype is None:
                        self.errors.append(
                            tag(
                                PLY001,
                                f"struct.field: '{field_name}' not found in "
                                f"{receiver_inner}. Available fields: "
                                f"{list(receiver_inner.fields.keys())}",
                            )
                        )
                        return None
                    result = field_dtype
            elif method == "rename_fields" and isinstance(receiver_inner, Struct):
                # Returns a Struct with same dtypes but renamed; we keep dtypes.
                result = receiver_inner

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

    def _analyze_method_chain(self, node: ast.expr) -> tuple[str | None, DataType] | None:
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
        # ``pl.col("ts").dt.year()``, ``pl.col("xs").list.get(0)``,
        # ``pl.col("s").struct.field("name")`` etc.
        if isinstance(receiver, ast.Attribute) and receiver.attr in (
            "str",
            "dt",
            "list",
            "arr",
            "struct",
        ):
            ns = receiver.attr
            col_name, col_type = self.analyze_select_expr(receiver.value)
            # Struct.field needs the field name (positional arg) — pass the
            # whole call node so the dispatcher can look at it.
            ns_result = self._dispatch_namespace_method(ns, method, col_type, node)
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
            receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
            result: DataType = Float64()
            if isinstance(receiver_type, Nullable):
                result = Nullable(Float64())
            return receiver_name, result

        # Dtype-preserving methods.
        if method in self._DTYPE_PRESERVING_METHODS and receiver_type is not None:
            return receiver_name, receiver_type

        # cum_count is the only cumulative that doesn't preserve dtype.
        if method == "cum_count":
            return receiver_name, UInt32()

        # Shift-like: head positions become NULL → wrap in Nullable.
        if method in self._SHIFT_LIKE_METHODS and receiver_type is not None:
            inner = receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
            return receiver_name, Nullable(inner)

        # Rolling reductions returning Float64.
        if method in self._ROLLING_FLOAT_METHODS and receiver_type is not None:
            if isinstance(receiver_type, Nullable):
                return receiver_name, Nullable(Float64())
            return receiver_name, Float64()

        # ``pl.col("x").map_elements(fn, return_dtype=pl.Float64)`` /
        # ``map_batches(fn, return_dtype=...)``. The return type is what the
        # user declared; without ``return_dtype`` it's uninferable, so we
        # fall back to the receiver dtype and emit ``PLW001`` so the user
        # knows to add the kwarg.
        if method in ("map_elements", "map_batches"):
            for kw in node.keywords:
                if kw.arg == "return_dtype":
                    declared = _resolve_pl_dtype(kw.value)
                    if declared is not None:
                        if receiver_type is not None and isinstance(receiver_type, Nullable):
                            return receiver_name, Nullable(declared)
                        return receiver_name, declared
            self.warnings.append(
                tag(
                    PLW001,
                    f"{method}: no `return_dtype=` was supplied, so polypolarism "
                    f"falls back to the receiver dtype. Add e.g. "
                    f"`return_dtype=pl.Float64` to make the result type precise.",
                )
            )
            return receiver_name, receiver_type if receiver_type is not None else Boolean()

        # ``pl.col("x").pipe(callable)`` — expression-level pipe. Use the
        # registry when possible; warn for lambda / external names.
        if method == "pipe" and node.args:
            callable_arg = node.args[0]
            if isinstance(callable_arg, ast.Name):
                func_info = self.registry.get(callable_arg.id)
                if func_info is not None and func_info.signature is not None:
                    declared_return = func_info.signature.return_type
                    if declared_return is not None:
                        # Helper has a frame return; not directly usable as expr dtype,
                        # but the same machinery can carry frame-returning expr pipes.
                        # Treat as receiver-typed for column inference here.
                        return receiver_name, receiver_type if receiver_type else Boolean()
                # Unknown callable in expr.pipe — uninferable.
                self.warnings.append(
                    tag(
                        PLW002,
                        f"expr.pipe: callable '{callable_arg.id}' is not annotated "
                        f"or not in this module. The expression's return dtype "
                        f"cannot be inferred precisely.",
                    )
                )
                return receiver_name, receiver_type if receiver_type else Boolean()
            if isinstance(callable_arg, ast.Lambda):
                self.warnings.append(
                    tag(
                        PLW004,
                        "expr.pipe: a lambda was passed; promote it to a top-level "
                        "function with a typed signature so polypolarism can infer "
                        "the return dtype.",
                    )
                )
                return receiver_name, receiver_type if receiver_type else Boolean()

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
                return receiver_name, infer_agg_result_type(agg_map[method], receiver_type)
            except GroupByTypeError as e:
                self.errors.append(str(e))
                return receiver_name, receiver_type

        # ``cast(pl.<dtype>)`` chained directly on column.
        if method == "cast" and node.args:
            target = _resolve_pl_dtype(node.args[0])
            if target is not None and receiver_type is not None:
                return receiver_name, _wrap_like(receiver_type, target)

        return None

    def _extract_lit_type(self, node: ast.expr) -> DataType | None:
        """Extract type from pl.lit(value) expression."""
        from polypolarism.types import Boolean, Float64, Int64, Null, Utf8

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
        registry: FunctionRegistry | None = None,
        schema_registry: SchemaRegistry | None = None,
        warnings: list[str] | None = None,
        class_registry: ClassRegistry | None = None,
        current_class_name: str | None = None,
    ):
        self.input_types = input_types
        self.errors = errors
        # Non-fatal advisories. Owned externally so nested analyses can append.
        self.warnings: list[str] = warnings if warnings is not None else []
        self.registry = registry or FunctionRegistry()
        self.schema_registry = schema_registry or SchemaRegistry()
        self.class_registry = class_registry or ClassRegistry()
        # Name of the class enclosing the function being analysed (None for
        # module-level functions). Used to resolve ``self.method()`` /
        # ``cls.method()`` to a class-local method's return annotation.
        self.current_class_name = current_class_name
        # Track variable -> FrameType mapping
        self.var_types: dict[str, FrameType] = dict(input_types)
        # Track variable -> FrameList element type (for partition_by results
        # and any other op that yields a list of frames).
        self.var_lists: dict[str, FrameType] = {}
        # Track variable -> class name for ``var = ClassName()`` assignments,
        # so a later ``var.method()`` call can resolve to the class's method.
        self.var_classes: dict[str, str] = {}
        self.return_type: FrameType | None = None
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
        # ``for part in df.partition_by(...):`` / ``for part in parts:`` —
        # bind the loop target to the FrameList element type so column
        # references inside the loop body resolve. The binding stays in
        # place after the loop, mirroring Python semantics.
        if isinstance(node.target, ast.Name):
            elem = self._resolve_frame_list_element(node.iter)
            if elem is not None:
                self.var_types[node.target.id] = elem
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
        inner = _unwrap_cast(node.value)
        if not isinstance(inner, ast.Call):
            return
        call = inner
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
            # Preserve laziness from the variable being narrowed.
            schema_ft.is_lazy = self.var_types[arg.id].is_lazy
            self.var_types[arg.id] = schema_ft

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle variable assignments."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            # ``parts = df.partition_by("k")`` — bind to FrameList element type.
            list_elem = self._infer_frame_list(node.value)
            if list_elem is not None:
                self.var_lists[var_name] = list_elem
                self.var_types.pop(var_name, None)
                self.var_classes.pop(var_name, None)
                self.generic_visit(node)
                return
            inferred = self._infer_expr_type(node.value)
            if inferred:
                self.var_types[var_name] = inferred
                self.var_lists.pop(var_name, None)
                self.var_classes.pop(var_name, None)
            else:
                # ``obj = ClassName()`` — track the class for later
                # ``obj.method()`` resolution.
                cls_name = self._instance_class_name(node.value)
                if cls_name is not None:
                    self.var_classes[var_name] = cls_name
                    self.var_types.pop(var_name, None)
                    self.var_lists.pop(var_name, None)
        self.generic_visit(node)

    def _instance_class_name(self, node: ast.expr) -> str | None:
        """If ``node`` is ``ClassName(...)`` for a class we know about,
        return ``ClassName``; otherwise ``None``."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.class_registry
        ):
            return node.func.id
        return None

    def _resolve_method_call(self, node: ast.Call) -> FrameType | None:
        """Resolve ``self.foo() / cls.foo() / Class().foo() / obj.foo()`` to
        the callee's annotated ``DataFrame[Schema]`` return type.

        Returns ``None`` when the receiver isn't class-typed, the method
        isn't in the class, or it has no DataFrame-shaped return
        annotation. Argument validation is intentionally skipped here —
        the goal is to recover the return type so chains like
        ``self._load() -> self._transform(...)`` keep their type through
        method boundaries.
        """
        if not isinstance(node.func, ast.Attribute):
            return None
        method_name = node.func.attr
        receiver = node.func.value

        receiver_class: str | None = None
        if isinstance(receiver, ast.Name):
            if receiver.id in ("self", "cls") and self.current_class_name is not None:
                receiver_class = self.current_class_name
            elif receiver.id in self.var_classes:
                receiver_class = self.var_classes[receiver.id]
            elif receiver.id in self.class_registry:
                # ``Class.method()`` — static/classmethod called without
                # instantiation. Same lookup table as the instance form.
                receiver_class = receiver.id
        elif isinstance(receiver, ast.Call):
            receiver_class = self._instance_class_name(receiver)

        if receiver_class is None:
            return None

        method_info = self.class_registry.get_method(receiver_class, method_name)
        if method_info is None or method_info.signature is None:
            return None
        return method_info.signature.return_type

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments like: df: DataFrame[Schema] = expr."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # Try to get type from a Pandera DataFrame[Schema] annotation
            frame_type, _ = _resolve_declared_type(node.annotation, self.schema_registry)
            if frame_type is not None:
                self.var_types[var_name] = frame_type
                # Still walk the RHS so warnings (e.g. PLW005 from pivot) and
                # column-resolution errors surface even though the declared
                # annotation wins for the variable's static type.
                if node.value:
                    self._infer_expr_type(node.value)
                return
            # Fall back to inference from value
            if node.value:
                inferred = self._infer_expr_type(node.value)
                if inferred:
                    self.var_types[var_name] = inferred
        self.generic_visit(node)

    def _infer_expr_type(self, node: ast.expr) -> FrameType | None:
        """Infer the FrameType of an expression."""
        # ``typing.cast(T, expr)`` / ``cast(T, expr)`` is a static-typing
        # passthrough — defer to the inner expression so existing narrowing
        # rules (Schema.validate, .pipe, etc.) keep working through the cast.
        node = _unwrap_cast(node)

        # Variable reference
        if isinstance(node, ast.Name):
            return self.var_types.get(node.id)

        # Subscript on a FrameList variable: ``parts[0]`` → element FrameType.
        # Whatever index expression the user wrote is irrelevant for typing —
        # every element shares the same schema.
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            elem = self.var_lists.get(node.value.id)
            if elem is not None:
                return elem

        # Method call chain or function call
        if isinstance(node, ast.Call):
            return self._infer_call_type(node)

        return None

    # -- partition_by / FrameList helpers --------------------------------

    def _is_partition_by_call(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "partition_by"
        )

    def _infer_frame_list(self, node: ast.expr) -> FrameType | None:
        """If ``node`` is a partition_by call (or names a FrameList variable),
        return the element FrameType. Otherwise ``None``."""
        if isinstance(node, ast.Name):
            return self.var_lists.get(node.id)
        if self._is_partition_by_call(node):
            assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            receiver_type = self._infer_expr_type(node.func.value)
            if receiver_type is None:
                return None
            return self._compute_partition_element(receiver_type, node)
        return None

    def _compute_partition_element(self, receiver_type: FrameType, node: ast.Call) -> FrameType:
        """``df.partition_by(*by, include_key=True)`` element schema.

        Each partition has the same columns as the receiver, minus the
        partition keys when ``include_key=False``.
        """
        keys: list[str] = []
        for arg in node.args:
            single = _str_constant(arg)
            multi = _str_list_or_tuple(arg)
            if single is not None:
                keys.append(single)
            elif multi is not None:
                keys.extend(multi)
        include_key = True
        for kw in node.keywords:
            if kw.arg == "include_key" and isinstance(kw.value, ast.Constant):
                include_key = bool(kw.value.value)
        if include_key:
            return receiver_type
        result_columns = {
            name: spec for name, spec in receiver_type.columns.items() if name not in keys
        }
        return FrameType(
            columns=result_columns,
            strict=receiver_type.strict,
            rest=receiver_type.rest,
        )

    def _resolve_frame_list_element(self, node: ast.expr) -> FrameType | None:
        """Reach the element type of ``for x in <node>:``.

        Accepts both already-bound variables (``parts``) and inline
        expressions (``df.partition_by("k")``).
        """
        return self._infer_frame_list(node)

    def _infer_call_type(self, node: ast.Call) -> FrameType | None:
        """Infer the type of a method or function call."""
        # Function call: func_name(args)
        if isinstance(node.func, ast.Name):
            return self._infer_function_call_type(node)

        # Method call: obj.method(args)
        if isinstance(node.func, ast.Attribute):
            # Canonicalize through compat — currently a no-op (METHOD_ALIASES
            # is empty), but the entry-point exists so a future polars rename
            # can be absorbed in one line rather than across every dispatch
            # site below.
            method_name = canonicalize_method(node.func.attr)
            receiver = node.func.value

            # ``pl.concat([...], how=...)`` — top-level pl function, not a frame method.
            if isinstance(receiver, ast.Name) and receiver.id == "pl" and method_name == "concat":
                return self._infer_concat_call(node)

            # Pandera narrowing: Schema.validate(df) -> Schema's FrameType
            if method_name == "validate":
                schema_ft = self._infer_validate_call(node)
                if schema_ft is not None:
                    return schema_ft

            # User-defined method calls (``self.method()`` /
            # ``Class().method()`` / ``obj.method()``) whose callee is
            # annotated. Tried before the polars/Pandera method dispatch
            # so the method's return annotation wins, but only fires when
            # the receiver actually resolves to a known class — frame
            # methods like ``df.select(...)`` are unaffected.
            method_ft = self._resolve_method_call(node)
            if method_ft is not None:
                return method_ft

            # df.pipe(callable) — resolve in this order:
            #   1) ``Schema.validate``                      → schema's FrameType
            #   2) a typed/untyped helper in the registry   → its return type
            #   3) anything else                            → identity, with a PLW002
            if method_name == "pipe":
                receiver_type = self._infer_expr_type(receiver)
                piped = self._infer_pipe_call(node, receiver_type)
                if piped is not None:
                    return piped
                return receiver_type

            # LazyFrame ↔ DataFrame transitions: preserve column shape
            # but flip ``is_lazy`` so downstream eager/lazy validation works.
            if method_name in ("collect", "collect_async", "collect_batches"):
                receiver_type = self._infer_expr_type(receiver)
                if receiver_type is not None and not receiver_type.is_lazy:
                    self.errors.append(
                        tag(
                            PLY031,
                            f"`.{method_name}()` is only available on LazyFrame, "
                            f"but the receiver is a DataFrame. Drop the call — "
                            f"it's already eager.",
                        )
                    )
                return _set_lazy(receiver_type, False)
            if method_name == "lazy":
                return _set_lazy(self._infer_expr_type(receiver), True)

            # Handle .agg() call (comes after .group_by()). The receiver of
            # the group_by call is the underlying DataFrame/LazyFrame whose
            # laziness we need to preserve onto the agg result.
            if method_name == "agg":
                source = None
                if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
                    source = self._infer_expr_type(receiver.func.value)
                return _lazy_like(self._infer_agg_call(receiver, node), source)

            # Handle other DataFrame methods
            receiver_type = self._infer_expr_type(receiver)
            if receiver_type is not None:
                self._validate_eager_lazy_method(method_name, receiver_type, node)
            if receiver_type:
                # All of these helpers operate on column shape only; the
                # eager/lazy bit is restamped via ``_lazy_like``.
                if method_name == "join":
                    return _lazy_like(self._infer_join_call(receiver_type, node), receiver_type)
                elif method_name in ("group_by", "group_by_dynamic", "rolling"):
                    # Opaque receiver — the actual frame type comes from .agg().
                    return None
                elif method_name == "join_asof":
                    return _lazy_like(
                        self._infer_join_asof_call(receiver_type, node), receiver_type
                    )
                elif method_name == "select":
                    return _lazy_like(self._infer_select_call(receiver_type, node), receiver_type)
                elif method_name == "with_columns":
                    return _lazy_like(
                        self._infer_with_columns_call(receiver_type, node), receiver_type
                    )
                elif method_name == "drop":
                    return _lazy_like(self._infer_drop_call(receiver_type, node), receiver_type)
                elif method_name == "rename":
                    return _lazy_like(self._infer_rename_call(receiver_type, node), receiver_type)
                elif method_name == "cast":
                    return _lazy_like(self._infer_cast_call(receiver_type, node), receiver_type)
                elif method_name == "drop_nulls":
                    return _lazy_like(
                        self._infer_drop_nulls_call(receiver_type, node), receiver_type
                    )
                elif method_name == "with_row_index":
                    return _lazy_like(
                        self._infer_with_row_index_call(receiver_type, node), receiver_type
                    )
                elif method_name == "filter":
                    return _lazy_like(self._infer_filter_call(receiver_type, node), receiver_type)
                elif method_name == "explode":
                    return _lazy_like(self._infer_explode_call(receiver_type, node), receiver_type)
                elif method_name == "vstack":
                    return _lazy_like(self._infer_vstack_call(receiver_type, node), receiver_type)
                elif method_name in ("hstack", "extend"):
                    return _lazy_like(self._infer_hstack_call(receiver_type, node), receiver_type)
                elif method_name in ("unpivot", "melt"):
                    return _lazy_like(self._infer_unpivot_call(receiver_type, node), receiver_type)
                elif method_name == "unnest":
                    return _lazy_like(self._infer_unnest_call(receiver_type, node), receiver_type)
                elif method_name == "pivot":
                    return self._infer_pivot_call(receiver_type, node)
                elif method_name in _IDENTITY_FRAME_METHODS:
                    return receiver_type

        return None

    def _validate_eager_lazy_method(self, method: str, receiver: FrameType, node: ast.Call) -> None:
        """Surface PLY030 / PLY031 when an eager-only or lazy-only method is
        called on the wrong side of the eager / lazy split."""
        if receiver.is_lazy and method in _EAGER_ONLY_METHODS:
            self.errors.append(
                tag(
                    PLY030,
                    f"`.{method}()` is only available on DataFrame, but the "
                    f"receiver is a LazyFrame. Insert `.collect()` before "
                    f"`.{method}()` or work with the eager API throughout.",
                )
            )
        elif (not receiver.is_lazy) and method in _LAZY_ONLY_METHODS:
            self.errors.append(
                tag(
                    PLY031,
                    f"`.{method}()` is only available on LazyFrame, but the "
                    f"receiver is a DataFrame. Call `.lazy()` before "
                    f"`.{method}()` or use the eager equivalent.",
                )
            )

    def _infer_validate_call(self, node: ast.Call) -> FrameType | None:
        """Resolve ``Schema.validate(df_or_lf)`` to the schema's FrameType.

        ``Schema.validate`` is eager/lazy-polymorphic, so the result inherits
        the laziness of the argument it was called on.
        """
        if not isinstance(node.func, ast.Attribute):
            return None
        schema_node = node.func.value
        if not isinstance(schema_node, ast.Name):
            return None
        ft = self.schema_registry.to_frame_type(schema_node.id)
        if ft is None or not node.args:
            return ft
        arg_type = self._infer_expr_type(node.args[0])
        if arg_type is not None:
            ft.is_lazy = arg_type.is_lazy
        return ft

    def _infer_pipe_call(
        self,
        node: ast.Call,
        receiver_type: FrameType | None = None,
    ) -> FrameType | None:
        """Resolve ``df.pipe(callable, *args, **kwargs)``.

        Recognised forms:
        - ``df.pipe(Schema.validate)`` → that schema's FrameType.
        - ``df.pipe(my_helper, ...)`` → if ``my_helper`` is in the registry,
          we treat the call like ``my_helper(df, *args, **kwargs)`` and
          delegate to the same inference path used by direct calls.
        - Anything else (``df.pipe(lambda d: ...)`` / ``df.pipe(some_import)``
          where the callable isn't analysable) → emit ``PLW002`` and return
          ``None`` so the caller falls back to identity.
        """
        if not node.args:
            return None
        callable_arg = node.args[0]

        # 1) Schema.validate — preserves laziness of the piped-in receiver
        if isinstance(callable_arg, ast.Attribute) and callable_arg.attr == "validate":
            if isinstance(callable_arg.value, ast.Name):
                ft = self.schema_registry.to_frame_type(callable_arg.value.id)
                if ft is not None and receiver_type is not None:
                    ft.is_lazy = receiver_type.is_lazy
                return ft

        # 2) registry helper — synthesise a function call
        if isinstance(callable_arg, ast.Name):
            func_info = self.registry.get(callable_arg.id)
            if func_info is not None:
                synthesized = ast.Call(
                    func=ast.Name(id=callable_arg.id, ctx=ast.Load()),
                    args=[node.func.value, *node.args[1:]]  # type: ignore[union-attr]
                    if isinstance(node.func, ast.Attribute)
                    else list(node.args[1:]),
                    keywords=list(node.keywords),
                )
                ast.copy_location(synthesized, node)
                ast.copy_location(synthesized.func, node)
                return self._infer_function_call_type(synthesized)
            # Unknown name — likely an external import. Warn.
            self.warnings.append(
                tag(
                    PLW002,
                    f"pipe: callable '{callable_arg.id}' is not annotated or not "
                    f"in this module; treating as identity. To make polypolarism "
                    f"check it, define '{callable_arg.id}' here with a "
                    f"DataFrame[Schema] return annotation, or call it directly.",
                )
            )
            return receiver_type

        # 3) lambda / arbitrary expression — uninferable.
        if isinstance(callable_arg, ast.Lambda):
            self.warnings.append(
                tag(
                    PLW004,
                    "pipe: a lambda was passed as the callable; polypolarism cannot "
                    "infer its return type. Promote the lambda to a top-level "
                    "function with a DataFrame[Schema] return annotation.",
                )
            )
            return receiver_type

        return None

    def _infer_function_call_type(self, node: ast.Call) -> FrameType | None:
        """Infer type of a function call like helper(df)."""
        if not isinstance(node.func, ast.Name):
            return None

        func_name = node.func.id
        func_info = self.registry.get(func_name)

        if func_info is None:
            # Unknown function — likely imported from another module. We can't
            # walk its body, so the return type is uninferable. Warn the user
            # so they know the downstream type tracking will be lost here.
            args_with_frame = any(self._infer_expr_type(arg) is not None for arg in node.args)
            if args_with_frame:
                self.warnings.append(
                    tag(
                        PLW003,
                        f"call to '{func_name}': function isn't defined in this "
                        f"module so polypolarism cannot infer its return type. "
                        f"Define '{func_name}' here with a DataFrame[Schema] "
                        f"return annotation, or inline the transformation.",
                    )
                )
            return None

        # Infer argument types
        arg_types: list[FrameType | None] = []
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
                # Eager/lazy must match — can't pass a LazyFrame where a
                # DataFrame is expected (and vice versa).
                if arg_type.is_lazy != expected_type.is_lazy:
                    expected_kind = "LazyFrame" if expected_type.is_lazy else "DataFrame"
                    actual_kind = "LazyFrame" if arg_type.is_lazy else "DataFrame"
                    fix = (
                        ".collect() the LazyFrame first"
                        if arg_type.is_lazy
                        else ".lazy() the DataFrame first"
                    )
                    self.errors.append(
                        tag(
                            PLY032,
                            f"Argument '{param_name}' expected {expected_kind}[...] "
                            f"but got {actual_kind}[...]; {fix}.",
                        )
                    )
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
        self, func_info: FunctionInfo, arg_types: list[FrameType | None]
    ) -> FrameType | None:
        """Analyze an untyped function body with propagated argument types."""
        # Create cache key from argument types
        cache_key = tuple(tuple(sorted(t.columns.items())) if t else None for t in arg_types)
        if cache_key in func_info.inferred_returns:
            return func_info.inferred_returns[cache_key]

        # Build input types from function parameters and provided arg types
        input_types: dict[str, FrameType] = {}
        func_node = func_info.node
        for idx, arg in enumerate(func_node.args.args):
            if idx < len(arg_types):
                arg_type = arg_types[idx]
                if arg_type is not None:
                    input_types[arg.arg] = arg_type

        # Analyze the function body — warnings bubble up to the calling
        # body analyzer so the user sees them on the outer function.
        errors: list[str] = []
        body_analyzer = FunctionBodyAnalyzer(
            input_types,
            errors,
            self.registry,
            self.schema_registry,
            warnings=self.warnings,
        )
        for stmt in func_node.body:
            body_analyzer.visit(stmt)

        # Cache and return the result
        result = body_analyzer.return_type
        if result is not None:
            func_info.inferred_returns[cache_key] = result
        return result

    def _infer_join_call(self, left_type: FrameType, node: ast.Call) -> FrameType | None:
        """Infer type of .join() call."""
        # Extract right frame
        if not node.args:
            return None
        right_expr = node.args[0]
        right_type = self._infer_expr_type(right_expr)
        if not right_type:
            return None

        # Extract keyword arguments
        on: str | None = None
        left_on: str | None = None
        right_on: str | None = None
        how: str = "inner"

        for kw in node.keywords:
            if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
                continue
            value = kw.value.value
            if kw.arg == "on":
                on = value
            elif kw.arg == "left_on":
                left_on = value
            elif kw.arg == "right_on":
                right_on = value
            elif kw.arg == "how":
                how = value

        if how == "inner" or how == "left" or how == "right" or how == "full":
            valid_how: JoinHow = how
        else:
            return None

        try:
            return infer_join(
                left_type,
                right_type,
                on=on,
                left_on=left_on,
                right_on=right_on,
                how=valid_how,
            )
        except JoinError as e:
            self.errors.append(tag(PLY010, str(e)))
            return None

    def _infer_join_asof_call(self, left_type: FrameType, node: ast.Call) -> FrameType | None:
        """``df.join_asof(other, ...)`` — same column shape as a left join."""
        if not node.args:
            return None
        right_expr = node.args[0]
        right_type = self._infer_expr_type(right_expr)
        if not right_type:
            return None
        on: str | None = None
        left_on: str | None = None
        right_on: str | None = None
        for kw in node.keywords:
            cand = _str_constant(kw.value)
            if cand is None:
                continue
            if kw.arg == "on":
                on = cand
            elif kw.arg == "left_on":
                left_on = cand
            elif kw.arg == "right_on":
                right_on = cand
        try:
            return infer_join(
                left_type,
                right_type,
                on=on,
                left_on=left_on,
                right_on=right_on,
                how="left",
            )
        except JoinError as e:
            self.errors.append(tag(PLY010, str(e)))
            return None

    def _infer_agg_call(self, groupby_receiver: ast.expr, node: ast.Call) -> FrameType | None:
        """Infer type of .group_by(...).agg(...) / .group_by_dynamic(...).agg(...) /
        .rolling(...).agg(...) calls.

        For the time-window variants the first positional or ``index_column``
        keyword argument is the index column and is treated like a group key.
        """
        # groupby_receiver should be a Call to one of the supported groupers.
        if not isinstance(groupby_receiver, ast.Call):
            return None
        if not isinstance(groupby_receiver.func, ast.Attribute):
            return None
        grouper = groupby_receiver.func.attr
        if grouper not in ("group_by", "group_by_dynamic", "rolling"):
            return None

        # Get the DataFrame being grouped
        df_expr = groupby_receiver.func.value
        input_frame = self._infer_expr_type(df_expr)
        if not input_frame:
            return None

        # Extract group keys
        keys: list[str] = []
        if grouper == "group_by":
            for arg in groupby_receiver.args:
                # ``group_by("a")`` and ``group_by("a", "b")`` (positional
                # strings), plus ``group_by(["a", "b"])`` / ``group_by(("a",))``
                # (list/tuple of strings) are all equivalent in polars.
                single = _str_constant(arg)
                if single is not None:
                    keys.append(single)
                    continue
                multi = _str_list_or_tuple(arg)
                if multi is not None:
                    keys.extend(multi)
        else:
            # group_by_dynamic / rolling: first positional / ``index_column`` is the time axis.
            index_col: str | None = None
            if groupby_receiver.args:
                index_col = _str_constant(groupby_receiver.args[0])
            for kw in groupby_receiver.keywords:
                if kw.arg == "index_column":
                    cand = _str_constant(kw.value)
                    if cand is not None:
                        index_col = cand
                if kw.arg == "by" or kw.arg == "group_by":
                    extra = _str_list_or_tuple(kw.value)
                    single = _str_constant(kw.value)
                    if extra is not None:
                        keys.extend(extra)
                    elif single is not None:
                        keys.append(single)
            if index_col is not None:
                keys.insert(0, index_col)

        # Extract aggregation expressions
        expr_analyzer = ExpressionAnalyzer(
            input_frame, warnings=self.warnings, registry=self.registry
        )
        agg_exprs: list[AggExpr] = []
        for arg in node.args:
            agg_expr = expr_analyzer.analyze_agg_expr(arg)
            if agg_expr:
                agg_exprs.append(agg_expr)

        # Kwarg-form ``agg(name=expr)`` — polars uses the kwarg name as the
        # output column. It overrides any ``.alias(...)`` buried in the
        # value, matching polars' own behaviour.
        for kw in node.keywords:
            if kw.arg is None:
                continue
            agg_expr = expr_analyzer.analyze_agg_expr(kw.value)
            if agg_expr:
                agg_expr.alias = kw.arg
                agg_exprs.append(agg_expr)

        self.errors.extend(expr_analyzer.errors)

        try:
            return infer_groupby_result(input_frame, keys, agg_exprs)
        except GroupByTypeError as e:
            self.errors.append(tag(PLY011, str(e)))
            return None

    def _resolve_plural_col(self, node: ast.expr) -> list[str] | None:
        """Detect ``pl.col("a", "b", ...)`` with multiple positional args.

        Single-arg calls return ``None`` so they keep using the regular
        ``analyze_select_expr`` path. Multi-arg or list-arg fans out to a
        list of names, matching the polars semantics.
        """
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, ast.Attribute):
            return None
        if node.func.attr != "col":
            return None
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "pl"):
            return None
        if len(node.args) <= 1:
            # Single arg can still be a list literal: ``pl.col(["a","b"])``.
            if len(node.args) == 1:
                multi = _str_list_or_tuple(node.args[0])
                if multi is not None and len(multi) >= 2:
                    return multi
            return None
        names: list[str] = []
        for a in node.args:
            s = _str_constant(a)
            if s is None:
                return None
            names.append(s)
        return names

    def _infer_select_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """Infer type of .select() call."""
        expr_analyzer = ExpressionAnalyzer(
            input_frame, warnings=self.warnings, registry=self.registry
        )
        result_columns: dict[str, DataType] = {}

        for arg in node.args:
            sel = _resolve_selector(arg, input_frame)
            if sel is not None:
                for c in sel:
                    spec = input_frame.columns.get(c)
                    if spec is not None:
                        result_columns[c] = spec.dtype
                continue
            plural = self._resolve_plural_col(arg)
            if plural is not None:
                for c in plural:
                    spec = input_frame.columns.get(c)
                    if spec is None:
                        # On an open frame the column may exist among the
                        # unknown extras — select it as Unknown.
                        if input_frame.rest is not None:
                            result_columns[c] = Unknown()
                            continue
                        self.errors.append(
                            tag(
                                PLY001,
                                f"Column '{c}' not found. Available columns: "
                                f"{list(input_frame.columns.keys())}",
                            )
                        )
                        continue
                    result_columns[c] = spec.dtype
                continue
            name, dtype = expr_analyzer.analyze_select_expr(arg)
            if name and dtype:
                result_columns[name] = dtype

        # Kwarg form ``select(name=expr)`` — polars treats it as
        # ``expr.alias("name")``. Same for ``with_columns``.
        for kw in node.keywords:
            if kw.arg is None:
                continue
            _, dtype = expr_analyzer.analyze_select_expr(kw.value)
            if dtype is not None:
                result_columns[kw.arg] = dtype

        self.errors.extend(expr_analyzer.errors)

        if result_columns:
            return FrameType(columns=result_columns)
        return None

    def _infer_with_columns_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """Infer type of .with_columns() call."""
        # Start with all existing columns
        result_columns: dict[str, ColumnSpec | DataType] = dict(input_frame.columns)

        expr_analyzer = ExpressionAnalyzer(
            input_frame, warnings=self.warnings, registry=self.registry
        )

        for arg in node.args:
            sel = _resolve_selector(arg, input_frame)
            if sel is not None:
                # cs.* selectors in with_columns are a no-op type-wise (re-include
                # existing columns). Skip — keep the existing entries.
                continue
            plural = self._resolve_plural_col(arg)
            if plural is not None:
                # ``with_columns(pl.col("a", "b"))`` is equivalent to keeping
                # the existing columns; nothing to add. On an open frame a
                # missing name may exist among the unknown extras — no error.
                for c in plural:
                    if c not in input_frame.columns and input_frame.rest is None:
                        self.errors.append(
                            tag(
                                PLY001,
                                f"Column '{c}' not found. Available columns: "
                                f"{list(input_frame.columns.keys())}",
                            )
                        )
                continue
            name, dtype = expr_analyzer.analyze_select_expr(arg)
            if name and dtype:
                result_columns[name] = dtype

        # Kwarg form ``with_columns(name=expr)`` — polars treats it as
        # ``expr.alias("name")``.
        for kw in node.keywords:
            if kw.arg is None:
                continue
            _, dtype = expr_analyzer.analyze_select_expr(kw.value)
            if dtype is not None:
                result_columns[kw.arg] = dtype

        self.errors.extend(expr_analyzer.errors)

        return FrameType(
            columns=result_columns,
            strict=input_frame.strict,
            rest=input_frame.rest,
        )

    # -- frame methods --------------------------------------------------

    def _collect_drop_targets(
        self, node: ast.Call, input_frame: FrameType | None = None
    ) -> list[str]:
        """Resolve column-name args for ``drop`` / ``drop_nulls(subset=...)``.

        Supports ``drop("a", "b")``, ``drop(["a", "b"])``, and ``drop(cs.numeric())``
        when ``input_frame`` is supplied (selectors need a frame to resolve).
        Returns column names in argument order (lists / selectors flattened in).
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
                continue
            if input_frame is not None:
                sel = _resolve_selector(arg, input_frame)
                if sel is not None:
                    names.extend(sel)
        return names

    def _infer_drop_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        targets = self._collect_drop_targets(node, input_frame)
        result_columns = dict(input_frame.columns)
        for name in targets:
            if name not in result_columns:
                # On an open frame the column may exist among the unknown
                # extras — dropping it is a no-op for the tracked schema.
                if input_frame.rest is None:
                    self.errors.append(tag(PLY002, f"drop: column '{name}' not found"))
                continue
            del result_columns[name]
        return FrameType(columns=result_columns, strict=input_frame.strict, rest=input_frame.rest)

    def _infer_rename_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        if not node.args or not isinstance(node.args[0], ast.Dict):
            return input_frame
        mapping_node = node.args[0]
        mapping: dict[str, str] = {}
        for key_node, val_node in zip(mapping_node.keys, mapping_node.values, strict=False):
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
                self.errors.append(tag(PLY003, f"rename: column '{old}' not found"))
        return FrameType(columns=result_columns, strict=input_frame.strict, rest=input_frame.rest)

    def _infer_cast_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        if not node.args:
            return input_frame
        first = node.args[0]
        if not isinstance(first, ast.Dict):
            # ``cast(pl.Int64)`` whole-frame form not handled — fall back to identity.
            return input_frame
        result_columns: dict[str, ColumnSpec] = dict(input_frame.columns)
        for key_node, val_node in zip(first.keys, first.values, strict=False):
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
                self.errors.append(tag(PLY004, f"cast: column '{col}' not found"))
                continue
            result_columns[col] = ColumnSpec(
                dtype=_wrap_like(spec.dtype, target),
                required=spec.required,
            )
        return FrameType(columns=result_columns, strict=input_frame.strict, rest=input_frame.rest)

    def _infer_drop_nulls_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        # subset can be passed positionally or as keyword
        subset: list[str] | None = None
        if node.args:
            single = _str_constant(node.args[0])
            cand = _str_list_or_tuple(node.args[0]) or (None if single is None else [single])
            if cand is not None:
                subset = cand
        for kw in node.keywords:
            if kw.arg == "subset":
                single_kw = _str_constant(kw.value)
                cand2 = _str_list_or_tuple(kw.value) or (None if single_kw is None else [single_kw])
                if cand2 is not None:
                    subset = cand2

        targets = subset if subset is not None else list(input_frame.columns.keys())
        result_columns: dict[str, ColumnSpec] = {}
        for col_name, spec in input_frame.columns.items():
            if col_name in targets:
                if col_name not in input_frame.columns and subset is not None:
                    self.errors.append(tag(PLY005, f"drop_nulls: column '{col_name}' not found"))
                inner = spec.dtype.inner if isinstance(spec.dtype, Nullable) else spec.dtype
                result_columns[col_name] = ColumnSpec(dtype=inner, required=spec.required)
            else:
                result_columns[col_name] = spec
        if subset is not None:
            for s in subset:
                if s not in input_frame.columns:
                    self.errors.append(tag(PLY005, f"drop_nulls: column '{s}' not found"))
        return FrameType(columns=result_columns, strict=input_frame.strict, rest=input_frame.rest)

    def _collect_concat_frames(self, list_node: ast.expr) -> list[FrameType] | None:
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

    def _infer_concat_call(self, node: ast.Call) -> FrameType | None:
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
            self.errors.append(tag(PLY020, str(e)))
            return None
        # Unsupported how — treat as vertical with a warning.
        try:
            return concat_vertical(frames)
        except ReshapeError as e:
            self.errors.append(tag(PLY020, str(e)))
            return None

    def _infer_vstack_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        if not node.args:
            return input_frame
        other = self._infer_expr_type(node.args[0])
        if other is None:
            return input_frame
        try:
            return concat_vertical([input_frame, other])
        except ReshapeError as e:
            self.errors.append(tag(PLY020, str(e)))
            return None

    def _infer_hstack_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        if not node.args:
            return input_frame
        other = self._infer_expr_type(node.args[0])
        if other is None:
            return input_frame
        try:
            return concat_horizontal([input_frame, other])
        except ReshapeError as e:
            self.errors.append(tag(PLY020, str(e)))
            return None

    def _infer_explode_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
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
                # On an open frame the column may exist among the unknown
                # extras — treat it as present, exploding to Unknown.
                if input_frame.rest is not None:
                    result_columns[col] = ColumnSpec(dtype=Unknown())
                    continue
                self.errors.append(tag(PLY021, f"explode: column '{col}' not found"))
                continue
            inner = spec.dtype
            outer_nullable = isinstance(inner, Nullable)
            if outer_nullable:
                inner = inner.inner  # type: ignore[union-attr]
            if isinstance(inner, Unknown):
                # An Unknown column might hold lists — exploding it yields
                # elements of an unknown dtype, never an error.
                unknown_elem: DataType = Unknown()
                if outer_nullable:
                    unknown_elem = Nullable(unknown_elem)
                result_columns[col] = ColumnSpec(dtype=unknown_elem, required=spec.required)
                continue
            if not isinstance(inner, ListT):
                self.errors.append(
                    tag(PLY021, f"explode: column '{col}' is {spec.dtype}, not List[T]")
                )
                continue
            elem_dtype: DataType = inner.inner
            if outer_nullable:
                elem_dtype = Nullable(elem_dtype)
            result_columns[col] = ColumnSpec(dtype=elem_dtype, required=spec.required)
        return FrameType(columns=result_columns, strict=input_frame.strict, rest=input_frame.rest)

    def _infer_unpivot_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
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
            self.errors.append(tag(PLY022, str(e)))
            return None

    def _infer_pivot_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """``df.pivot(on=..., index=..., values=...)``.

        The output schema depends on the *runtime values* of the ``on``
        column, which polypolarism cannot see. Instead of guessing, we emit
        ``PLW005`` with a copy-pasteable annotation suggestion built from
        the call site. Users who assign the result to a
        ``DataFrame[Schema]``-annotated variable get the schema applied
        through the existing ``AnnAssign`` path.
        """
        index_cols: list[str] = []
        on_col: str | None = None
        values_col: str | None = None
        for kw in node.keywords:
            if kw.arg == "index":
                lst = _str_list_or_tuple(kw.value)
                single = _str_constant(kw.value)
                if lst is not None:
                    index_cols = lst
                elif single is not None:
                    index_cols = [single]
            elif kw.arg == "on":
                cand = _str_constant(kw.value)
                if cand is not None:
                    on_col = cand
            elif kw.arg == "values":
                cand = _str_constant(kw.value)
                if cand is not None:
                    values_col = cand

        # Build a minimally helpful annotation hint when the call shape is
        # readable. Otherwise just emit the generic warning.
        hint = ""
        if index_cols and values_col is not None:
            value_dtype = input_frame.get_column_type(values_col)
            value_str = str(value_dtype) if value_dtype is not None else "T"
            index_lines = []
            for c in index_cols:
                idx_dtype = input_frame.get_column_type(c)
                idx_str = str(idx_dtype) if idx_dtype is not None else "T"
                index_lines.append(f"{c}: pl.{idx_str}")
            on_label = f"each value of '{on_col}'" if on_col else "each pivoted column"
            hint = (
                " Suggested annotation:\n"
                "      class PivotedOut(pa.DataFrameModel):\n"
                + "".join(f"          {line}\n" for line in index_lines)
                + f"          # one column per {on_label}, dtype pl.{value_str}\n"
                "      result: DataFrame[PivotedOut] = df.pivot(...)"
            )

        self.warnings.append(
            tag(
                PLW005,
                "pivot: output schema depends on the runtime values of the "
                "`on` column, so polypolarism cannot infer it. Assign the "
                "result to a `DataFrame[Schema]`-annotated variable to give "
                "the analyser a schema to check against." + hint,
            )
        )
        return None

    def _infer_unnest_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """``df.unnest("s")`` / ``unnest(["a","b"])`` flattens Struct columns."""
        targets: list[str] = []
        for arg in node.args:
            single = _str_constant(arg)
            multi = _str_list_or_tuple(arg)
            if single is not None:
                targets.append(single)
            elif multi is not None:
                targets.extend(multi)
        if not targets:
            return input_frame

        result_columns: dict[str, ColumnSpec] = dict(input_frame.columns)
        result_rest = input_frame.rest
        for col in targets:
            spec = result_columns.get(col)
            if spec is None:
                # On an open frame the column may exist among the unknown
                # extras — its fields stay unknown, the frame stays open.
                if input_frame.rest is None:
                    self.errors.append(tag(PLY021, f"unnest: column '{col}' not found"))
                continue
            inner = spec.dtype
            outer_nullable = isinstance(inner, Nullable)
            if outer_nullable:
                inner = inner.inner  # type: ignore[union-attr]
            if isinstance(inner, Unknown):
                # Unnesting a struct whose fields we can't see: the column
                # disappears and an unknown set of field columns appears —
                # the result is an open frame.
                del result_columns[col]
                result_rest = RowVar("unnest")
                continue
            if not isinstance(inner, Struct):
                self.errors.append(
                    tag(PLY021, f"unnest: column '{col}' is {spec.dtype}, not Struct{{...}}")
                )
                continue
            del result_columns[col]
            for field_name, field_dtype in inner.fields.items():
                wrapped: DataType = field_dtype
                if outer_nullable and not isinstance(wrapped, Nullable):
                    wrapped = Nullable(wrapped)
                result_columns[field_name] = ColumnSpec(dtype=wrapped, required=spec.required)
        return FrameType(columns=result_columns, strict=input_frame.strict, rest=result_rest)

    def _infer_filter_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """Identity-typed, but walk every predicate sub-expression to validate columns."""
        expr_analyzer = ExpressionAnalyzer(
            input_frame, warnings=self.warnings, registry=self.registry
        )
        for arg in node.args:
            expr_analyzer.analyze_select_expr(arg)
        for kw in node.keywords:
            expr_analyzer.analyze_select_expr(kw.value)
        self.errors.extend(expr_analyzer.errors)
        return input_frame

    def _infer_with_row_index_call(
        self, input_frame: FrameType, node: ast.Call
    ) -> FrameType | None:
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
                self.errors.append(tag(PLY006, f"with_row_index: column '{name}' already exists"))
                continue
            result_columns[col_name] = spec
        return FrameType(columns=result_columns, strict=input_frame.strict, rest=input_frame.rest)


def _extract_function_signature(
    func_node: ast.FunctionDef,
    schema_registry: SchemaRegistry,
) -> FunctionSignature | None:
    """Extract type signature from a function definition."""
    parameters: dict[str, tuple[int, FrameType]] = {}
    return_type: FrameType | None = None

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
    registry: FunctionRegistry | None = None,
    schema_registry: SchemaRegistry | None = None,
    class_registry: ClassRegistry | None = None,
    current_class_name: str | None = None,
) -> FunctionAnalysis | None:
    """Analyze a single function definition."""
    schema_registry = schema_registry or SchemaRegistry()
    class_registry = class_registry or ClassRegistry()
    input_types: dict[str, FrameType] = {}
    declared_return: FrameType | None = None
    errors: list[str] = []
    warnings: list[str] = []
    has_df_annotation = False
    unresolved_schemas: list[str] = []

    # Extract input parameter types
    for arg in func_node.args.args:
        if arg.annotation:
            if _annotation_declares_frame(arg.annotation, schema_registry):
                has_df_annotation = True
                frame_type, parse_error = _resolve_declared_type(arg.annotation, schema_registry)
                if frame_type is not None:
                    input_types[arg.arg] = frame_type
                elif parse_error:
                    errors.append(f"Parameter '{arg.arg}': {parse_error}")
            else:
                schema_name = frame_annotation_schema_name(arg.annotation)
                if schema_name is not None:
                    has_df_annotation = True
                    unresolved_schemas.append(schema_name)

    # Extract return type
    if func_node.returns:
        if _annotation_declares_frame(func_node.returns, schema_registry):
            has_df_annotation = True
            declared_return, parse_error = _resolve_declared_type(
                func_node.returns, schema_registry
            )
            if parse_error:
                errors.append(f"Return type: {parse_error}")
        else:
            schema_name = frame_annotation_schema_name(func_node.returns)
            if schema_name is not None:
                has_df_annotation = True
                unresolved_schemas.append(schema_name)

    # If no DataFrame annotations found, skip this function
    if not has_df_annotation:
        return None

    # Surface unresolved-schema names so the user sees the file isn't
    # being silently skipped because of a missing import. Deduplicate to
    # one warning per name.
    for name in dict.fromkeys(unresolved_schemas):
        warnings.append(
            tag(
                PLW006,
                f"schema '{name}' referenced in annotation but not found. "
                f"Define it in this module, or import it from a project-local "
                f"module via `from <module> import {name}`. "
                f"Stdlib/third-party imports aren't followed.",
            )
        )

    # Analyze function body with registry
    body_analyzer = FunctionBodyAnalyzer(
        input_types,
        errors,
        registry,
        schema_registry,
        warnings=warnings,
        class_registry=class_registry,
        current_class_name=current_class_name,
    )
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
        warnings=body_analyzer.warnings,
    )


def analyze_source(source: str, file_path: Path | None = None) -> list[FunctionAnalysis]:
    """
    Analyze Python source code for DataFrame type annotations.

    Uses a 3-pass approach:
    1. Collect all function AST nodes
    2. Build registry with signatures (typed) and nodes (all)
    3. Analyze function bodies with registry for call resolution

    Args:
        source: Python source code as a string
        file_path: Optional path of the file the source came from.
            When provided, ``from <module> import <Schema>`` statements
            are followed on disk so schemas defined in sibling/related
            project files are resolvable. Without it, only schemas
            defined in ``source`` itself are visible (legacy behaviour
            for tests that pass raw strings).

    Returns:
        List of FunctionAnalysis results for functions with
        ``DataFrame[Schema]`` / ``LazyFrame[Schema]`` annotations
    """
    tree = ast.parse(source)

    # Pass 0: Collect Pandera DataFrameModel schemas — from this module
    # plus any project-local modules it imports from.
    if file_path is not None:
        schema_registry = collect_schemas_with_imports(tree, file_path)
    else:
        schema_registry = collect_schemas(tree)

    # Pass 1: Collect functions, separating module-level from class methods.
    # ``func_to_class`` maps each function node's ``id()`` to the name of
    # its enclosing class (or ``None`` for module-level), so during pass 3
    # we know which class context to give each method analyser.
    module_funcs: list[ast.FunctionDef] = []
    class_methods: list[tuple[str, ast.FunctionDef]] = []
    func_to_class: dict[int, str | None] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            module_funcs.append(node)
            func_to_class[id(node)] = None
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    class_methods.append((node.name, stmt))
                    func_to_class[id(stmt)] = node.name

    # Pass 2: Build the function and class registries.
    registry = FunctionRegistry()
    class_registry = ClassRegistry()
    for func_node in module_funcs:
        signature = _extract_function_signature(func_node, schema_registry)
        info = FunctionInfo(
            name=func_node.name,
            node=func_node,
            signature=signature,
            inferred_returns={},
        )
        registry.register(info)
    for class_name, func_node in class_methods:
        signature = _extract_function_signature(func_node, schema_registry)
        info = FunctionInfo(
            name=func_node.name,
            node=func_node,
            signature=signature,
            inferred_returns={},
        )
        class_registry.register_method(class_name, info)

    # Pass 3: Analyze each function/method body with the registries.
    func_nodes = module_funcs + [m for _, m in class_methods]
    results: list[FunctionAnalysis] = []
    for func_node in func_nodes:
        analysis = analyze_function(
            func_node,
            registry,
            schema_registry,
            class_registry=class_registry,
            current_class_name=func_to_class.get(id(func_node)),
        )
        if analysis:
            results.append(analysis)

    return results
