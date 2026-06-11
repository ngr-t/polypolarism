"""AST analysis and data flow tracking."""

from __future__ import annotations

import ast
import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from polypolarism.compat.polars_api import (
    AGG_SHORTHAND_NAMES,
    ARR_NAMESPACE_ELEMENT_RETURN,
    ARR_NAMESPACE_PRESERVING,
    ARR_NAMESPACE_RETURN,
    ARR_NAMESPACE_TO_LIST,
    BIN_NAMESPACE_RETURN,
    CAT_NAMESPACE_RETURN,
    CONTAINER_AGG_METHODS,
    DECIMAL_DEFAULT,
    DT_NAMESPACE_PRESERVING,
    DT_NAMESPACE_RETURN,
    DTYPE_NAME_MAP,
    EAGER_FRAME_RETURNING_METHODS,
    EAGER_ONLY_METHODS,
    EAGER_READ_FUNCTIONS,
    IDENTITY_FRAME_METHODS,
    LAZY_FRAME_RETURNING_METHODS,
    LAZY_ONLY_METHODS,
    LAZY_SCAN_FUNCTIONS,
    LIST_NAMESPACE_ELEMENT_RETURN,
    LIST_NAMESPACE_PRESERVING,
    STR_NAMESPACE_RETURN,
    TIME_UNITS,
    ContainerAggInvalid,
    agg_function_for,
    canonicalize_method,
    coarser_time_unit,
    container_agg_return,
    parse_array_shape,
    parse_datetime_call,
    parse_decimal_call,
    parse_duration_call,
    parse_enum_call,
    time_unit_refines,
)
from polypolarism.diagnostics import (
    PLW001,
    PLW002,
    PLW003,
    PLW004,
    PLW005,
    PLW006,
    PLW007,
    PLW008,
    PLW011,
    PLY001,
    PLY002,
    PLY003,
    PLY004,
    PLY005,
    PLY006,
    PLY007,
    PLY008,
    PLY009,
    PLY010,
    PLY011,
    PLY012,
    PLY013,
    PLY014,
    PLY015,
    PLY016,
    PLY017,
    PLY020,
    PLY021,
    PLY022,
    PLY030,
    PLY031,
    PLY032,
    PLY033,
    PLY041,
    PLY042,
    tag,
)
from polypolarism.expr_infer import (
    ColumnNotFoundError,
    TypePromotionError,
    TypeUnificationError,
    infer_col,
    infer_lit,
    infer_shift_fill,
    promote_types,
    supertype,
    unify_types,
)
from polypolarism.ops.groupby import (
    AggExpr,
    AggFunction,
    GroupByTypeError,
    grouped_agg_panics,
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
    bare_frame_annotation,
    extract_dataframe_annotation,
    frame_annotation_schema_name,
)
from polypolarism.pandera_schema import (
    SchemaRegistry,
    collect_schemas,
    collect_schemas_with_imports,
)
from polypolarism.types import (
    NUMERIC_DTYPES,
    Array,
    Binary,
    Boolean,
    Categorical,
    ColumnSpec,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float16,
    Float32,
    Float64,
    FrameType,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Null,
    Nullable,
    RowVar,
    Struct,
    Time,
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
_EAGER_FRAME_RETURNING_METHODS = EAGER_FRAME_RETURNING_METHODS
_LAZY_FRAME_RETURNING_METHODS = LAZY_FRAME_RETURNING_METHODS


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
            # ``pl.List(pl.Int64)`` — resolve the element dtype recursively;
            # an unresolvable element degrades to List[Unknown].
            if node.func.attr == "List" and node.args:
                element = _resolve_pl_dtype(node.args[0])
                return ListT(element if element is not None else Unknown())
            # ``pl.Array(pl.Int64, 3)`` — same recursion; the width is
            # tracked (backlog C-7), non-literal/multi-dim shapes -> None.
            if node.func.attr == "Array" and node.args:
                element = _resolve_pl_dtype(node.args[0])
                return Array(element if element is not None else Unknown(), parse_array_shape(node))
            # ``pl.Decimal(p, s)`` preserves precision/scale (issue #38);
            # omitted args take polars' defaults. Non-literal args are
            # unknowable — return unresolved (None) rather than fabricating
            # the bare default, which would be a false-positive trap.
            if node.func.attr == "Decimal":
                return parse_decimal_call(node)
            # ``pl.Datetime("us", "UTC")`` preserves the time unit and zone
            # (shared parser in compat; issues #50/#66). A non-literal
            # argument is unknowable — unresolved (None), like Decimal above.
            if node.func.attr == "Datetime":
                return parse_datetime_call(node)
            # ``pl.Duration("ms")`` preserves the time unit (issue #66).
            if node.func.attr == "Duration":
                return parse_duration_call(node)
            # ``pl.Enum(["a", "b"])`` preserves the ordered category tuple
            # (issue #67). A non-literal category list still provably
            # constructs SOME Enum — categories=None, the checker wildcard.
            if node.func.attr == "Enum":
                return parse_enum_call(node)
            return _PL_DTYPE_NAME_MAP.get(node.func.attr)
    return None


# Python builtin type names accepted as dtypes in ``schema=`` dicts
# (``pl.DataFrame({...}, schema={"a": int})``).
_PY_BUILTIN_DTYPE_MAP: dict[str, DataType] = {
    "int": Int64(),
    "float": Float64(),
    "str": Utf8(),
    "bool": Boolean(),
}


# Eager range constructors usable as ``pl.DataFrame({...})`` dict values:
# ``pl.<name>(..., eager=True)`` produces a Series of a fixed dtype.
# ``datetime_range`` is argument-dependent (tz / unit) and resolved by the
# dedicated branch in ``_frame_literal_value_dtype``; the entry documents
# the no-argument default.
_RANGE_CONSTRUCTOR_DTYPES: dict[str, DataType] = {
    "date_range": Date(),
    "datetime_range": Datetime(),
    "time_range": Time(),
    "int_range": Int64(),
}


def _datetime_range_unit(node: ast.Call) -> str | None:
    """Time unit of ``pl.datetime_range(...)`` (issue #66).

    The ``time_unit=`` keyword wins; otherwise polars derives ns from an
    interval string containing "ns" and defaults to us (probed 1.41.2:
    ``interval="1500ns"`` -> ns; ``"1us"`` / ``"1h"`` / the default
    ``"1d"`` -> us). ``None`` means a non-literal argument made the unit
    statically unknowable.
    """
    for kw in node.keywords:
        if kw.arg == "time_unit":
            if isinstance(kw.value, ast.Constant):
                if kw.value.value is None:
                    break  # explicit None — the interval decides
                if isinstance(kw.value.value, str) and kw.value.value in TIME_UNITS:
                    return kw.value.value
            return None
    interval_node: ast.expr | None = node.args[2] if len(node.args) > 2 else None
    for kw in node.keywords:
        if kw.arg == "interval":
            interval_node = kw.value
    if interval_node is None:
        return "us"
    interval = _str_constant(interval_node)
    if interval is None:
        return None  # a variable / timedelta object — unknowable
    return "ns" if "ns" in interval else "us"


def _resolve_schema_dtype(node: ast.expr) -> DataType:
    """Resolve one dtype value of a ``schema=`` / ``schema_overrides=`` dict.

    Accepts the polars dtype literals handled by ``_resolve_pl_dtype`` plus
    the python builtins ``int``/``float``/``str``/``bool``. Anything else
    (variables, exotic dtypes) degrades to ``Unknown`` — the column still
    registers under its declared name.
    """
    resolved = _resolve_pl_dtype(node)
    if resolved is not None:
        return resolved
    if isinstance(node, ast.Name):
        builtin = _PY_BUILTIN_DTYPE_MAP.get(node.id)
        if builtin is not None:
            return builtin
    return Unknown()


def _unify_literal_values(values: list[object]) -> DataType:
    """Fold ``infer_lit`` over python literal values with ``unify_types``.

    ``[1, None]`` → ``Nullable[Int64]``; empty, non-literal or
    non-unifiable values → ``Unknown``.
    """
    if not values:
        return Unknown()
    unified: DataType | None = None
    for value in values:
        if value is not None and not isinstance(value, (bool, int, float, str, bytes)):
            return Unknown()
        elem = infer_lit(value)
        if unified is None:
            unified = elem
            continue
        try:
            unified = unify_types(unified, elem)
        except TypeUnificationError:
            return Unknown()
    assert unified is not None
    return unified


def _frame_literal_value_dtype(
    node: ast.expr,
    lookup_const: Callable[[str], str | list[str] | int | None] | None = None,
) -> DataType:
    """Per-column dtype for one value of a ``pl.DataFrame({...})`` data dict.

    - list/tuple of constants → element dtypes folded with ``unify_types``
      (``[1, None]`` → ``Nullable[Int64]``); empty or any non-constant /
      non-unifiable element → ``Unknown``.
    - a ``Name`` bound to a string(-list) constant (issue #39): resolved
      through ``lookup_const`` — a ``list[str]`` value behaves like the
      equivalent list literal, a ``str`` like the broadcast scalar. Other
      constant kinds (ints, recorded for backlog B-5) stay ``Unknown``.
    - recognised eager range constructors (``pl.date_range(...)`` etc.)
      → their fixed Series dtype.
    - a bare scalar constant (broadcast) → ``infer_lit``.
    - anything else → ``Unknown``.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        if not all(isinstance(elt, ast.Constant) for elt in node.elts):
            return Unknown()
        return _unify_literal_values(
            [elt.value for elt in node.elts if isinstance(elt, ast.Constant)]
        )

    if isinstance(node, ast.Name) and lookup_const is not None:
        const_val = lookup_const(node.id)
        if isinstance(const_val, str):
            return infer_lit(const_val)
        if isinstance(const_val, list):
            return _unify_literal_values(list(const_val))
        return Unknown()

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
    ):
        range_dtype = _RANGE_CONSTRUCTOR_DTYPES.get(node.func.attr)
        if range_dtype is not None:
            # ``pl.datetime_range(..., time_zone="UTC", eager=True)`` is
            # probed to yield a tz-aware Series (issue #50 collateral),
            # with the unit from ``time_unit=`` / the interval (issue #66).
            # A non-literal time_zone / time_unit / interval is unknowable
            # -> Unknown.
            if node.func.attr == "datetime_range":
                tz: str | None = None
                for kw in node.keywords:
                    if kw.arg == "time_zone":
                        if isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, str):
                                tz = kw.value.value
                                continue
                            if kw.value.value is None:
                                continue
                        return Unknown()
                unit = _datetime_range_unit(node)
                if unit is None:
                    return Unknown()
                return Datetime(tz=tz, unit=unit)
            return range_dtype

    if isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (bool, int, float, str, bytes))
    ):
        return infer_lit(node.value)

    return Unknown()


def _base_is_unknown(dtype: DataType) -> bool:
    """True when the dtype (after Nullable unwrap) is Unknown."""
    inner = dtype.inner if isinstance(dtype, Nullable) else dtype
    return isinstance(inner, Unknown)


def _unmodeled_method_warning(call_desc: str, *, frame: bool = False) -> str:
    """PLW007 text for an unmodeled method call (backlog B-4 / N-3).

    Emitted only when the receiver was precisely known — the call is the
    exact point where inference gives up and downstream checks weaken.
    The ``frame`` variant covers frame-level methods, where the whole
    variable untracks (and a ``.cast(...)`` cannot repair it — only a
    schema validation can).
    """
    if frame:
        return tag(
            PLW007,
            f"`{call_desc}` is not modeled by polypolarism — the frame's "
            f"schema is no longer tracked and downstream checks go silent. "
            f"Validate the result against a schema "
            f"(`Schema.validate(...)`) to keep checking precise.",
        )
    return tag(
        PLW007,
        f"`{call_desc}` is not modeled by polypolarism — the result dtype "
        f"degrades to Unknown and downstream dtype checks weaken. Pin the "
        f"dtype with `.cast(...)` after the call, or validate the frame "
        f"against a schema, to keep checking precise.",
    )


def _wrap_nullable_if_any(result: DataType, operands: list[DataType]) -> DataType:
    """Wrap ``result`` in Nullable when any operand is Nullable (or Null).

    Elementwise operations propagate nulls: if any input value can be
    null, the output value can be null (``null > 0`` is null, ``1 + null``
    is null, ...). Already-Nullable results are returned unchanged.
    """
    if isinstance(result, Nullable):
        return result
    if any(isinstance(t, (Nullable, Null)) for t in operands):
        return Nullable(result)
    return result


# ---- binary-arithmetic dtype rules (issue #30) ------------------------------
#
# Classification of ``left <op> right`` over the dtypes whose arithmetic we
# fully understand: numeric ∪ {Utf8, Boolean, Date, Datetime, Time, Duration}
# plus a dedicated Decimal arm (issue #52, ``_decimal_arith``). Verified
# against polars 1.41.2 by driving the full dtype x op product through
# ``df.select`` (table in TestArithmeticIncompatibleDtypes). Anything outside
# this set keeps the legacy silent fallback — false positives are worse than
# false negatives here.

# Sentinel: both operand bases are understood and polars rejects the pair+op
# at runtime (InvalidOperationError) — report PLY009.
_ARITH_INVALID = object()

_OP_SYMBOLS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
}


def _arith_category(dtype: DataType) -> str | None:
    """Category of a (Nullable-unwrapped) dtype for arithmetic classification.

    ``None`` means the dtype is outside the fully-understood set (Unknown,
    Categorical, List, ...) and the caller must stay silent. Decimal is
    resolved by ``_decimal_arith`` before this classification runs.
    """
    if type(dtype) in NUMERIC_DTYPES:
        return "num"
    if isinstance(dtype, Utf8):
        return "str"
    if isinstance(dtype, Boolean):
        return "bool"
    if isinstance(dtype, Date):
        return "date"
    if isinstance(dtype, Datetime):
        return "datetime"
    if isinstance(dtype, Time):
        return "time"
    if isinstance(dtype, Duration):
        return "dur"
    return None


def _promote_or_none(left: DataType, right: DataType) -> DataType | None:
    """``promote_types`` that degrades to the legacy fallback instead of raising.

    ``expr_infer.promote_types`` only knows Int32/Int64/Float32/Float64;
    exotic numerics (Int8, UInt*, Float16, ...) raise TypePromotionError.
    Those are still *valid* polars arithmetic, so return ``None`` to let the
    caller keep the historical keep-left-dtype behaviour.
    """
    try:
        return promote_types(left, right)
    except TypePromotionError:
        return None


def _numeric_arith(
    op: ast.operator, left: DataType, lcat: str, right: DataType, rcat: str
) -> DataType | object | None:
    """Arithmetic where both operands are numeric or Boolean."""
    if isinstance(op, ast.Div):
        # True division always yields Float64, bool operands included.
        return Float64()
    if isinstance(op, ast.Pow):
        # ``**`` rejects bool as base or exponent.
        if "bool" in (lcat, rcat):
            return _ARITH_INVALID
        return _promote_or_none(left, right)
    if not isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        return None  # << >> @ ... — not polars expression arithmetic.
    if lcat == "bool" and rcat == "bool":
        # Only ``+`` is defined on a pair of Booleans (count of trues).
        return UInt32() if isinstance(op, ast.Add) else _ARITH_INVALID
    if lcat == "bool":
        return right  # bool acts as an int — numeric operand's dtype wins
    if rcat == "bool":
        return left
    return _promote_or_none(left, right)


def _datetime_plus_duration(dt: Datetime, dur: Duration) -> Datetime:
    """``Datetime ± Duration`` keeps the Datetime's tz; mixed units resolve
    to the coarser one (probed: Datetime[us] + Duration[ms] -> Datetime[ms],
    every unit pair, both operand orders)."""
    return Datetime(tz=dt.tz, unit=coarser_time_unit(dt.unit, dur.unit))


def _temporal_arith(
    op: ast.operator, left: DataType, lcat: str, right: DataType, rcat: str
) -> DataType | object | None:
    """Arithmetic where at least one operand is Date/Datetime/Time/Duration.

    The non-temporal side can only be numeric or Boolean here (string pairs
    are resolved before this is called). Time units (issue #66) follow the
    probed mixed-unit rule: the coarser operand unit wins; ``Date`` carries
    no unit and adopts the other operand's (Date - Date is ``us``,
    Time - Time is ``ns`` — both probed on 1.41.2).
    """
    if isinstance(op, ast.Add):
        if lcat == "date" and rcat == "dur" or lcat == "dur" and rcat == "date":
            return Date()
        if lcat == "datetime" and rcat == "dur":
            assert isinstance(left, Datetime) and isinstance(right, Duration)
            return _datetime_plus_duration(left, right)
        if lcat == "dur" and rcat == "datetime":
            assert isinstance(left, Duration) and isinstance(right, Datetime)
            return _datetime_plus_duration(right, left)
        if lcat == "dur" and rcat == "dur":
            assert isinstance(left, Duration) and isinstance(right, Duration)
            return Duration(unit=coarser_time_unit(left.unit, right.unit))
        return _ARITH_INVALID  # Time+Duration errors too (verified)
    if isinstance(op, ast.Sub):
        if lcat in ("date", "datetime") and rcat in ("date", "datetime"):
            # Two Datetimes must agree on tz (probed: aware - naive and
            # UTC - Asia/Tokyo both raise SchemaError; issue #50). Date vs
            # tz-aware Datetime is probed-valid in either order.
            if isinstance(left, Datetime) and isinstance(right, Datetime):
                if left.tz != right.tz:
                    return _ARITH_INVALID
                return Duration(unit=coarser_time_unit(left.unit, right.unit))
            # Date x Datetime[u] (either order) -> Duration[u]; Date x Date
            # -> Duration[us] (both probed).
            if isinstance(left, Datetime):
                return Duration(unit=left.unit)
            if isinstance(right, Datetime):
                return Duration(unit=right.unit)
            return Duration(unit="us")
        if lcat == "time" and rcat == "time":
            return Duration(unit="ns")  # Time is ns-based (probed)
        if lcat == "date" and rcat == "dur":
            return Date()
        if lcat == "datetime" and rcat == "dur":
            assert isinstance(left, Datetime) and isinstance(right, Duration)
            return _datetime_plus_duration(left, right)
        if lcat == "dur" and rcat == "dur":
            assert isinstance(left, Duration) and isinstance(right, Duration)
            return Duration(unit=coarser_time_unit(left.unit, right.unit))
        return _ARITH_INVALID  # Duration-Date, Time-Duration, ... all error
    if isinstance(op, ast.Mult):
        if (lcat, rcat) in (("dur", "num"), ("num", "dur")):
            return left if lcat == "dur" else right  # unit preserved (probed)
        return _ARITH_INVALID  # Duration*bool and Duration*Duration error
    if isinstance(op, ast.Div):
        if lcat == "dur" and rcat == "num":
            return left  # unit preserved (probed)
        if lcat == "dur" and rcat == "dur":
            return Float64()
        return _ARITH_INVALID  # num/Duration, Duration/bool, Date/... error
    if isinstance(op, (ast.FloorDiv, ast.Mod, ast.Pow)):
        # No temporal operand supports // % ** — even Duration // int errors.
        return _ARITH_INVALID
    return None


# Float widths for the Decimal arm — every width (incl. Float16) was probed
# to behave identically against Decimal.
_DECIMAL_FLOAT_PARTNERS = (Float16, Float32, Float64)

# The four operators that keep a Decimal result against Decimal/int/Null.
_DECIMAL_KEEPING_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _decimal_arith(op: ast.operator, left: DataType, right: DataType) -> DataType | object | None:
    """Arithmetic where at least one Nullable-unwrapped operand is Decimal.

    Probed on polars 1.41.2 over Decimal x {Decimal, every int width incl.
    128-bit, every float width incl. Float16, Boolean, Utf8, Null literal,
    Date/Datetime/Time/Duration, Categorical, Enum, List} with all seven
    operators in both operand orders (issue #52). Eager results and
    ``LazyFrame.collect()`` agree on every cell; the lazy
    ``collect_schema()`` reports a stale pre-growth precision for
    Decimal x int and Decimal x Null cells — polypolarism claims the
    materialized dtype.
    """
    if not isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
        return None  # << >> @ ... — not polars expression arithmetic.
    if isinstance(left, Decimal) and isinstance(right, Decimal):
        if isinstance(op, _DECIMAL_KEEPING_OPS):
            # Precision saturates to 38 whatever the input precisions; the
            # scale is max(ls, rs). Probed: ``*`` does NOT add scales and
            # ``/`` stays Decimal (polars rescales: 1.00 / 3.00 -> 0.33).
            return Decimal(38, max(left.scale, right.scale))
        return _ARITH_INVALID  # // % ** raise InvalidOperationError
    if isinstance(left, Decimal):
        dec, other, dec_is_left = left, right, True
    elif isinstance(right, Decimal):
        dec, other, dec_is_left = right, left, False
    else:  # pragma: no cover — caller guarantees a Decimal operand
        return None
    if isinstance(other, _DECIMAL_FLOAT_PARTNERS):
        # Probed: every float width, either order -> Float64 for all
        # operators except ** (Decimal cannot be a pow base or exponent).
        return _ARITH_INVALID if isinstance(op, ast.Pow) else Float64()
    if type(other) in NUMERIC_DTYPES or isinstance(other, Null):
        # Integer partner (every signed/unsigned width, either order; int
        # literals behave like Int64 columns) and the all-null literal:
        # + - * / widen the precision to 38 and keep the Decimal's scale;
        # // % ** raise InvalidOperationError. A Null operand makes the
        # caller wrap the result in Nullable.
        if isinstance(op, _DECIMAL_KEEPING_OPS):
            return Decimal(38, dec.scale)
        return _ARITH_INVALID
    if isinstance(other, Boolean):
        # Asymmetric (probed): ``Boolean / Decimal`` -> Float64; every
        # other cell raises (SchemaError "failed to determine supertype",
        # or InvalidOperationError for **).
        if isinstance(op, ast.Div) and not dec_is_left:
            return Float64()
        return _ARITH_INVALID
    if isinstance(other, Utf8):
        # ``+`` stringifies the Decimal and concatenates; all else errors.
        return Utf8() if isinstance(op, ast.Add) else _ARITH_INVALID
    if isinstance(other, (Date, Datetime, Time, Duration, Categorical, Enum, ListT)):
        return _ARITH_INVALID  # probed: all seven operators, both orders
    # Unknown, Struct, Binary, ... — unprobed: silent legacy fallback.
    return None


def _arith_verdict(
    op: ast.operator, left_inner: DataType, right_inner: DataType
) -> DataType | object | None:
    """Three-way outcome for binary arithmetic on Nullable-unwrapped bases.

    Returns the result ``DataType`` when polars allows the combination,
    ``_ARITH_INVALID`` when polars provably rejects it, or ``None`` when an
    operand is outside the fully-understood set (silent legacy fallback).
    """
    if isinstance(left_inner, Decimal) or isinstance(right_inner, Decimal):
        return _decimal_arith(op, left_inner, right_inner)
    lcat = _arith_category(left_inner)
    rcat = _arith_category(right_inner)
    if lcat is None or rcat is None:
        return None
    if lcat in ("num", "bool") and rcat in ("num", "bool"):
        return _numeric_arith(op, left_inner, lcat, right_inner, rcat)
    if "str" in (lcat, rcat):
        # Concat: ``+`` over str/str or str/bool (Boolean casts to its
        # string repr). Every other op or partner dtype errors at runtime.
        if isinstance(op, ast.Add) and {lcat, rcat} <= {"str", "bool"}:
            return Utf8()
        return _ARITH_INVALID
    return _temporal_arith(op, left_inner, lcat, right_inner, rcat)


# ---- comparison / is_in dtype rules (issue #33) ------------------------------
#
# Classification of ``left <cmp> right`` and ``recv.is_in(elements)`` over the
# dtypes whose comparability we fully understand: numeric ∪ {Utf8, Boolean,
# Date, Datetime, Time, Duration, Categorical, Enum}. Verified against polars
# 1.41.2 by driving the full dtype x dtype product through ``df.select``
# (tables in TestComparisonIncompatibleDtypes / TestIsInIncompatibleDtypes).
# Anything outside this set stays silent — false positives are worse than
# false negatives here.

_CMP_OP_SYMBOLS: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _cmp_category(dtype: DataType) -> str | None:
    """``_arith_category`` extended with Categorical / Enum.

    ``None`` means the dtype is outside the fully-understood set (Unknown,
    Null, Decimal, List, ...) and the caller must stay silent.
    """
    cat = _arith_category(dtype)
    if cat is not None:
        return cat
    if isinstance(dtype, Categorical):
        return "cat"
    if isinstance(dtype, Enum):
        return "enum"
    return None


# Unordered category pairs where every comparison operator errors at runtime
# in BOTH operand orders (all six operators share one validity table). The
# asymmetric str/time quirk (``s == t`` ok, ``t == s`` errors) is deliberately
# absent — asymmetric cells stay silent.
_CMP_INVALID_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("str", "num"),
        ("str", "date"),
        ("str", "datetime"),
        ("str", "dur"),
        ("bool", "date"),
        ("bool", "datetime"),
        ("bool", "time"),
        ("bool", "dur"),
        ("bool", "cat"),
        ("bool", "enum"),
        ("date", "time"),
        ("date", "dur"),
        ("datetime", "time"),
        ("datetime", "dur"),
        ("time", "dur"),
        ("cat", "num"),
        ("cat", "date"),
        ("cat", "datetime"),
        ("cat", "time"),
        ("cat", "dur"),
        ("cat", "enum"),
        ("enum", "num"),
        ("enum", "date"),
        ("enum", "datetime"),
        ("enum", "time"),
        ("enum", "dur"),
    )
)


def _comparison_invalid(left_inner: DataType, right_inner: DataType) -> bool:
    """True when polars provably rejects comparing the (unwrapped) dtype pair."""
    # Datetime pairs must agree on tz (probed: every comparison operator
    # raises SchemaError for aware vs naive AND for two different zones —
    # polars does not compare instants across zones; issue #50). Date vs
    # tz-aware Datetime stays valid, and ``is_in`` across zones is
    # probed-OK, so this rule lives here and not in the category table.
    if (
        isinstance(left_inner, Datetime)
        and isinstance(right_inner, Datetime)
        and left_inner.tz != right_inner.tz
    ):
        return True
    lcat = _cmp_category(left_inner)
    rcat = _cmp_category(right_inner)
    if lcat is None or rcat is None:
        return False
    return frozenset((lcat, rcat)) in _CMP_INVALID_PAIRS


# ``is_in`` is stricter than comparison: the receiver and element categories
# must match exactly, except int/float interchange and the string-likes
# (str accepts cat and enum partners — but cat x enum errors). Enum x str
# failures are value-dependent (out-of-category values), hence valid here.
_IS_IN_VALID_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("num", "num"),
        ("str", "str"),
        ("str", "cat"),
        ("str", "enum"),
        ("cat", "cat"),
        ("enum", "enum"),
        ("bool", "bool"),
        ("date", "date"),
        ("datetime", "datetime"),
        ("time", "time"),
        ("dur", "dur"),
    )
)


def _is_in_invalid(receiver_inner: DataType, element_inner: DataType) -> bool:
    """True when polars provably rejects ``receiver.is_in([element, ...])``.

    Both dtypes must be inside the fully-understood set; the full category
    product was probed, so any non-valid pair within the set is an error.
    """
    rcat = _cmp_category(receiver_inner)
    ecat = _cmp_category(element_inner)
    if rcat is None or ecat is None:
        return False
    return frozenset((rcat, ecat)) not in _IS_IN_VALID_PAIRS


# ---- cast structural rules (issue #34) ---------------------------------------
#
# Casts whose source -> target dtype pair polars rejects with BOTH
# ``strict=True`` and ``strict=False`` (``InvalidOperationError`` /
# ``ComputeError`` — structural, not value-dependent). Verified against
# polars 1.41.2 by driving the full source x target product through
# ``df.select`` in both modes (table in TestCastImpossibleDtypes).
# Value-dependent failures (``Utf8 -> Int64``, ``num -> Categorical``,
# ``str/int -> Enum``, ``str -> Struct``) and anything outside the
# understood set stay silent.


def _cast_category(dtype: DataType) -> str | None:
    """Category of a (Nullable-unwrapped) dtype for cast classification.

    Unlike ``_cmp_category`` this splits int/float (``Categorical -> int``
    is allowed via the physical repr while ``Categorical -> float`` is
    rejected) and adds Decimal / List / Struct.
    """
    if isinstance(dtype, (Float16, Float32, Float64)):
        return "float"
    if type(dtype) in NUMERIC_DTYPES:
        return "int"
    if isinstance(dtype, Decimal):
        return "decimal"
    if isinstance(dtype, Utf8):
        return "str"
    if isinstance(dtype, Boolean):
        return "bool"
    if isinstance(dtype, Date):
        return "date"
    if isinstance(dtype, Datetime):
        return "datetime"
    if isinstance(dtype, Time):
        return "time"
    if isinstance(dtype, Duration):
        return "dur"
    if isinstance(dtype, Categorical):
        return "cat"
    if isinstance(dtype, Enum):
        return "enum"
    if isinstance(dtype, ListT):
        return "list"
    if isinstance(dtype, Array):
        return "array"
    if isinstance(dtype, Struct):
        return "struct"
    return None


# Directional (source_category, target_category) pairs that fail in both
# strict modes. Struct *sources* mostly never appear: polars casts a
# Struct's fields instead of rejecting (``Struct -> Utf8`` yields String) —
# except Struct -> Array, probed InvalidOperationError. Scalar ->
# List/Struct only fails for temporal/categorical sources (numeric, str
# and bool sources wrap) but EVERY probed scalar -> Array cast fails
# (issue #53: int/float/str/bool/date/datetime/time/dur InvalidOperation-
# Error, cat/enum ComputeError, both modes). Array -> any non-list-like
# target is "cannot cast Array type" in both modes; Array -> List is
# probed-OK for every probed element pair and ``list -> array`` is
# value-dependent (width) — both handled in ``_cast_invalid``.
_CAST_INVALID_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {("str", "bool"), ("bool", "cat"), ("bool", "enum")}
    | {("date", t) for t in ("bool", "time", "dur", "cat", "enum", "list", "struct")}
    | {("datetime", t) for t in ("bool", "dur", "cat", "enum", "list", "struct")}
    | {("time", t) for t in ("bool", "date", "datetime", "cat", "enum", "list", "struct")}
    | {
        ("dur", t)
        for t in ("str", "bool", "date", "datetime", "time", "cat", "enum", "list", "struct")
    }
    | {
        (s, t)
        for s in ("cat", "enum")
        for t in ("float", "bool", "date", "datetime", "time", "dur", "list", "struct")
    }
    | {
        ("list", t)
        for t in (
            "int",
            "float",
            "str",
            "bool",
            "date",
            "datetime",
            "time",
            "dur",
            "cat",
            "enum",
            "struct",
        )
    }
    | {
        ("array", t)
        for t in (
            "int",
            "float",
            "str",
            "bool",
            "date",
            "datetime",
            "time",
            "dur",
            "cat",
            "enum",
            "struct",
        )
    }
    | {
        (s, "array")
        for s in (
            "int",
            "float",
            "str",
            "bool",
            "date",
            "datetime",
            "time",
            "dur",
            "cat",
            "enum",
            "struct",
        )
    }
)


# Directional (source_category, target_category) pairs whose cast is
# probed VALUE-INDEPENDENT: it succeeds for every value of the source
# dtype, so pandera ``Config.coerce`` can never fail on it (issue #58).
# Probed against pandera coerce + polars strict cast (1.41.2) with
# adversarial values (extreme ints/dates, NaN/inf, non-ASCII, invalid
# UTF-8 bytes):
#
# - formatting into String — but NOT ``dur -> str`` (probed
#   InvalidOperationError, see _CAST_INVALID_PAIRS) and NOT
#   ``Binary -> str`` (depends on the bytes being valid UTF-8);
# - bool <-> numeric (0/1 out, "!= 0" in — NaN/inf included);
# - date -> datetime (midnight), datetime -> date (truncation),
#   datetime -> time (extraction), time -> dur (since midnight);
# - any string is a valid Categorical category; Enum categories are
#   strings;
# - datetime -> datetime and duration -> duration changes are handled
#   structurally in ``_cast_verdict`` (issues #50/#66): tz changes and
#   unit coarsening are value-independent; unit refining is not.
#
# Deliberately absent: numeric -> numeric (narrowing overflows are
# value-dependent — pandera coerce of Float64(-1e308) into Float32 is a
# probed SchemaError; the blanket numeric tolerance lives in
# ``checker._is_coercible_difference`` as legacy policy), Enum targets
# (category membership), Decimal targets (precision overflow), and every
# unprobed cell.
_CAST_ALWAYS_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        (s, "str")
        for s in ("int", "float", "bool", "date", "datetime", "time", "cat", "enum", "decimal")
    }
    | {("bool", "int"), ("bool", "float"), ("int", "bool"), ("float", "bool")}
    | {("date", "datetime"), ("datetime", "date"), ("datetime", "time"), ("time", "dur")}
    | {("str", "cat"), ("enum", "cat")}
)


# Three-way cast classification (issues #34 / #58):
# - "never":  polars provably rejects the cast even with ``strict=False``
#             (structural — the PLY013 trigger);
# - "always": the cast is probed value-independent (the coerce-tolerance
#             trigger);
# - "value-dependent": everything in between, plus every cell outside the
#             probed category set — neither flagged nor tolerated.
CastVerdict = Literal["always", "value-dependent", "never"]


def _cast_verdict(source_inner: DataType, target_inner: DataType) -> CastVerdict:
    """Classify ``source.cast(target)`` — see ``CastVerdict``.

    Equal dtypes are trivially "always" (Unknown excluded: claiming
    anything about it would break gradual typing). ``list -> list`` and
    ``array -> array`` recurse on the element dtypes — a "never" element
    pair makes the container cast "never" (``List(Date) ->
    List(Duration)`` fails in both strict modes), an "always" element pair
    makes ``list -> list`` "always" (probed: pandera coerce casts
    elements), but ``array -> array`` stays "value-dependent" (the widths
    are not modeled). ``array -> list`` is probed-OK for every probed
    element pair and ``list -> array`` only fails when the list lengths
    don't match the width — both value-dependent territory here; the
    issue-#53 coerce leniency for Array-sided pairs is policy in
    ``checker._is_coercible_difference``, not a verdict.
    """
    if isinstance(source_inner, Unknown) or isinstance(target_inner, Unknown):
        return "value-dependent"
    if source_inner == target_inner:
        return "always"
    scat = _cast_category(source_inner)
    tcat = _cast_category(target_inner)
    if scat is None or tcat is None:
        return "value-dependent"
    if (scat == "list" and tcat == "list") or (scat == "array" and tcat == "array"):
        assert isinstance(source_inner, (ListT, Array)) and isinstance(target_inner, (ListT, Array))
        if (
            isinstance(source_inner, Array)
            and isinstance(target_inner, Array)
            and source_inner.width is not None
            and target_inner.width is not None
            and source_inner.width != target_inner.width
        ):
            # Probed (polars 1.41.2, backlog C-7): "cannot cast Array to a
            # different width" raises in both strict modes.
            return "never"
        source_elem = source_inner.inner
        target_elem = target_inner.inner
        if isinstance(source_elem, Nullable):
            source_elem = source_elem.inner
        if isinstance(target_elem, Nullable):
            target_elem = target_elem.inner
        elem_verdict = _cast_verdict(source_elem, target_elem)
        if elem_verdict == "never":
            return "never"
        if elem_verdict == "always" and scat == "list":
            return "always"
        return "value-dependent"
    if {scat, tcat} == {"list", "array"}:
        return "value-dependent"
    if isinstance(source_inner, Datetime) and isinstance(target_inner, Datetime):
        # tz changes are probed value-independent (issue #50). Unit changes
        # (issue #66) are value-independent only toward a coarser-or-equal
        # unit (a division); refining multiplies and overflows for extreme
        # values (probed: Datetime[us] year 9999 -> ns raises
        # InvalidOperationError strict / nulls non-strict).
        if time_unit_refines(source_inner.unit, target_inner.unit):
            return "value-dependent"
        return "always"
    if isinstance(source_inner, Duration) and isinstance(target_inner, Duration):
        # Same unit rule as Datetime (probed: extreme Duration[ms] -> ns
        # raises InvalidOperationError; coarsening always succeeds).
        if time_unit_refines(source_inner.unit, target_inner.unit):
            return "value-dependent"
        return "always"
    if (scat, tcat) in _CAST_INVALID_PAIRS:
        return "never"
    if (scat, tcat) in _CAST_ALWAYS_PAIRS:
        return "always"
    return "value-dependent"


def _cast_invalid(source_inner: DataType, target_inner: DataType) -> bool:
    """True when polars provably rejects the cast even with ``strict=False``."""
    return _cast_verdict(source_inner, target_inner) == "never"


# ``Expr.diff`` on an unsigned-int receiver widens to the signed dtype of
# the next width so negative differences are representable (probed polars
# 1.41.2; issue #46). UInt128 is absent: it has no wider signed dtype and
# polars keeps UInt128.
_DIFF_UNSIGNED_WIDENING: dict[type[DataType], DataType] = {
    UInt8: Int16(),
    UInt16: Int32(),
    UInt32: Int64(),
    UInt64: Int64(),
}


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


def _missing_column_diag(frame: FrameType, name: str) -> tuple[str, str]:
    """(code, message) for a missing-column reference on a CLOSED frame.

    Checked-island semantics (issue #83): on a frame bound from a
    non-strict declared schema the lookup is an interface violation
    against the declaration (PLY042, honest wording — the schema admits
    caller extras at runtime); on an exact frame it is a provable
    runtime miss (PLY001).
    """
    if frame.nonstrict_schema is not None:
        return PLY042, (
            f"column '{name}' is not declared in schema "
            f"'{frame.nonstrict_schema}' — the (non-strict) schema admits "
            f"extra columns at runtime, but this function's declaration "
            f"does not promise it. Declare the column on the schema, or "
            f"take a bare pl.DataFrame parameter for row-polymorphic helpers"
        )
    return PLY001, f"Column '{name}' not found. Available columns: {list(frame.columns.keys())}"


def _call_result_frame(declared: FrameType | None, source_name: str) -> FrameType | None:
    """The FrameType a call site binds for a callee's declared return.

    A ``strict=False`` return schema is pandera's "at least these
    columns": the callee may preserve arbitrary caller columns (the
    row-polymorphic helper pattern) and ``check_types`` passes them
    through — so the call result is an OPEN frame (issue #81). The
    pandera-expressible signature cannot share the row variable between
    parameter and return, so the caller's extra columns degrade to
    Unknown through the call rather than keeping their dtypes.
    ``strict=True`` returns stay closed — that closure is what makes
    select-style proofs possible. Always a fresh FrameType so downstream
    laziness stamping never mutates the registry's cached signature.
    """
    if declared is None:
        return None
    return FrameType(
        columns=dict(declared.columns),
        strict=declared.strict,
        rest=None if declared.strict else RowVar(source_name),
        is_lazy=declared.is_lazy,
        coerce=declared.coerce,
    )


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


class _ReplaceNode(ast.NodeTransformer):
    """Swap one specific node — matched by identity — for a replacement.

    Used by the plural-``pl.col`` expansion (issue #42): each clone of the
    surrounding expression gets its plural node replaced by a single-name
    ``pl.col(name)``. Identity matching makes the target unambiguous even
    when the tree contains structurally equal calls.
    """

    def __init__(self, target: ast.AST, replacement: ast.AST) -> None:
        self.target = target
        self.replacement = replacement

    def visit(self, node: ast.AST) -> ast.AST:
        if node is self.target:
            return self.replacement
        return self.generic_visit(node)


def _contains_name_accessor(node: ast.expr) -> bool:
    """Whether the expression tree contains a ``.name`` accessor (issue #56).

    Consulted by the select / with_columns layers when a positional
    expression resolved to a dtype but no output name: a ``.name.*`` chain
    with a statically unknowable result name (``map``, a non-literal
    prefix, ...) still produces a column at runtime, so the result frame
    is opened (``rest`` set) instead of losing the column — a missing
    declared column downstream would be a false positive.
    """
    return any(isinstance(sub, ast.Attribute) and sub.attr == "name" for sub in ast.walk(node))


def _nonboolean_predicate_error(
    dtype: DataType | None, op: str = "filter", noun: str = "predicate"
) -> str | None:
    """Return a PLY008 message when a filter predicate's / when-condition's
    dtype is known and not Boolean (issues #28 / #37).

    ``None`` (unresolved) and ``Unknown`` dtypes are never flagged — we don't
    guess. A ``Nullable`` wrapper is unwrapped first: ``Nullable[Boolean]``
    is a valid predicate (null rows are simply dropped by ``filter``; a null
    ``when`` condition takes the otherwise branch — both probed).
    """
    if dtype is None:
        return None
    inner = dtype.inner if isinstance(dtype, Nullable) else dtype
    if isinstance(inner, (Boolean, Unknown)):
        return None
    return tag(
        PLY008,
        f"{op}: {noun} has dtype {dtype}, expected Boolean — "
        f"polars only accepts boolean {noun}s "
        f"(use a comparison or `.is_*()` method)",
    )


# ---- namespace accessor receiver dtypes (issue #31) -------------------------
#
# Which (Nullable-unwrapped) receiver dtypes each expression sub-namespace
# accepts. Verified against polars 1.41.2:
# - ``.str`` rejects every non-String dtype at runtime, Categorical/Enum
#   included ("InvalidOperationError: expected String type, got: cat"), so
#   only Utf8 passes.
# - ``.list`` requires List and ``.arr`` requires Array (issue #53) —
#   probed: every ``.arr.*`` method on a List column raises
#   "InvalidOperationError: expected Array datatype for array operation"
#   and ``.list.*`` on an Array column raises "expected List data type".
# - ``.bin`` rejects every non-Binary dtype (issue #51) — probed: Int64
#   and String receivers both raise "SchemaError: expected `Binary`".
# - ``.cat`` requires Categorical or Enum (issue #54) — probed: Int64,
#   String, Date, Boolean and List receivers all raise "SchemaError:
#   expected an Enum or Categorical type".
# Unknown / unresolved receivers are exempted by the caller.
_NAMESPACE_VALID_DTYPES: dict[str, tuple[type[DataType], ...]] = {
    "str": (Utf8,),
    "dt": (Date, Datetime, Time, Duration),
    "list": (ListT,),
    "arr": (Array,),
    "struct": (Struct,),
    "bin": (Binary,),
    "cat": (Categorical, Enum),
}

_NAMESPACE_EXPECTED_DESC: dict[str, str] = {
    "str": "a String column",
    "dt": "a temporal column (Date, Datetime, Time or Duration)",
    "list": "a List column",
    "arr": "an Array column",
    "struct": "a Struct column",
    "bin": "a Binary column",
    "cat": "a Categorical or Enum column",
}

# Exception polars raises for a wrong receiver dtype — named in the PLY012
# message. Every namespace raises InvalidOperationError except ``.cat``
# (probed: SchemaError "expected an Enum or Categorical type").
_NAMESPACE_RUNTIME_ERROR: dict[str, str] = {
    "cat": "SchemaError",
}


def _str_constant(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_arg(call: ast.Call | None, *, position: int, name: str) -> ast.expr | None:
    """One call argument: positional index or keyword (keyword wins)."""
    if call is None:
        return None
    node: ast.expr | None = None
    if position < len(call.args):
        node = call.args[position]
    for kw in call.keywords:
        if kw.arg == name:
            node = kw.value
    return node


def _int_literal(node: ast.expr | None) -> int | None:
    """The value of an explicit int literal (bools excluded), else None."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _stdvar_ddof_zero(call: ast.Call) -> bool:
    """True when a ``std``/``var`` call carries an explicit literal ``ddof=0``.

    ``Expr.std(ddof: int = 1)`` / ``Expr.var(ddof: int = 1)`` take ``ddof``
    as the first positional parameter (probed, polars 1.41.2). ``ddof=0``
    is total on non-empty input (a singleton group yields 0.0, not null);
    anything else — including a non-literal value — stays conservative.
    """
    return _int_literal(_call_arg(call, position=0, name="ddof")) == 0


def _rolling_min_samples_total(
    method: str,
    call: ast.Call,
    const_int: Callable[[ast.expr | None], int | None] = _int_literal,
) -> bool:
    """True when a ``rolling_*`` call provably fills every row's window.

    Rows whose window holds fewer than ``min_samples`` (default:
    ``window_size``) non-null values are null (probed, polars 1.41.2). The
    window always contains the row itself, so an explicit ``min_samples``
    of 0/1 — or ``window_size=1`` with min_samples unset — makes the
    rolling output total on a non-null receiver. ``min_samples`` is
    keyword-only; ``min_periods`` is its deprecated pre-1.21 spelling
    (still accepted, with a warning). ``window_size`` is the first
    positional parameter except for ``rolling_quantile(quantile,
    interpolation, window_size, ...)``. Args resolve through ``const_int``
    (a literal, or — backlog B-5 — a name bound to an int constant);
    unresolvable values stay conservative (Nullable result). A literal
    ``min_samples<=1`` is total for ANY accepted window_size (probed:
    window_size=0 is an expanding window with no nulls; a negative
    window_size raises OverflowError before producing a frame).
    """
    ms_node = next(
        (kw.value for kw in call.keywords if kw.arg in ("min_samples", "min_periods")),
        None,
    )
    if ms_node is not None:
        return const_int(ms_node) in (0, 1)
    ws_position = 2 if method == "rolling_quantile" else 0
    return const_int(_call_arg(call, position=ws_position, name="window_size")) == 1


def _rolling_ddof_zero(
    call: ast.Call,
    const_int: Callable[[ast.expr | None], int | None] = _int_literal,
) -> bool:
    """Explicit ``ddof=0`` on ``rolling_std``/``rolling_var``.

    Unlike ``Expr.std``/``Expr.var``, the rolling variants take ``ddof``
    as keyword-only (probed signature, polars 1.41.2). The default
    ``ddof=1`` is null on 1-sample windows even with ``min_samples=1``;
    ``ddof=0`` restores totality (1-sample window -> 0.0). The arg
    resolves through ``const_int`` like the window args (backlog B-5).
    """
    ddof_node = next((kw.value for kw in call.keywords if kw.arg == "ddof"), None)
    return const_int(ddof_node) == 0


def _time_zone_arg_dtype(method: str, call_node: ast.Call | None, receiver: Datetime) -> DataType:
    """Result dtype of ``dt.replace_time_zone(...)`` / ``dt.convert_time_zone(...)``
    on a Datetime receiver (issue #50 collateral).

    Probed (polars 1.41.2): both methods yield ``Datetime[arg]`` for naive
    AND aware receivers, preserving the receiver's time unit (issue #66);
    ``replace_time_zone(None)`` strips the tz; ``convert_time_zone(None)``
    raises TypeError at expression-construction time ("argument
    'time_zone': 'None' is not an instance of 'str'"), so its silent
    Unknown is sound — no frame ever materializes.

    A non-literal time_zone stays Unknown deliberately (backlog B-6): the
    result is always SOME Datetime (or naive, if a variable None reaches
    ``replace_time_zone``), but ``types.Datetime`` equality/subtyping is
    exact on tz and has no "aware, tz unknown" wildcard, so any concrete
    claim would be a guess — a wrong precise type is worse than Unknown.
    """
    tz_node = _call_arg(call_node, position=0, name="time_zone")
    if isinstance(tz_node, ast.Constant):
        if isinstance(tz_node.value, str):
            return Datetime(tz=tz_node.value, unit=receiver.unit)
        if tz_node.value is None and method == "replace_time_zone":
            return Datetime(unit=receiver.unit)
    return Unknown()


# chrono offset directives: %z, %:z, %::z, %:::z, %#z. Probed (polars
# 1.41.2): a format literal containing ANY of them resolves the dtype to
# Datetime[UTC] — even when row-level parsing later fails (the error
# message names `datetime[μs, UTC]` as the target). Polars' own dtype
# resolution is substring-level: the escaped ``%%z`` also resolves to
# UTC (probed), so a plain scan — not an escape-aware one — is faithful.
_TZ_DIRECTIVE_RE = re.compile(r"%(?::{0,3}|#)z")

# chrono fractional-second directives: %f / %.f / %3f / %.3f / %6f / %.6f /
# %9f / %.9f. Probed (polars 1.41.2): the 3-digit forms resolve
# ``str.to_datetime`` to Datetime[ms] and the 9-digit forms to Datetime[ns];
# every other variant (and no directive at all) keeps the us default.
_FRACTION_DIRECTIVE_RE = re.compile(r"%\.?([369])?f")
_FRACTION_DIGIT_UNIT: dict[str | None, str] = {"3": "ms", "9": "ns"}


def _str_to_datetime_dtype(call_node: ast.Call | None) -> DataType:
    """Result dtype of ``str.to_datetime(...)`` (issues #50/#66 collateral).

    Probed (polars 1.41.2): no tz-affecting arguments -> naive Datetime;
    ``time_zone="X"`` (string literal) -> ``Datetime[X]``; a format literal
    containing an offset directive (``%z`` and its ``%:z`` / ``%::z`` /
    ``%:::z`` / ``%#z`` variants) -> ``Datetime[UTC]``. The unit defaults
    to us; ``time_unit=`` (string literal) wins, otherwise a fractional-
    second format directive selects it (``%.3f`` -> ms, ``%.9f`` -> ns).
    A non-literal ``time_zone`` / ``time_unit`` / format degrades to
    Unknown: the result is always SOME Datetime, but ``types.Datetime``
    equality/subtyping is exact on tz and unit and has no wildcard, so
    claiming the defaults would be a guess.
    """
    tz: str | None = None
    tz_from_kw = False
    unit: str | None = None
    if call_node is not None:
        for kw in call_node.keywords:
            if kw.arg == "time_zone":
                if isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        tz = kw.value.value
                        tz_from_kw = True
                        continue
                    if kw.value.value is None:
                        continue  # explicit None — same as omitted
                return Unknown()
            if kw.arg == "time_unit":
                if isinstance(kw.value, ast.Constant):
                    if kw.value.value is None:
                        continue  # explicit None — the format decides
                    if isinstance(kw.value.value, str) and kw.value.value in TIME_UNITS:
                        unit = kw.value.value
                        continue
                return Unknown()
    fmt_node = _call_arg(call_node, position=0, name="format")
    if fmt_node is not None:
        fmt = _str_constant(fmt_node)
        if fmt is None:
            return Unknown()
        if not tz_from_kw and _TZ_DIRECTIVE_RE.search(fmt):
            tz = "UTC"
        if unit is None:
            match = _FRACTION_DIRECTIVE_RE.search(fmt)
            if match is not None:
                unit = _FRACTION_DIGIT_UNIT.get(match.group(1), "us")
    return Datetime(tz=tz, unit=unit if unit is not None else "us")


def _str_to_decimal_dtype(call_node: ast.Call | None) -> DataType:
    """Result dtype of ``str.to_decimal(...)`` (issue #61).

    Probed (polars 1.41.2): ``scale`` is keyword-only and required on the
    expression namespace — ``to_decimal(scale=N)`` yields ``Decimal(38, N)``
    (precision is polars' default 38). A positional or missing scale raises
    TypeError before any frame exists, and a non-literal scale is
    unknowable — all of those degrade to Unknown rather than claiming a
    fixed scale (the issue #61 false positive was a hardcoded
    ``Decimal(38, 0)``).
    """
    if call_node is not None:
        for kw in call_node.keywords:
            if kw.arg == "scale":
                v = kw.value
                if (
                    isinstance(v, ast.Constant)
                    and isinstance(v.value, int)
                    and not isinstance(v.value, bool)
                ):
                    return Decimal(DECIMAL_DEFAULT.precision, v.value)
                return Unknown()
    return Unknown()


# ``dt.epoch`` time_unit literals and their probed return dtypes (issue
# #73). Probed (polars 1.41.2): "ns"/"us"/"ms"/"s" -> Int64 but "d" ->
# Int32 (days since epoch) — on every accepting receiver (Date and
# Datetime at any unit/tz behave identically). Any other literal raises
# ValueError at expression-construction time.
_EPOCH_UNIT_DTYPES: dict[str, DataType] = {
    "ns": Int64(),
    "us": Int64(),
    "ms": Int64(),
    "s": Int64(),
    "d": Int32(),
}


def _dt_epoch_dtype(call_node: ast.Call | None) -> DataType:
    """Result dtype of ``dt.epoch(...)`` (issue #73).

    The return dtype depends on the ``time_unit`` argument (positional or
    keyword, probed): see ``_EPOCH_UNIT_DTYPES``; the no-arg default is
    "us" -> Int64. A non-literal time_unit is unknowable and an invalid
    literal never reaches a frame — both degrade to Unknown rather than
    claiming Int64 (the issue #73 false positive was a fixed table entry).
    """
    unit_node = _call_arg(call_node, position=0, name="time_unit")
    if unit_node is None:
        return Int64()  # default time_unit="us"
    unit = _str_constant(unit_node)
    if unit is not None and unit in _EPOCH_UNIT_DTYPES:
        return _EPOCH_UNIT_DTYPES[unit]
    return Unknown()


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


def _flatten_expr_args(args: list[ast.expr]) -> list[ast.expr]:
    """Flatten list/tuple literal arguments into their elements.

    Multi-expression helpers (``pl.struct``, ``pl.coalesce``,
    ``pl.concat_str``, ``pl.sum_horizontal``, ...) accept both varargs and a
    list of expressions: ``pl.struct(pl.col("a"), pl.col("b"))`` ≡
    ``pl.struct([pl.col("a"), pl.col("b")])``. Expanding top-level
    ``ast.List`` / ``ast.Tuple`` args lets the caller analyze both forms
    uniformly (issue #16). MIXED forms (a list/tuple literal next to other
    positional args) never reach this helper — they are flagged PLY017 by
    ``_mixed_list_args`` first (issue #59).
    """
    out: list[ast.expr] = []
    for arg in args:
        if isinstance(arg, (ast.List, ast.Tuple)):
            out.extend(arg.elts)
        else:
            out.append(arg)
    return out


# The ``pl.*`` helpers that accept either varargs or one list of
# expressions — exactly the issue-#16 flatten consumers handled in
# ``_analyze_pl_func``. ``pl.format`` takes a template string first but is
# probed to crash the same way when a list literal follows it.
_MULTI_EXPR_HELPERS: frozenset[str] = frozenset(
    {
        "struct",
        "coalesce",
        "concat_str",
        "format",
        "concat_list",
        "sum_horizontal",
        "min_horizontal",
        "max_horizontal",
        "mean_horizontal",
    }
)


def _mixed_list_args(args: list[ast.expr]) -> bool:
    """True when a list/tuple literal is mixed with other positional args.

    Probed (polars 1.41.2; issue #59) for every multi-expression helper:
    the mix is never flattened at runtime — an expression-bearing list
    raises TypeError ("Nested object types"); a string-only list either
    raises (coalesce / concat_str / horizontal: supertype failure) or
    misparses as a nested *literal* column (struct / concat_list). A single
    list/tuple argument (with or without keyword args) is the supported
    sequence form and stays exempt.
    """
    return len(args) > 1 and any(isinstance(arg, (ast.List, ast.Tuple)) for arg in args)


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
    if not isinstance(node.func.value, ast.Name):
        return None

    # ``pl.all()`` / ``pl.exclude(...)`` are selector-flavored top-level
    # expressions with the same column-set semantics as ``cs.all()`` /
    # ``cs.exclude(...)`` (issue #20).
    if node.func.value.id == "pl":
        if node.func.attr == "all":
            # Only the no-arg form selects columns; ``pl.all("col")`` is the
            # "all values truthy" boolean aggregation — leave it to
            # expression analysis.
            if node.args:
                return None
            return list(frame.columns.keys())
        if node.func.attr == "exclude":
            # String constants and list/tuple-of-string args only; anything
            # else (variables, dtypes, nested selectors) is unresolvable
            # here and falls through to expression analysis.
            excluded: set[str] = set()
            for arg in node.args:
                single = _str_constant(arg)
                multi = _str_list_or_tuple(arg)
                if single is not None:
                    excluded.add(single)
                elif multi is not None:
                    excluded.update(multi)
                else:
                    return None
            return [c for c in frame.columns if c not in excluded]
        return None

    if node.func.value.id != "cs":
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

    # Nullable actual cannot fill a non-nullable expected slot.
    if isinstance(actual, Nullable) and not isinstance(expected, Nullable):
        return False

    # List / Array containers: compare element types with the same rules so
    # the Unknown leniency reaches nested dtypes (e.g. List[Unknown] from
    # an un-inferable ``list.eval`` body vs a declared List[T]). Array vs
    # List falls through to False — probed (issue #53): the containers are
    # not mutually substitutable.
    if isinstance(actual_base, ListT) and isinstance(expected_base, ListT):
        return _is_column_subtype(actual_base.inner, expected_base.inner)
    if isinstance(actual_base, Array) and isinstance(expected_base, Array):
        return _is_column_subtype(actual_base.inner, expected_base.inner)

    # Non-nullable is subtype of nullable with same base
    if isinstance(expected, Nullable) and not isinstance(actual, Nullable):
        return actual_base == expected_base

    return False


def _is_frame_subtype(actual: FrameType, expected: FrameType) -> bool:
    """Check if actual FrameType is subtype of expected.

    Rules:
    - actual must contain every required column expected has — unless
      ``actual`` is an open frame (``rest`` is not None), whose unknown
      extra columns may satisfy the requirement
    - For columns present on both sides, the actual dtype must be a subtype
      and an actual optional column cannot satisfy a required expected column
    - When ``expected.coerce`` is True (Pandera ``Config.coerce``),
      coercible numeric dtype differences are tolerated — ``pa.check_types``
      coerces input frames at call time
    - actual may have extra columns unless ``expected.strict`` is True
    """
    # Deferred import: checker imports analyzer at module level, so a
    # top-level import here would create a cycle.
    from polypolarism.checker import _is_coercible_difference

    for col_name, expected_spec in expected.columns.items():
        actual_spec = actual.columns.get(col_name)
        if actual_spec is None:
            # ``lacks``: closed-and-missing, or provably removed on an
            # open frame (negative knowledge, issue #78).
            if expected_spec.required and actual.lacks(col_name):
                return False
            continue
        if expected_spec.required and not actual_spec.required:
            return False
        if not _is_column_subtype(actual_spec.dtype, expected_spec.dtype):
            if expected.coerce and _is_coercible_difference(actual_spec.dtype, expected_spec.dtype):
                continue
            return False
    if expected.strict:
        for col_name, actual_spec in actual.columns.items():
            # Only REQUIRED extras are provable; an Optional column may be
            # absent at runtime (issue #84).
            if col_name not in expected.columns and actual_spec.required:
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


def _schema_definition_error(schema_name: str, schema_registry: SchemaRegistry) -> str | None:
    """PLY041 message for a schema whose definition provably crashes pandera.

    Issue #69: a wrong-arity ``Annotated[pl.<Dtype>, ...]`` field makes
    pandera raise a deferred TypeError the first time the schema is used
    (``to_schema`` / ``validate`` / ``@pa.check_types``), so every function
    referencing the schema is dead on arrival regardless of what its body
    does. Returns ``None`` for unknown or healthy schemas.
    """
    schema = schema_registry.get(schema_name)
    if schema is None or not schema.definition_errors:
        return None
    details = "; ".join(
        f"field '{field_name}': {detail}" for field_name, detail in schema.definition_errors.items()
    )
    return tag(
        PLY041,
        f"schema '{schema_name}' cannot be used at runtime: {details}",
    )


def _schema_definition_warning(schema_name: str, schema_registry: SchemaRegistry) -> str | None:
    """PLW011 message for a schema with unrecognized field annotations.

    Issue #77: a field annotation polypolarism cannot translate registers
    as a column of Unknown dtype instead of silently vanishing (which
    produced phantom "extra column" FPs on strict schemas and vanished-
    column FNs on open ones). A warning rather than an error: the name may
    be a runtime alias of a real dtype (``MyAlias = pl.Int64`` resolves
    fine in pandera), so the schema is not provably broken — but if it
    does NOT resolve, pandera raises TypeError the first time the schema
    is used. Returns ``None`` for unknown or fully-recognized schemas.
    """
    schema = schema_registry.get(schema_name)
    if schema is None or not schema.definition_warnings:
        return None
    details = "; ".join(
        f"field '{field_name}': {detail}"
        for field_name, detail in schema.definition_warnings.items()
    )
    return tag(
        PLW011,
        f"schema '{schema_name}' has unrecognized field annotations: {details} — "
        f"each column is treated as Unknown dtype (pandera raises TypeError at "
        f"first use if the annotation does not resolve to a dtype at runtime)",
    )


class ExpressionAnalyzer(ast.NodeVisitor):
    """Analyze expressions to infer their types and output column names."""

    def __init__(
        self,
        current_frame: FrameType,
        warnings: list[str] | None = None,
        registry: FunctionRegistry | None = None,
        element_dtype: DataType | None = None,
        int_consts: Mapping[str, int] | None = None,
    ):
        self.current_frame = current_frame
        self.errors: list[str] = []
        # Shared advisory channel (passed in by the body analyzer so warnings
        # bubble up to FunctionAnalysis). New list when used standalone.
        self.warnings: list[str] = warnings if warnings is not None else []
        self.registry = registry or FunctionRegistry()
        # Dtype that ``pl.element()`` resolves to. Set only by the
        # ``list.eval(...)`` body analysis (issue #44), where polars binds
        # the element to the list's inner dtype; ``None`` everywhere else
        # keeps ``pl.element()`` unresolved (silent).
        self.element_dtype = element_dtype
        # Name -> int constant bindings snapshot (function locals shadowing
        # module-level), passed in by the body analyzer (backlog B-5). Feeds
        # ``_const_int`` so int-valued call args (rolling min_samples /
        # window_size / ddof) resolve like literals. Empty when standalone.
        self.int_consts: Mapping[str, int] = int_consts if int_consts is not None else {}
        # True while ``analyze_agg_expr``'s chain fallback re-enters the
        # expression analyser (backlog N-5): expression-level aggregations
        # then infer with the grouped ("agg") context, so the probed
        # grouped-panic cells (e.g. ``mean`` on Float16) are still rejected
        # inside ``agg(...)`` method chains. False everywhere else —
        # ``select``/``with_columns`` are whole-frame reductions.
        self._in_agg_chain = False

    def _const_int(self, node: ast.expr | None) -> int | None:
        """Resolve ``node`` to an int: a literal, or a ``Name`` bound to an
        int constant (mirrors the body analyzer's ``_const_str``)."""
        value = _int_literal(node)
        if value is not None:
            return value
        if isinstance(node, ast.Name):
            return self.int_consts.get(node.id)
        return None

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

        # Zero-arg ``pl.len()`` / ``pl.count()`` — group-size aggregations
        # that take no input column. polars returns its IDX dtype (UInt32)
        # and defaults the output column name to the function name. Uses
        # the pre-resolved-dtype AggExpr form; the kwarg path in
        # ``_infer_agg_call`` overwrites ``alias`` with the kwarg name.
        if (
            isinstance(agg_node, ast.Call)
            and isinstance(agg_node.func, ast.Attribute)
            and agg_node.func.attr in ("len", "count")
            and isinstance(agg_node.func.value, ast.Name)
            and agg_node.func.value.id == "pl"
            and not agg_node.args
        ):
            return AggExpr(
                column=alias or agg_node.func.attr,
                function=None,
                alias=alias,
                dtype=UInt32(),
            )

        # Now look for the aggregation function call
        if isinstance(agg_node, ast.Call):
            if isinstance(agg_node.func, ast.Attribute):
                agg_func_name = agg_node.func.attr
                col_expr = agg_node.func.value

                agg_func_value = agg_function_for(agg_func_name)
                if agg_func_value is not None:
                    # Form 1: ``pl.col("X").<agg>()`` — receiver is the col.
                    col_name = self._extract_col_name(col_expr)

                    # std/var with an explicit literal ``ddof=0`` flip the
                    # result's nullability (issue #60) — call-level detail
                    # the direct AggExpr form cannot carry. Skip the direct
                    # form so the chain fallback below routes through the
                    # ddof-aware expression path. Form-1 only: in the
                    # ``pl.std("X", 0)`` shorthand the ddof sits at a
                    # different positional slot and stays Nullable (safe
                    # side; the shorthand path lives in _analyze_pl_func).
                    if (
                        col_name is not None
                        and agg_func_name in ("std", "var")
                        and _stdvar_ddof_zero(agg_node)
                    ):
                        col_name = None
                    elif col_name is None:
                        # Form 2: ``pl.<agg>("X")`` top-level shorthand —
                        # the column name is the first positional arg of
                        # the call, not buried inside a separate
                        # ``pl.col(...)`` receiver.
                        if isinstance(col_expr, ast.Name) and col_expr.id == "pl" and agg_node.args:
                            col_name = _str_constant(agg_node.args[0])

                    if col_name:
                        return AggExpr(
                            column=col_name,
                            function=agg_func_value,
                            alias=alias,
                        )

        # Implicit list aggregation (issue #27): a bare column reference in
        # ``agg`` with no reducing function collects each group's values
        # into a list — ``agg(vs=pl.col("v"))`` is ``List(Int64)`` at
        # runtime, not the element dtype. Single-arg ``pl.col("x")`` only;
        # anything with a method chain was consumed by the branches above
        # or falls through to the chain fallback. A bare string constant is
        # the same thing — polars parses ``agg("v")`` as ``pl.col("v")``.
        if isinstance(agg_node, ast.Call) and len(agg_node.args) == 1 and not agg_node.keywords:
            bare_col = self._extract_col_name(agg_node)
            if bare_col is not None:
                return AggExpr(column=bare_col, function=AggFunction.LIST, alias=alias)
        bare_str = _str_constant(agg_node)
        if bare_str is not None:
            return AggExpr(column=bare_str, function=AggFunction.LIST, alias=alias)

        # Chain fallback: anything more elaborate (post-aggregation method
        # chains like ``pl.col("ts").max().dt.year()``, arithmetic on the
        # aggregated value, sub-namespace methods on the aggregated value,
        # etc.) is handled by reusing the expression analyser. We delegate to
        # ``analyze_select_expr`` on the *original* node (it strips the
        # ``.alias(...)`` itself), and turn its (name, dtype) into an
        # AggExpr with a pre-resolved ``dtype`` override. The flag keeps the
        # grouped ("agg") inference context alive through the re-entry
        # (backlog N-5).
        prev_in_agg = self._in_agg_chain
        self._in_agg_chain = True
        try:
            chain_name, chain_dtype = self.analyze_select_expr(node)
        finally:
            self._in_agg_chain = prev_in_agg
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
    # ``not_`` is deliberately absent (issue #72): it negates Booleans but
    # operates BITWISE on integers — see ``_NOT_VALID_RECEIVERS`` below.
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
        }
    )

    # ---- bitwise/logical NOT (issue #72) -------------------------------------
    # Probed (polars 1.41.2) receiver matrix for ``Expr.not_`` / ``~``. The
    # integer behaviour is documented contract — the ``Expr.not_`` docstring
    # says it "operates bitwise on integers":
    # - Boolean -> Boolean (null-preserving: ~null is null);
    # - every int width (Int8..Int128, UInt8..UInt128) -> same dtype
    #   (bitwise NOT: ~1 == -2, ~UInt8(1) == 254) — i.e. every valid
    #   receiver is dtype-preserving;
    # - everything else (floats incl. Float16, Utf8, Binary, Date,
    #   Datetime, Time, Duration, Decimal, Categorical, Enum, List, Array,
    #   Struct, Null) raises InvalidOperationError
    #   ("dtype `X` not supported in 'not' operation") -> PLY016 and the
    #   output degrades to Unknown.
    _NOT_VALID_RECEIVERS = (
        Boolean,
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
            # Window methods are absent: ``over``'s dtype depends on
            # ``mapping_strategy`` (issues #32/#45), the cum_* family is
            # strictly typed (``_CUM_INVALID_RECEIVERS`` below, issue #49),
            # and rolling_sum/min/max are strictly typed AND windowed-
            # nullable (``_ROLLING_INVALID_RECEIVERS`` below, issue #57).
            "set_sorted",
            "reverse",
        }
    )

    # ---- cumulative reducers (issue #49) ------------------------------------
    # Probed (polars 1.41.2) receiver-dtype matrix for the cum_* family.
    # Receivers listed here raise InvalidOperationError at runtime
    # ("'cum_sum' operation not supported for dtype 'str'") -> PLY016 and
    # the output degrades to Unknown. ``cum_count`` is deliberately absent:
    # it returns UInt32 for EVERY receiver dtype (probed incl. String /
    # List / Struct / Null) and keeps its own branch. The Float64-returning
    # rolling family (rolling_mean & co.) is NOT mirrored here — polars
    # accepts e.g. rolling_mean on String (all-null Float64, probed), so it
    # stays in the silent path; rolling_sum/min/max ARE strictly typed —
    # see ``_ROLLING_INVALID_RECEIVERS`` below (issue #57).
    # Array receivers probed too (issue #53): cum_sum/cum_prod/cum_min/
    # cum_max all raise "operation not supported for dtype array[...]";
    # cum_count is fine (UInt32 for every receiver, Array included).
    _CUM_INVALID_RECEIVERS: dict[str, tuple[type[DataType], ...]] = {
        "cum_sum": (Utf8, Date, Datetime, Time, ListT, Array, Struct, Categorical, Enum, Null),
        "cum_prod": (
            Utf8,
            Date,
            Datetime,
            Time,
            Duration,
            Decimal,
            ListT,
            Array,
            Struct,
            Categorical,
            Enum,
            Null,
        ),
        "cum_min": (Utf8, ListT, Array, Struct, Null),
        "cum_max": (Utf8, ListT, Array, Struct, Null),
    }

    # Probed-valid non-preserving cells. cum_sum upcasts narrow ints to
    # Int64 as an overflow guard (matches the polars docs); Boolean sums
    # to UInt32 and Decimal widens its precision to 38 (scale kept).
    _CUM_SUM_INT64_RECEIVERS = (Int8, Int16, UInt8, UInt16)
    # cum_prod computes in Int64 for every int dtype narrower than UInt64
    # (UInt64 / Int128 / UInt128 keep their dtype); Boolean also -> Int64.
    _CUM_PROD_INT64_RECEIVERS = (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, Boolean)

    # ---- dtype-carrying rolling reducers (issue #57; deferred from #49) -----
    # Probed (polars 1.41.2) receiver-dtype matrix for rolling_sum/min/max.
    # Receivers listed here raise InvalidOperationError at runtime ->
    # PLY016 and the output degrades to Unknown. rolling_min/max accept
    # Boolean and the temporals (dtype-preserving); rolling_sum rejects
    # them all except Boolean (-> UInt32).
    _ROLLING_INVALID_RECEIVERS: dict[str, tuple[type[DataType], ...]] = {
        "rolling_sum": (
            Utf8,
            Date,
            Datetime,
            Time,
            Duration,
            Decimal,
            ListT,
            Array,
            Struct,
            Categorical,
            Enum,
            Null,
        ),
        "rolling_min": (Utf8, Decimal, ListT, Array, Struct, Categorical, Enum, Null),
        "rolling_max": (Utf8, Decimal, ListT, Array, Struct, Categorical, Enum, Null),
    }

    # Probed-valid non-preserving cells: rolling_sum upcasts narrow ints to
    # Int64 (overflow guard, mirrors cum_sum) and Boolean to UInt32; every
    # other accepted receiver keeps its dtype (incl. Int128/UInt128).
    _ROLLING_SUM_INT64_RECEIVERS = (Int8, Int16, UInt8, UInt16)

    # ---- numeric-only elementwise methods (issue #62) -----------------------
    # Probed (polars 1.41.2) receiver-dtype matrix for the numeric-ish
    # members of ``_DTYPE_PRESERVING_METHODS``. Receivers listed here raise
    # InvalidOperationError at runtime (e.g. "rounding ('half_to_even') can
    # only be used on numeric types") -> PLY016 and the output degrades to
    # Unknown. Deliberately unlisted:
    # - round/floor/ceil on Decimal(p, s) -> Decimal(p, s) (preserving;
    #   floor/ceil are NOT float-only and are int-identity);
    # - abs/neg on Duration and Decimal -> preserving;
    # - clip accepts everything physically numeric — temporals, Decimal,
    #   Categorical and Enum included (probed with numeric AND with
    #   matching-dtype literal bounds; bare strings bounds are column refs).
    #   Bound-dtype interactions are out of scope: only receivers that
    #   reject every bound are flagged;
    # - sign on floats keeps the float dtype in 1.41.2 (no Int8 cast);
    # - shrink_dtype: deprecated no-op in 1.41.2, accepts every dtype.
    _NON_NUMERIC_DTYPES = (
        Utf8,
        Binary,
        Boolean,
        Date,
        Datetime,
        Time,
        Duration,
        Categorical,
        Enum,
        ListT,
        Array,
        Struct,
        Null,
    )
    _ELEMENTWISE_INVALID_RECEIVERS: dict[str, tuple[type[DataType], ...]] = {
        "round": _NON_NUMERIC_DTYPES,
        "floor": _NON_NUMERIC_DTYPES,
        "ceil": _NON_NUMERIC_DTYPES,
        "clip": (Utf8, Binary, Boolean, ListT, Array, Struct, Null),
        "abs": (
            Utf8,
            Binary,
            Boolean,
            Date,
            Datetime,
            Time,
            Categorical,
            Enum,
            ListT,
            Array,
            Struct,
            Null,
        ),
        "sign": _NON_NUMERIC_DTYPES,
        # neg additionally rejects EVERY unsigned int and Int128
        # ("`neg` operation not supported for dtype `u128`" / `i128`).
        "neg": (
            Utf8,
            Binary,
            Boolean,
            Date,
            Datetime,
            Time,
            Categorical,
            Enum,
            ListT,
            Array,
            Struct,
            Null,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            UInt128,
            Int128,
        ),
    }

    # Probed matrix for ``_FLOAT_RETURN_METHODS`` (issue #62). The error
    # class varies per cell (InvalidOperationError / ComputeError / rust
    # panic) -> PLY016 and the output degrades to Unknown. Deliberately
    # unlisted: String, Boolean, the temporals, Decimal and Null are
    # ACCEPTED by every member except entropy (polars casts them into
    # Float64 non-strictly; String yields all-null Float64) and must stay
    # silent; log1p/exp also accept Categorical/Enum. interpolate and
    # ewm_mean (issue #62 report) accept String too — both stay on the
    # silent/unhandled path.
    _FLOAT_RETURN_INVALID_RECEIVERS: dict[str, tuple[type[DataType], ...]] = {
        "log": (Binary, Categorical, Enum, ListT, Array, Struct),
        "log10": (Binary, Categorical, Enum, ListT, Array, Struct),
        "log1p": (Binary, ListT, Array, Struct),
        "exp": (Binary, ListT, Array, Struct),
        "sqrt": (Binary, Categorical, Enum, ListT, Array, Struct),
        "cbrt": (Binary, Categorical, Enum, ListT, Array, Struct),
        # entropy is the aggregating member: it rejects String/Boolean/
        # Null too, yet accepts temporals/Decimal/Categorical/Enum.
        "entropy": (Utf8, Binary, Boolean, ListT, Array, Struct, Null),
    }

    # Shift-like methods: receiver dtype, but head positions become NULL.
    # ``pct_change`` is deliberately absent (issue #71): it divides, so it
    # is NOT dtype-preserving — see ``_PCT_CHANGE_INVALID_RECEIVERS``.
    _SHIFT_LIKE_METHODS = frozenset({"shift", "diff"})

    # ---- pct_change (issue #71) ----------------------------------------------
    # Probed (polars 1.41.2) receiver matrix. ``pct_change`` divides by the
    # most-recent non-null element (same family as ``/``-division, issue
    # #14): float receivers keep their width (Float16 -> Float16,
    # Float32 -> Float32); every other accepted receiver — ints at all
    # widths (Int128/UInt128 included), Boolean, Date / Datetime (any
    # unit/tz) / Time / Duration, Decimal and Null — is cast to Float64
    # first, and Utf8 is accepted via polars' non-strict cast (all-null
    # Float64). Receivers listed here raise at runtime
    # (InvalidOperationError / ComputeError: "casting from X to Float64
    # not supported") — except Struct, which ABORTS the process in rust
    # (probed SIGSEGV, not a catchable error) — so the cell is flagged all
    # the same -> PLY016 and the output degrades to Unknown.
    _PCT_CHANGE_INVALID_RECEIVERS = (Binary, Categorical, Enum, ListT, Array, Struct)

    # ---- over(mapping_strategy="join") cardinality classification ----------
    # Probed (polars 1.41.2; issue #45): "join" only gathers the windowed
    # expression into a List when it is multi-valued per group. A
    # scalar-per-group expression (aggregation — arithmetic on one
    # included, e.g. ``sum() + 1``) broadcasts WITHOUT a List wrapper
    # under every strategy. Methods that reduce a group to one value:
    _OVER_SCALAR_METHODS = frozenset(
        {
            "sum",
            "mean",
            "count",
            "len",
            "n_unique",
            "first",
            "last",
            "min",
            "max",
            "std",
            "var",
            "median",
            "quantile",
            "product",
            "implode",
            "entropy",
        }
    )

    # Length-changing but multi-valued-per-group methods: their output is
    # not a broadcastable scalar, so "join" gathers it into a List
    # (probed: ``head(1)`` / ``unique()`` / ``drop_nulls()`` -> List).
    _OVER_VECTOR_METHODS = frozenset(
        {
            "head",
            "tail",
            "limit",
            "slice",
            "sample",
            "gather",
            "gather_every",
            "unique",
            "filter",
            "drop_nulls",
            "drop_nans",
            "sort",
            "sort_by",
            "top_k",
            "bottom_k",
            "explode",
            "flatten",
        }
    )

    # Methods whose output cardinality is decided by an opaque callable.
    _OVER_OPAQUE_CARDINALITY_METHODS = frozenset({"map_batches", "pipe"})

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
    # All six tables are re-exports from compat.polars_api so the polars
    # surface knowledge stays in one place.

    _STR_RETURN = STR_NAMESPACE_RETURN
    _DT_RETURN = DT_NAMESPACE_RETURN
    _DT_PRESERVING = DT_NAMESPACE_PRESERVING
    _LIST_PRESERVING = LIST_NAMESPACE_PRESERVING
    _LIST_ELEMENT_RETURN = LIST_NAMESPACE_ELEMENT_RETURN
    _BIN_RETURN = BIN_NAMESPACE_RETURN
    _CAT_RETURN = CAT_NAMESPACE_RETURN

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

        # Bare Python constants are literal columns (``df.select(x=1)`` is
        # ``pl.lit(1).alias("x")`` in polars). Bare *positional* strings in
        # select/with_columns are column names — but those are consumed by
        # the callers (``_str_constant``) before this method ever sees them.
        # ``infer_lit`` checks bool before int (bool subclasses int).
        if isinstance(inner_node, ast.Constant) and (
            inner_node.value is None or isinstance(inner_node.value, (bool, int, float, str, bytes))
        ):
            return alias, infer_lit(inner_node.value)

        # Comparison expressions (==, !=, <, <=, >, >=) -> Boolean.
        # A Nullable operand makes the result Nullable (``null > 0`` is null).
        # Probed-invalid dtype pairs flag PLY009 (issue #33) — each adjacent
        # pair of a chained comparison is checked. The result stays Boolean:
        # the dtype is not in question, the error is the signal.
        if isinstance(inner_node, ast.Compare):
            operand_types = [self.analyze_select_expr(inner_node.left)[1]]
            for cmp in inner_node.comparators:
                operand_types.append(self.analyze_select_expr(cmp)[1])
            for op, left_type, right_type in zip(
                inner_node.ops, operand_types, operand_types[1:], strict=False
            ):
                op_sym = _CMP_OP_SYMBOLS.get(type(op))
                if op_sym is None or left_type is None or right_type is None:
                    continue
                left_inner = left_type.inner if isinstance(left_type, Nullable) else left_type
                right_inner = right_type.inner if isinstance(right_type, Nullable) else right_type
                if _comparison_invalid(left_inner, right_inner):
                    self.errors.append(
                        tag(
                            PLY009,
                            f"comparison '{left_inner} {op_sym} {right_inner}' is not "
                            f"supported — polars rejects comparing these dtypes at "
                            f"runtime; cast an operand first",
                        )
                    )
            resolved = [t for t in operand_types if t is not None]
            return alias, _wrap_nullable_if_any(Boolean(), resolved)

        # Logical operators expressed as bitwise: a & b, a | b, a ^ b -> Boolean.
        # Nullability propagates from either operand (``null & true`` is null).
        if isinstance(inner_node, ast.BinOp) and isinstance(
            inner_node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            _, left_type = self.analyze_select_expr(inner_node.left)
            _, right_type = self.analyze_select_expr(inner_node.right)
            resolved = [t for t in (left_type, right_type) if t is not None]
            return alias, _wrap_nullable_if_any(Boolean(), resolved)

        # ``~expr`` negates Booleans but operates BITWISE on integers,
        # preserving the dtype (issue #72) — same matrix as ``Expr.not_``;
        # see ``_NOT_VALID_RECEIVERS``. ``~null`` is null, so nullability
        # carries through.
        if isinstance(inner_node, ast.UnaryOp) and isinstance(inner_node.op, ast.Invert):
            _, operand_type = self.analyze_select_expr(inner_node.operand)
            return alias, self._not_dtype(operand_type, op_desc="~")

        # Python ``not expr`` -> Boolean. On a polars Expr it raises
        # TypeError at expression-construction time (``Expr.__bool__`` is
        # ambiguous), so this only ever reaches a frame for plain Python
        # truthiness — which IS a bool.
        if isinstance(inner_node, ast.UnaryOp) and isinstance(inner_node.op, ast.Not):
            _, operand_type = self.analyze_select_expr(inner_node.operand)
            resolved = [operand_type] if operand_type is not None else []
            return alias, _wrap_nullable_if_any(Boolean(), resolved)

        # Arithmetic binary operations like pl.col("x") * 2. Both operands
        # are resolved (keeping the missing-column error side-effects) and
        # classified three ways (issue #30): allowed -> result dtype,
        # known-invalid -> PLY009 + Unknown output, otherwise the legacy
        # promote-or-keep-left fallback.
        if isinstance(inner_node, ast.BinOp):
            left_name, left_type = self.analyze_select_expr(inner_node.left)
            _, right_type = self.analyze_select_expr(inner_node.right)
            output_name = alias if alias else left_name
            resolved = [t for t in (left_type, right_type) if t is not None]

            if left_type is not None and right_type is not None:
                left_inner = left_type.inner if isinstance(left_type, Nullable) else left_type
                right_inner = right_type.inner if isinstance(right_type, Nullable) else right_type
                # Null literals keep promote_types' Null -> Nullable[T]
                # rules — except next to a Decimal, where polars widens the
                # precision even against an all-null literal (probed:
                # Decimal(10,2) + None -> all-null Decimal(38,2)), so the
                # Decimal arm owns those cells (issue #52).
                has_null = isinstance(left_type, Null) or isinstance(right_type, Null)
                has_decimal = isinstance(left_inner, Decimal) or isinstance(right_inner, Decimal)
                if not has_null or has_decimal:
                    verdict = _arith_verdict(inner_node.op, left_inner, right_inner)
                    if verdict is _ARITH_INVALID:
                        op_sym = _OP_SYMBOLS.get(type(inner_node.op), "?")
                        self.errors.append(
                            tag(
                                PLY009,
                                f"arithmetic '{left_inner} {op_sym} {right_inner}' is not "
                                f"supported — polars raises InvalidOperationError at "
                                f"runtime; cast an operand first",
                            )
                        )
                        # The error is the signal; don't fabricate a dtype —
                        # the named output registers as Unknown downstream.
                        return output_name, None
                    if isinstance(verdict, DataType):
                        return output_name, _wrap_nullable_if_any(verdict, resolved)
                    # verdict is None — operand outside the understood set;
                    # fall through to the legacy behaviour below.

            if isinstance(inner_node.op, ast.Div):
                # polars true division always yields Float64, even for
                # int/int (issue #14) — ``//`` is the dtype-preserving one.
                if any(_base_is_unknown(t) for t in resolved):
                    return output_name, Unknown()
                if resolved:
                    return output_name, _wrap_nullable_if_any(Float64(), resolved)
                # Neither operand resolved — fall through so the output is
                # registered as Unknown downstream.
            elif left_type is not None and right_type is not None:
                try:
                    return output_name, promote_types(left_type, right_type)
                except TypePromotionError:
                    # Arithmetic outside the understood set (Decimal,
                    # List, exotic ints, ...): keep the left operand's
                    # dtype, but nullability from either side still
                    # propagates.
                    return output_name, _wrap_nullable_if_any(left_type, resolved)
            elif resolved:
                # Only one operand resolved — use its type (the historical
                # take-left behaviour, generalised to either side).
                return output_name, resolved[0]

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
                self.errors.append(tag(getattr(e, "code", PLY001), str(e)))
                return None, None

        # Check for pl.lit(value)
        lit_type = self._extract_lit_type(inner_node)
        if lit_type:
            return alias, lit_type

        # Unrecognised expression — surface the alias (if any) so a trailing
        # ``.alias("x")`` still names an Unknown-typed output column.
        return alias, None

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

        # Zero-arg ``pl.len()`` — row count, polars' IDX dtype (UInt32).
        # The default output column name is "len".
        if name == "len" and not node.args:
            return "len", UInt32()

        # ``pl.element()`` — the per-element placeholder inside
        # ``list.eval(...)`` bodies (issue #44). Resolves to the bound
        # element dtype when this analyzer was spun up for an eval body;
        # outside eval it stays unresolved (polars errors at runtime, but
        # flagging that is out of scope — silence avoids false positives).
        if name == "element" and not node.args and self.element_dtype is not None:
            return None, self.element_dtype

        # A list/tuple literal mixed with further positional args does NOT
        # flatten at runtime (issue #59) — flag it and degrade to Unknown
        # instead of typing the flatten that never happens.
        if name in _MULTI_EXPR_HELPERS and _mixed_list_args(node.args):
            self.errors.append(
                tag(
                    PLY017,
                    f"pl.{name}: a list literal mixed with other positional "
                    f"arguments is not flattened — polars raises TypeError "
                    f"(or misparses the list as a nested literal) at runtime; "
                    f"pass either varargs or one single list",
                )
            )
            return None, Unknown()

        if name == "concat_str" or name == "format":
            for arg in _flatten_expr_args(node.args):
                self._validate_subexpr(arg)
            return None, Utf8()

        if name == "struct":
            fields: dict[str, DataType] = {}
            for arg in _flatten_expr_args(node.args):
                # In expression-input position a bare string is a column
                # reference: ``pl.struct(["a", "b"])`` ≡
                # ``pl.struct(pl.col("a"), pl.col("b"))``.
                col = _str_constant(arg)
                if col is None:
                    col = self._extract_col_name(arg)
                if col is None:
                    continue
                try:
                    fields[col] = infer_col(col, self.current_frame)
                except ColumnNotFoundError as e:
                    self.errors.append(tag(getattr(e, "code", PLY001), str(e)))
            # Keyword args name the fields: ``pl.struct(x=expr)`` → field
            # ``x`` (issue #47). The value can be any expression; an
            # un-inferable one keeps the field as Unknown rather than
            # dropping it (issue #8 philosophy).
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                _, kw_dtype = self.analyze_select_expr(kw.value)
                fields[kw.arg] = kw_dtype if kw_dtype is not None else Unknown()
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
                    self.errors.append(tag(getattr(e, "code", PLY001), str(e)))
                    return col, None  # type: ignore[return-value]
                try:
                    result_type = infer_agg_result_type(
                        agg_func,
                        col_type,
                        context="agg" if self._in_agg_chain else "select",
                    )
                except GroupByTypeError as e:
                    self.errors.append(tag(PLY011, str(e)))
                    return col, None  # type: ignore[return-value]
                return col, result_type

        if name == "coalesce":
            args = _flatten_expr_args(node.args)
            inferred: list[DataType] = []
            any_non_nullable = False
            for arg in args:
                _, t = self._resolve_expr_or_col_str(arg)
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
            # Preserve the first-element column name as the default output
            # name (a bare string element names the output like pl.col does).
            first_name = None
            if args:
                first_name = self._extract_col_name(args[0]) or _str_constant(args[0])
            return first_name, unified

        if name == "concat_list":
            elem_types: list[DataType] = []
            for arg in _flatten_expr_args(node.args):
                _, t = self._resolve_expr_or_col_str(arg)
                if t is not None:
                    elem_types.append(t)
            if not elem_types:
                return None, ListT(Unknown())
            elem: DataType = elem_types[0]
            for t in elem_types[1:]:
                try:
                    elem = unify_types(elem, t)
                except TypeUnificationError:
                    elem = Unknown()
                    break
            return None, ListT(elem)

        if name in ("sum_horizontal", "min_horizontal", "max_horizontal", "mean_horizontal"):
            operands: list[DataType] = []
            for arg in _flatten_expr_args(node.args):
                _, t = self._resolve_expr_or_col_str(arg)
                if t is not None:
                    operands.append(t)
            if not operands:
                return None, Unknown()
            # Horizontal ops skip nulls: the result is null only when every
            # input is null, so the result is Nullable only if *all* resolved
            # operands are. Strip per-operand wrappers before promotion.
            all_nullable = all(isinstance(t, Nullable) for t in operands)
            inners = [t.inner if isinstance(t, Nullable) else t for t in operands]
            result: DataType
            if name == "mean_horizontal":
                # Probed (polars 1.41.2; backlog N-4): Float32 iff every
                # operand is Float32; any other operand widens to Float64.
                result = Float32() if all(isinstance(t, Float32) for t in inners) else Float64()
            else:
                result = inners[0]
                for t in inners[1:]:
                    try:
                        result = promote_types(result, t)
                    except TypePromotionError:
                        result = Unknown()
                        break
            if all_nullable and not isinstance(result, Unknown):
                result = Nullable(result)
            return None, result

        return None

    def _resolve_expr_or_col_str(self, node: ast.expr) -> tuple[str | None, DataType | None]:
        """Resolve one element of a multi-expression helper's arguments.

        polars treats a bare string in expression-input position as a column
        reference (``pl.coalesce(["a", "b"])`` ≡
        ``pl.coalesce(pl.col("a"), pl.col("b"))``), so strings resolve via
        the frame — with PLY001 on closed frames, mirroring the ``pl.col``
        path. Everything else goes through ``analyze_select_expr``.
        """
        s = _str_constant(node)
        if s is not None:
            try:
                return s, infer_col(s, self.current_frame)
            except ColumnNotFoundError as e:
                self.errors.append(tag(getattr(e, "code", PLY001), str(e)))
                return s, None
        return self.analyze_select_expr(node)

    def _eval_body_dtype(
        self, element_dtype: DataType, call_node: ast.Call | None
    ) -> DataType | None:
        """Type-check a ``list.eval`` / ``arr.eval`` body (issue #44 / #53).

        Runs the body through a child analyzer with ``pl.element()`` bound
        to the container's element dtype; the child's errors (e.g. PLY009
        from Int+String arithmetic) bubble up. Returns the body dtype, or
        ``None`` when it cannot be resolved.
        """
        if call_node is None or not call_node.args:
            return None
        child = ExpressionAnalyzer(
            self.current_frame,
            warnings=self.warnings,
            registry=self.registry,
            element_dtype=element_dtype,
            int_consts=self.int_consts,
        )
        _, body_dtype = child.analyze_select_expr(call_node.args[0])
        self.errors.extend(child.errors)
        return body_dtype

    def _container_agg_result(
        self, namespace: str, method: str, receiver_inner: ListT | Array
    ) -> DataType | None:
        """Verdict for a ``list.<agg>()`` / ``arr.<agg>()`` reduction (issue #55).

        Probed-invalid cells flag PLY016 — the runtime error class varies
        per cell (InvalidOperationError / ComputeError / rust panic) — and
        degrade the output to Unknown. Unclaimed cells return ``None``
        (silent Unknown downstream). The element's own Nullable wrapper
        does not change the verdict.
        """
        element = receiver_inner.inner
        if isinstance(element, Nullable):
            element = element.inner
        verdict = container_agg_return(namespace, method, element)
        if isinstance(verdict, ContainerAggInvalid):
            self.errors.append(
                tag(
                    PLY016,
                    f"{namespace}.{method}: operation not supported for dtype "
                    f"{receiver_inner} — polars raises an error at runtime",
                )
            )
            return Unknown()
        return verdict

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
        passed in so per-namespace handlers that need arguments
        (``struct.field`` / ``struct.rename_fields``, ``list.eval``,
        ``str.to_datetime``, ``dt.replace_time_zone`` /
        ``dt.convert_time_zone``) can read them.
        """
        receiver_inner = receiver_type
        receiver_is_nullable = False
        if isinstance(receiver_type, Nullable):
            receiver_inner = receiver_type.inner
            receiver_is_nullable = True

        result: DataType | None = None

        if namespace == "bin":
            result = self._BIN_RETURN.get(method)
        elif namespace == "str":
            if method == "to_datetime":
                # The output tz depends on the arguments (issue #50
                # collateral): default naive, ``time_zone=`` literal sets
                # the tz, a format literal containing ``%z`` is probed to
                # yield Datetime[UTC]. Unknowable arguments -> Unknown.
                result = _str_to_datetime_dtype(call_node)
            elif method == "to_decimal":
                # The scale comes from the (required, keyword-only)
                # ``scale=`` argument (issue #61) — Decimal(38, scale) for
                # an int literal, Unknown otherwise.
                result = _str_to_decimal_dtype(call_node)
            else:
                result = self._STR_RETURN.get(method)
        elif namespace == "dt":
            if method in ("replace_time_zone", "convert_time_zone") and isinstance(
                receiver_inner, Datetime
            ):
                # These SET the tz — blanket receiver-preservation would
                # claim the old tz and manufacture false positives now
                # that tz mismatches are flagged (issue #50 collateral).
                # The receiver's time unit is preserved (issue #66).
                result = _time_zone_arg_dtype(method, call_node, receiver_inner)
            elif method == "epoch":
                # Argument-dependent (issue #73): "d" -> Int32, the
                # sub-second units -> Int64 — dispatched before the fixed
                # table below.
                result = _dt_epoch_dtype(call_node)
            elif method in self._DT_RETURN:
                result = self._DT_RETURN[method]
            elif method in self._DT_PRESERVING and receiver_inner is not None:
                result = receiver_inner
        elif namespace == "list":
            if method == "len":
                result = UInt32()
            elif method == "eval" and isinstance(receiver_inner, ListT):
                # ``list.eval(body)`` runs ``body`` element-wise with
                # ``pl.element()`` bound to the list's inner dtype
                # (issue #44). A child analyzer type-checks the body —
                # its errors (e.g. PLY009 from Int+String arithmetic)
                # bubble up — and the result is List(body dtype);
                # an unresolvable body degrades to List(Unknown).
                body_dtype = self._eval_body_dtype(receiver_inner.inner, call_node)
                result = ListT(body_dtype if body_dtype is not None else Unknown())
            elif method in self._LIST_PRESERVING and receiver_inner is not None:
                result = receiver_inner
            elif isinstance(receiver_inner, ListT):
                if method in self._LIST_ELEMENT_RETURN:
                    result = receiver_inner.inner
                elif method in CONTAINER_AGG_METHODS:
                    result = self._container_agg_result("list", method, receiver_inner)
        elif namespace == "arr":
            # Issue #53: the polars arr namespace mirrors most of the list
            # namespace but is dispatched separately — several methods
            # de-array into a List, and the receiver must be Array.
            if method in ARR_NAMESPACE_RETURN:
                result = ARR_NAMESPACE_RETURN[method]
            elif method == "eval" and isinstance(receiver_inner, Array):
                # ``arr.eval(body)`` runs ``body`` element-wise like
                # ``list.eval`` — the body is type-checked with
                # ``pl.element()`` bound to the element dtype and its
                # errors bubble up. Probed (polars 1.41.2): the default
                # keeps the Array container; ``as_list=True`` (keyword-
                # only, added in polars 1.41 — issue #53) yields
                # ``List(body dtype)`` instead, for dtype-changing,
                # aggregating AND length-changing bodies alike (probe:
                # ``arr.eval(pl.element().filter(...), as_list=True)``
                # -> List(Int64)). A non-literal ``as_list`` leaves the
                # container kind unknowable -> Unknown.
                body_dtype = self._eval_body_dtype(receiver_inner.inner, call_node)
                element = body_dtype if body_dtype is not None else Unknown()
                as_list_node = (
                    next((kw.value for kw in call_node.keywords if kw.arg == "as_list"), None)
                    if call_node is not None
                    else None
                )
                if as_list_node is None:
                    result = Array(element, receiver_inner.width)
                elif isinstance(as_list_node, ast.Constant) and isinstance(
                    as_list_node.value, bool
                ):
                    result = (
                        ListT(element)
                        if as_list_node.value
                        else Array(element, receiver_inner.width)
                    )
                else:
                    result = Unknown()
            elif method in ARR_NAMESPACE_PRESERVING and receiver_inner is not None:
                result = receiver_inner
            elif isinstance(receiver_inner, Array):
                element = receiver_inner.inner
                if method in ARR_NAMESPACE_ELEMENT_RETURN:
                    result = element
                elif method in ARR_NAMESPACE_TO_LIST:
                    result = ListT(element)
                elif method in CONTAINER_AGG_METHODS:
                    result = self._container_agg_result("arr", method, receiver_inner)
        elif namespace == "cat":
            # Issue #54. ``get_categories`` returns the category list (one
            # row per category): length-changing, and its output carries no
            # nulls even for a nullable receiver (probed) — skip the
            # receiver-nullability wrap.
            result = self._CAT_RETURN.get(method)
            if method == "get_categories":
                return result
        elif namespace == "struct":
            if method == "field" and call_node is not None and call_node.args:
                field_name = _str_constant(call_node.args[0])
                if field_name is not None and isinstance(receiver_inner, Struct):
                    field_dtype = receiver_inner.fields.get(field_name)
                    if field_dtype is None:
                        if receiver_inner.open:
                            # OPEN struct (backlog C-9): the field may
                            # exist among the unknown ones — ADR-0006
                            # assumption semantics.
                            field_dtype = Unknown()
                        else:
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
            elif method == "rename_fields":
                # Probed (polars 1.41.2; issue #48): the new names are
                # applied positionally to the existing fields. Length
                # mismatches do NOT raise — fewer names truncate the struct
                # to the renamed prefix, surplus names are ignored — exactly
                # ``zip`` semantics. Non-literal names or an unresolved
                # receiver degrade to Unknown: keeping the original field
                # names was issue #48's false positive.
                new_names: list[str] | None = None
                if call_node is not None and call_node.args:
                    new_names = _str_list_or_tuple(call_node.args[0])
                if (
                    new_names is not None
                    and isinstance(receiver_inner, Struct)
                    and not receiver_inner.open
                ):
                    # strict=False: zip-truncation IS the probed semantics.
                    result = Struct(
                        dict(zip(new_names, receiver_inner.fields.values(), strict=False))
                    )
                else:
                    result = Unknown()

        if result is None:
            return None
        if receiver_is_nullable and not isinstance(result, (Nullable, Unknown)):
            return Nullable(result)
        return result

    def _not_dtype(self, receiver_type: DataType | None, *, op_desc: str) -> DataType | None:
        """Result dtype of ``~expr`` / ``Expr.not_()`` (issue #72).

        Every valid receiver (Boolean + all integer widths, see
        ``_NOT_VALID_RECEIVERS``) is dtype-preserving — the receiver
        instance is returned so Nullable wrappers (and any future
        parameters) flow through. Invalid receivers are a guaranteed
        runtime InvalidOperationError -> PLY016 and the output degrades
        to Unknown; Unknown / unresolved receivers stay silent.
        """
        if receiver_type is None:
            return None
        inner = receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
        if isinstance(inner, Unknown):
            return receiver_type
        if isinstance(inner, self._NOT_VALID_RECEIVERS):
            return receiver_type
        self.errors.append(
            tag(
                PLY016,
                f"{op_desc}: operation not supported for dtype {inner} — "
                f"polars raises InvalidOperationError at runtime",
            )
        )
        return None

    def _validate_subexpr(self, node: ast.expr) -> None:
        """Run a sub-expression through analyze_select_expr to surface column errors.

        We discard the type/name; the only side-effect of interest is appending
        to ``self.errors`` when ``pl.col("missing")`` shows up.
        """
        self.analyze_select_expr(node)

    def _check_over_key(self, node: ast.expr) -> None:
        """Validate one ``over`` partition / order key (issue #32).

        A string constant is a column name — checked through ``infer_col``
        so a missing column is PLY001 on a closed frame and silently
        ``Unknown`` on an open one. List/tuple literals are checked
        elementwise. Anything else (``pl.col(...)`` chains etc.) is walked
        through the expression analyzer for its error side effects; the
        ExpressionAnalyzer has no constant environment, so a non-literal
        name stays silent (no false positives).
        """
        s = _str_constant(node)
        if s is not None:
            try:
                infer_col(s, self.current_frame)
            except ColumnNotFoundError as e:
                self.errors.append(tag(getattr(e, "code", PLY001), str(e)))
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                self._check_over_key(elt)
            return
        self._validate_subexpr(node)

    def _over_receiver_cardinality(self, node: ast.expr) -> str:
        """Per-group cardinality of a windowed expression (issue #45).

        Returns ``"scalar"`` (one value per group — ``over`` broadcasts it
        under every mapping strategy), ``"vector"`` (length-preserving or
        multi-valued — ``mapping_strategy="join"`` gathers it into a List
        per row), or ``"unknown"`` (never guess; the caller degrades the
        dtype to Unknown). Probed against polars 1.41.2 — see
        ``_OVER_SCALAR_METHODS`` / ``_OVER_VECTOR_METHODS``.
        """
        # Bare literal: broadcastable scalar.
        if isinstance(node, ast.Constant):
            return "scalar"
        # Elementwise operators preserve cardinality; an expression is
        # vector as soon as one operand is (``col - col.mean()`` is
        # elementwise even though one side aggregates — probed).
        if isinstance(node, ast.UnaryOp):
            return self._over_receiver_cardinality(node.operand)
        if isinstance(node, (ast.BinOp, ast.Compare)):
            if isinstance(node, ast.BinOp):
                parts = [node.left, node.right]
            else:
                parts = [node.left, *node.comparators]
            kinds = {self._over_receiver_cardinality(part) for part in parts}
            if "unknown" in kinds:
                return "unknown"
            return "vector" if "vector" in kinds else "scalar"
        # Sub-namespace accessor (``.str`` / ``.dt`` / ...): the namespace
        # methods are elementwise — cardinality comes from the receiver.
        if isinstance(node, ast.Attribute) and node.attr in _NAMESPACE_VALID_DTYPES:
            return self._over_receiver_cardinality(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if isinstance(func.value, ast.Name) and func.value.id == "pl":
                if func.attr in ("col", "nth", "element"):
                    return "vector"
                if func.attr == "lit":
                    return "scalar"
                # ``pl.sum("a")`` shorthands and zero-arg ``pl.len()`` /
                # ``pl.count()`` reduce the group to one value (probed).
                if func.attr in ("len", "count") or func.attr in AGG_SHORTHAND_NAMES:
                    return "scalar"
                # ``pl.when(...)`` roots and anything else: branch values
                # decide the cardinality — not inspected here.
                return "unknown"
            method = canonicalize_method(func.attr)
            if method in self._OVER_SCALAR_METHODS:
                return "scalar"
            if method in self._OVER_VECTOR_METHODS:
                return "vector"
            if method in self._OVER_OPAQUE_CARDINALITY_METHODS:
                return "unknown"
            # Every other recognised expression method is elementwise —
            # cardinality passes through to its own receiver. (A chain
            # containing a method dtype inference does not understand
            # already resolves to a None dtype, so a misjudged tail here
            # cannot fabricate a dtype.)
            return self._over_receiver_cardinality(func.value)
        return "unknown"

    def _is_in_element_dtype(self, arg: ast.expr) -> DataType | None:
        """Element dtype of an ``is_in`` argument, or ``None`` when not understood.

        - list/tuple/set literal of constants → ``infer_lit`` per element,
          folded with ``unify_types`` (``[1, None]`` → ``Nullable[Int64]``;
          the caller unwraps). Empty, non-constant or non-unifiable → ``None``.
        - any other expression → its inferred dtype; a ``List(T)`` or
          ``Array(T)`` expr contributes ``T`` (both probed), and a
          non-container expr of dtype ``T`` is imploded by polars so it
          contributes ``T`` as well (probed).
        - a bare scalar constant is outside the understood surface → ``None``.
        """
        if isinstance(arg, (ast.List, ast.Tuple, ast.Set)):
            unified: DataType | None = None
            for elt in arg.elts:
                if not isinstance(elt, ast.Constant):
                    return None
                value = elt.value
                if value is not None and not isinstance(value, (bool, int, float, str)):
                    return None
                element = infer_lit(value)
                if unified is None:
                    unified = element
                    continue
                try:
                    unified = unify_types(unified, element)
                except TypeUnificationError:
                    return None
            return unified
        if isinstance(arg, ast.Constant):
            return None
        _, arg_dtype = self.analyze_select_expr(arg)
        if arg_dtype is None:
            return None
        inner = arg_dtype.inner if isinstance(arg_dtype, Nullable) else arg_dtype
        if isinstance(inner, (ListT, Array)):
            return inner.inner
        return inner

    @staticmethod
    def _parse_when_chain(
        node: ast.expr,
    ) -> tuple[list[ast.Call], list[ast.expr], ast.expr | None] | None:
        """Parse a ``pl.when(c1).then(v1)[.when(c2).then(v2)...][.otherwise(v)]``
        chain into ``(when_calls, then_values, otherwise_value)``.

        Returns ``None`` when ``node`` is not such a chain (the walk must
        terminate at a ``pl.when(...)`` root). ``when_calls`` / ``then_values``
        are in source order; ``otherwise_value`` is ``None`` for the
        no-otherwise form.
        """
        otherwise_node: ast.expr | None = None
        cur: ast.expr = node
        if (
            isinstance(cur, ast.Call)
            and isinstance(cur.func, ast.Attribute)
            and cur.func.attr == "otherwise"
        ):
            if len(cur.args) != 1 or cur.keywords:
                return None
            otherwise_node = cur.args[0]
            cur = cur.func.value

        when_calls: list[ast.Call] = []
        then_nodes: list[ast.expr] = []
        while True:
            if not (
                isinstance(cur, ast.Call)
                and isinstance(cur.func, ast.Attribute)
                and cur.func.attr == "then"
            ):
                return None
            if len(cur.args) != 1 or cur.keywords:
                return None
            then_nodes.append(cur.args[0])
            cur = cur.func.value
            if not (
                isinstance(cur, ast.Call)
                and isinstance(cur.func, ast.Attribute)
                and cur.func.attr == "when"
            ):
                return None
            when_calls.append(cur)
            base = cur.func.value
            if isinstance(base, ast.Name) and base.id == "pl":
                break
            cur = base

        # The walk above went outside-in; restore source order.
        when_calls.reverse()
        then_nodes.reverse()
        return when_calls, then_nodes, otherwise_node

    def _analyze_when_chain(self, node: ast.expr) -> tuple[str | None, DataType | None] | None:
        """Type a ``pl.when(...).then(...)[...].otherwise(...)`` chain
        (issues #37/#40).

        Probed (polars 1.41.2):
          - a non-Boolean condition raises ``SchemaError: expected Boolean``
            -> PLY008, same gap class as ``filter`` (#28). Bare strings in
            ``when(...)`` are column references; kwargs are equality
            constraints (boolean by construction).
          - the result dtype is the common supertype of every then-value
            plus the otherwise value (``then(pl.lit(1)).otherwise(pl.lit("x"))``
            -> String); strings in then/otherwise are *column references*.
          - a null condition row takes the otherwise branch
            (``[True, None, False]`` -> ``[10, 20, 20]``, null_count 0), so a
            Nullable condition does NOT make the result nullable.
          - without ``.otherwise(...)`` the unmatched rows are null
            (``[10, None, None]``) -> the result is Nullable.

        Returns ``None`` when ``node`` is not a when-chain, ``(None, None)``
        when it is one but a branch (or the supertype fold) is unresolved —
        the caller keeps today's Unknown registration. The default output
        name (polars' ``literal`` / first-branch column name) is deliberately
        not modelled; kwarg/alias naming works as before.
        """
        parsed = self._parse_when_chain(node)
        if parsed is None:
            return None
        when_calls, then_nodes, otherwise_node = parsed

        # Conditions: every positional predicate must be Boolean (issue #37).
        for when_call in when_calls:
            for arg in when_call.args:
                _, cond_dtype = self._resolve_expr_or_col_str(arg)
                cond_error = _nonboolean_predicate_error(cond_dtype, op="when", noun="condition")
                if cond_error is not None:
                    self.errors.append(cond_error)
            for kw in when_call.keywords:
                self._validate_subexpr(kw.value)

        # Branches: fold the supertype over all then-values + otherwise
        # (issue #40). Every branch is walked even after one fails to
        # resolve, so missing-column references keep surfacing PLY001.
        branch_nodes = list(then_nodes)
        if otherwise_node is not None:
            branch_nodes.append(otherwise_node)
        branch_types: list[DataType] = []
        unresolved = False
        for branch in branch_nodes:
            _, branch_dtype = self._resolve_expr_or_col_str(branch)
            if branch_dtype is None:
                unresolved = True
            else:
                branch_types.append(branch_dtype)
        if unresolved or not branch_types:
            return None, None

        folded: DataType = branch_types[0]
        for branch_dtype in branch_types[1:]:
            merged = supertype(folded, branch_dtype)
            if merged is None:
                # No polars supertype — the runtime error is out of scope
                # here; keep the silent Unknown registration.
                return None, None
            folded = merged
        if _base_is_unknown(folded):
            return None, Unknown()
        if otherwise_node is None and not isinstance(folded, (Nullable, Null)):
            folded = Nullable(folded)
        return None, folded

    def _shift_fill_dtype(self, receiver_type: DataType, fill_node: ast.expr) -> DataType | None:
        """Result dtype of ``shift(n, fill_value=<fill_node>)`` (issue #43).

        Returns ``None`` when the fill is a null literal (``fill_value=None``
        / ``pl.lit(None)``) — probed to behave exactly like no fill, so the
        caller falls back to the Nullable wrap. Bare constants and
        ``pl.lit(<const>)`` take the probed literal-leniency rules; any other
        expression resolves through the analyzer and follows the supertype
        matrix (see ``infer_shift_fill``). An unresolved fill keeps the
        receiver dtype: the slots are filled with *something*, so no
        Nullable wrap, and claiming any other dtype would be a guess.
        """
        lit_dtype: DataType | None = None
        if isinstance(fill_node, ast.Constant) and (
            fill_node.value is None or isinstance(fill_node.value, (bool, int, float, str))
        ):
            lit_dtype = infer_lit(fill_node.value)
        else:
            lit_dtype = self._extract_lit_type(fill_node)

        if lit_dtype is not None:
            if isinstance(lit_dtype, Null):
                return None
            return infer_shift_fill(receiver_type, lit_dtype, fill_is_literal=True)

        _, fill_dtype = self.analyze_select_expr(fill_node)
        if fill_dtype is None:
            inner, nullable = (
                (receiver_type.inner, True)
                if isinstance(receiver_type, Nullable)
                else (receiver_type, False)
            )
            return Nullable(inner) if nullable else inner
        if isinstance(fill_dtype, Null):
            return None
        return infer_shift_fill(receiver_type, fill_dtype, fill_is_literal=False)

    def _analyze_name_method(
        self, method: str, inner_expr: ast.expr, call_node: ast.Call
    ) -> tuple[str | None, DataType | None] | None:
        """Resolve ``<expr>.name.<method>(...)`` to ``(output_name, dtype)``
        (issue #56).

        Probed (polars 1.41.2): the rename applies to the expression's
        CURRENT output name — an earlier ``.alias`` included — while
        ``keep`` restores the chain's ROOT column name, overriding any
        earlier ``.alias`` or name transform. A trailing ``.alias`` after
        the name method wins (handled naturally: ``analyze_select_expr``
        strips it before this runs). The dtype is never changed; the
        ``*_fields`` variants transform a Struct receiver's FIELD names
        instead (dtype transform, output name unchanged).

        ``(None, dtype)`` means the output name is unknowable (non-literal
        argument, ``map``, unmodeled methods like ``replace``) — the
        select / with_columns layer opens the result frame so the column
        is not provably absent downstream. Returns ``None`` when the
        receiver expression itself isn't recognised.
        """
        inner_name, inner_type = self.analyze_select_expr(inner_expr)
        if inner_name is None and inner_type is None:
            # Unrecognised receiver (a bare selector reaching expression
            # level, a frame variable, ...) — not claimed as a chain.
            return None
        if method == "keep":
            root = self._find_deep_col(inner_expr)
            return (root if root is not None else inner_name), inner_type
        if method in ("prefix", "suffix"):
            arg = _call_arg(call_node, position=0, name=method)
            lit = _str_constant(arg) if arg is not None else None
            if lit is None or inner_name is None:
                return None, inner_type
            new_name = lit + inner_name if method == "prefix" else inner_name + lit
            return new_name, inner_type
        if method == "to_uppercase":
            return (inner_name.upper() if inner_name is not None else None), inner_type
        if method == "to_lowercase":
            return (inner_name.lower() if inner_name is not None else None), inner_type
        if method in ("prefix_fields", "suffix_fields"):
            kwarg = "prefix" if method == "prefix_fields" else "suffix"
            arg = _call_arg(call_node, position=0, name=kwarg)
            lit = _str_constant(arg) if arg is not None else None
            base = inner_type.inner if isinstance(inner_type, Nullable) else inner_type
            if lit is not None and isinstance(base, Struct):
                renamed = Struct(
                    {
                        (lit + f if method == "prefix_fields" else f + lit): t
                        for f, t in base.fields.items()
                    }
                )
                result: DataType = (
                    Nullable(renamed) if isinstance(inner_type, Nullable) else renamed
                )
                return inner_name, result
            # Non-literal argument, or a non-Struct / unresolved receiver
            # (a probed runtime InvalidOperationError, but flagging it is
            # out of scope) — the field names are unknowable.
            return inner_name, None
        if method in ("map", "map_fields"):
            unknowable = "column name" if method == "map" else "struct field names"
            self.warnings.append(
                tag(
                    PLW004,
                    f"name.{method}: the callable cannot be evaluated statically, "
                    f"so the output {unknowable} cannot be inferred. Rename with "
                    f"explicit `.alias(...)` / literal prefix-suffix methods to "
                    f"keep the schema precise.",
                )
            )
            if method == "map_fields":
                return inner_name, None
            return None, inner_type
        # Unmodeled ``.name`` method (``replace``, future additions): every
        # name method preserves the dtype; the output name is unknowable.
        return None, inner_type

    def _analyze_method_chain(self, node: ast.expr) -> tuple[str | None, DataType | None] | None:
        """Analyze ``pl.col("x").<method>(...)`` style chains.

        Returns ``(default_name, dtype)`` or ``None`` if the node isn't a
        recognised method chain. ``(name, None)`` means the node *is* a
        chain on that column but the method's result dtype is unknown —
        the caller registers the column as ``Unknown`` so later references
        still resolve.
        """
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, ast.Attribute):
            return None
        method = node.func.attr
        receiver = node.func.value

        # when/then/otherwise chains (issues #37/#40), recognised by walking
        # the chain back to the ``pl.when(...)`` root. Handled before the
        # generic receiver analysis below — the partial chain links
        # (``pl.when(...)`` / ``.then(...)``) don't resolve on their own.
        if method in ("then", "otherwise"):
            when_result = self._analyze_when_chain(node)
            if when_result is not None:
                return when_result

        # Sub-namespace: ``pl.col("x").str.contains(...)``,
        # ``pl.col("ts").dt.year()``, ``pl.col("xs").list.get(0)``,
        # ``pl.col("s").struct.field("name")``, ``pl.col("b").bin.size()``
        # etc. — the recognised accessor names are exactly the keys of the
        # receiver-validity table.
        if isinstance(receiver, ast.Attribute) and receiver.attr in _NAMESPACE_VALID_DTYPES:
            ns = receiver.attr
            col_name, col_type = self.analyze_select_expr(receiver.value)
            # Receiver-dtype validation (issue #31): a known dtype outside
            # the namespace's accepted set is a guaranteed runtime
            # InvalidOperationError. Unknown / unresolved receivers stay
            # silent; valid receivers fall through to dispatch unchanged.
            if col_type is not None and not _base_is_unknown(col_type):
                base = col_type.inner if isinstance(col_type, Nullable) else col_type
                if not isinstance(base, _NAMESPACE_VALID_DTYPES[ns]):
                    subject = f"column '{col_name}'" if col_name else "the receiver expression"
                    runtime_error = _NAMESPACE_RUNTIME_ERROR.get(ns, "InvalidOperationError")
                    self.errors.append(
                        tag(
                            PLY012,
                            f"`.{ns}` accessor requires "
                            f"{_NAMESPACE_EXPECTED_DESC[ns]}, but {subject} is "
                            f"{col_type} — polars raises {runtime_error} "
                            f"at runtime",
                        )
                    )
                    # One clear error; the output column degrades to Unknown.
                    return col_name, None
            # Struct.field needs the field name (positional arg) — pass the
            # whole call node so the dispatcher can look at it.
            ns_result = self._dispatch_namespace_method(ns, method, col_type, node)
            if ns_result is None:
                # Unmodeled namespace method on a receiver that passed the
                # dtype-validity gate above: the column silently degrades —
                # warn (backlog B-4). Unresolved/Unknown receivers stay
                # silent (the degradation happened upstream).
                if col_type is not None and not _base_is_unknown(col_type):
                    self.warnings.append(_unmodeled_method_warning(f".{ns}.{method}()"))
                return None
            return col_name, ns_result

        # ``.name`` namespace (issue #56): renames the OUTPUT column (or
        # struct FIELDS for the ``*_fields`` variants) — the dtype is
        # never changed. NOT a dtype-validated namespace (works on any
        # column), so it is deliberately absent from _NAMESPACE_VALID_DTYPES.
        if isinstance(receiver, ast.Attribute) and receiver.attr == "name":
            name_result = self._analyze_name_method(method, receiver.value, node)
            if name_result is not None:
                return name_result

        # Warning watermark: the ``cast`` branch below retracts a PLW007 the
        # receiver analysis emits when the cast repairs the degradation.
        warnings_before_receiver = len(self.warnings)
        receiver_result = self.analyze_select_expr(receiver)
        receiver_name, receiver_type = receiver_result
        if receiver_type is None and receiver_name is None:
            # Receiver wasn't recognised — bail out.
            return None

        # ``is_in``: validate the element dtype against the receiver dtype
        # (issue #33) before falling through to the Boolean predicate path.
        # Unresolvable arguments / receivers stay silent.
        if method == "is_in" and node.args and receiver_type is not None:
            element_dtype = self._is_in_element_dtype(node.args[0])
            if element_dtype is not None:
                receiver_inner = (
                    receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
                )
                element_inner = (
                    element_dtype.inner if isinstance(element_dtype, Nullable) else element_dtype
                )
                if _is_in_invalid(receiver_inner, element_inner):
                    self.errors.append(
                        tag(
                            PLY009,
                            f"is_in: cannot check for List({element_inner}) values in "
                            f"{receiver_inner} data — polars raises "
                            f"InvalidOperationError at runtime; cast an operand first",
                        )
                    )
            # The result dtype is Boolean either way — handled just below.

        # Boolean predicates always produce Boolean; column name carried through.
        if method in self._BOOLEAN_PREDICATE_METHODS:
            return receiver_name, Boolean()

        # ``not_`` left the Boolean-predicate family (issue #72): Boolean
        # receivers negate, integer receivers get a dtype-preserving
        # bitwise NOT, everything else raises — see _NOT_VALID_RECEIVERS.
        if method == "not_":
            return receiver_name, self._not_dtype(receiver_type, op_desc="not_")

        # fill_null / fill_nan strip the Nullable wrapper.
        if method in ("fill_null", "fill_nan"):
            inner_dtype = receiver_type
            if isinstance(receiver_type, Nullable):
                inner_dtype = receiver_type.inner
            return receiver_name, inner_dtype if inner_dtype is not None else Boolean()

        # ``Expr.filter(...)`` is row-subsetting: the dtype is preserved
        # (Nullable wrapper included — nulls may survive the predicate).
        # Predicate sub-expressions are validated so missing-column refs
        # surface PLY001 (issue #23: conditional aggregation chains like
        # ``pl.col("v").filter(pred).sum()``) and a known non-Boolean
        # predicate dtype surfaces PLY008 (issue #28). A bare string is a
        # column reference, not a Utf8 literal. Kwargs are equality
        # constraints — boolean by construction, no dtype flag.
        if method == "filter":
            for arg in node.args:
                _, pred_dtype = self._resolve_expr_or_col_str(arg)
                pred_error = _nonboolean_predicate_error(pred_dtype)
                if pred_error is not None:
                    self.errors.append(pred_error)
            for kw in node.keywords:
                self._validate_subexpr(kw.value)
            return receiver_name, receiver_type

        # ``Expr.drop_nulls()`` removes the null rows, so the result keeps
        # the receiver dtype with the Nullable wrapper stripped.
        if method == "drop_nulls":
            if isinstance(receiver_type, Nullable):
                return receiver_name, receiver_type.inner
            return receiver_name, receiver_type

        # Float-returning numeric methods. Strictly typed receivers
        # (issue #62) — see the probed matrix on
        # ``_FLOAT_RETURN_INVALID_RECEIVERS``; Unknown / unresolved
        # receivers stay silent (Float64, the pre-existing default). A
        # Float32 receiver keeps Float32 (probed); every other accepted
        # receiver yields Float64.
        if method in self._FLOAT_RETURN_METHODS:
            inner = receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
            if inner is not None and isinstance(
                inner, self._FLOAT_RETURN_INVALID_RECEIVERS[method]
            ):
                self.errors.append(
                    tag(
                        PLY016,
                        f"{method}: operation not supported for dtype {inner} — "
                        f"polars raises an error at runtime",
                    )
                )
                return receiver_name, None
            result: DataType = Float32() if isinstance(inner, Float32) else Float64()
            if isinstance(receiver_type, Nullable):
                result = Nullable(result)
            return receiver_name, result

        # ``Expr.over(...)`` — windowed expression. Every partition / order
        # key must exist on the frame (issue #32; same gap class as sort
        # #29). Keys come from the positional args plus the
        # ``partition_by=`` / ``order_by=`` kwargs — polars resolves string
        # kwargs as column names too (verified 1.41.2).
        #
        # The result dtype depends on ``mapping_strategy=`` (issue #45;
        # probed): the default "group_to_rows" and "explode" preserve the
        # dtype; "join" gathers a multi-valued expression into a
        # ``List(<element dtype>)`` per row (inner nullability kept inside
        # the list) but broadcasts a scalar-per-group expression unchanged.
        # An unknown / non-literal strategy degrades to Unknown.
        if method == "over":
            # Grouped-context panic cells (backlog N-5): mean/median/quantile
            # on Float16 and product on UInt128 panic in rust inside ANY
            # grouped evaluation — over windows included, not just
            # group_by().agg() (probed, polars 1.41.2). The receiver's dtype
            # was inferred with the lenient "select" context; re-check the
            # cell here. Every panic cell is width-preserving, so the
            # aggregation's output dtype (= ``receiver_type``) equals its
            # input dtype.
            if (
                receiver_type is not None
                and isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Attribute)
            ):
                over_agg = agg_function_for(receiver.func.attr)
                if over_agg is not None and grouped_agg_panics(over_agg, receiver_type):
                    self.errors.append(
                        tag(
                            PLY011,
                            f"over: {receiver.func.attr} on {receiver_type} panics in "
                            f"rust at runtime in a grouped (window) context — probed "
                            f"(polars 1.41.2). The same reduction is valid in a plain "
                            f"select; cast the column to a wider dtype first",
                        )
                    )
            key_nodes: list[ast.expr] = list(node.args)
            strategy: str | None = "group_to_rows"
            for kw in node.keywords:
                if kw.arg in ("partition_by", "order_by"):
                    key_nodes.append(kw.value)
                elif kw.arg == "mapping_strategy":
                    strategy = _str_constant(kw.value)
            for key_node in key_nodes:
                self._check_over_key(key_node)
            if strategy in ("group_to_rows", "explode"):
                return receiver_name, receiver_type
            if strategy == "join":
                cardinality = self._over_receiver_cardinality(receiver)
                if cardinality == "scalar":
                    return receiver_name, receiver_type
                if cardinality == "vector" and receiver_type is not None:
                    return receiver_name, ListT(receiver_type)
                return receiver_name, None
            return receiver_name, None

        # Dtype-preserving methods. The numeric-only elementwise members
        # are strictly typed (issue #62) — see the probed matrix on
        # ``_ELEMENTWISE_INVALID_RECEIVERS``. An invalid receiver dtype is
        # a guaranteed runtime InvalidOperationError -> PLY016 and the
        # output degrades to Unknown; Unknown receivers stay silent.
        if method in self._DTYPE_PRESERVING_METHODS and receiver_type is not None:
            invalid = self._ELEMENTWISE_INVALID_RECEIVERS.get(method)
            if invalid is not None:
                inner = (
                    receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
                )
                if isinstance(inner, invalid):
                    self.errors.append(
                        tag(
                            PLY016,
                            f"{method}: operation not supported for dtype {inner} — "
                            f"polars raises InvalidOperationError at runtime",
                        )
                    )
                    return receiver_name, None
            return receiver_name, receiver_type

        # Cumulative reducers are strictly typed (issue #49) — see the
        # probed matrix on ``_CUM_INVALID_RECEIVERS``. An invalid receiver
        # dtype is a guaranteed runtime InvalidOperationError -> PLY016 and
        # the output degrades to Unknown; Unknown receivers stay silent.
        if method in self._CUM_INVALID_RECEIVERS and receiver_type is not None:
            inner = receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
            if isinstance(inner, Unknown):
                return receiver_name, receiver_type
            if isinstance(inner, self._CUM_INVALID_RECEIVERS[method]):
                self.errors.append(
                    tag(
                        PLY016,
                        f"{method}: operation not supported for dtype {inner} — "
                        f"polars raises InvalidOperationError at runtime",
                    )
                )
                return receiver_name, None
            result_inner: DataType = inner
            if method == "cum_sum":
                if isinstance(inner, self._CUM_SUM_INT64_RECEIVERS):
                    result_inner = Int64()
                elif isinstance(inner, Boolean):
                    result_inner = UInt32()
                elif isinstance(inner, Decimal):
                    result_inner = Decimal(38, inner.scale)
            elif method == "cum_prod" and isinstance(inner, self._CUM_PROD_INT64_RECEIVERS):
                result_inner = Int64()
            return receiver_name, _wrap_like(receiver_type, result_inner)

        # cum_count never raises: UInt32 for every receiver dtype (probed
        # incl. the dtypes the other cumulatives reject).
        if method == "cum_count":
            return receiver_name, UInt32()

        # ``pct_change`` divides — NOT dtype-preserving (issue #71; it left
        # the shift-like family below). Probed matrix on
        # ``_PCT_CHANGE_INVALID_RECEIVERS``: float receivers keep their
        # width, every other accepted receiver yields Float64, invalid
        # receivers flag PLY016. The head slot is always null and there is
        # no fill_value parameter, so the result is always Nullable.
        if method == "pct_change" and receiver_type is not None:
            inner = receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
            if isinstance(inner, Unknown):
                return receiver_name, Nullable(inner)
            if isinstance(inner, self._PCT_CHANGE_INVALID_RECEIVERS):
                self.errors.append(
                    tag(
                        PLY016,
                        f"pct_change: operation not supported for dtype {inner} — "
                        f"polars raises an error at runtime",
                    )
                )
                return receiver_name, None
            if isinstance(inner, (Float16, Float32)):
                return receiver_name, Nullable(inner)
            return receiver_name, Nullable(Float64())

        # Shift-like: head positions become NULL → wrap in Nullable. Only
        # ``shift`` takes a fill (``shift(n, *, fill_value=...)`` — keyword
        # only, probed): a non-null fill plugs the shifted-in slots, so the
        # receiver's own nullability is preserved instead (issue #43);
        # ``diff`` has no fill_value parameter.
        if method in self._SHIFT_LIKE_METHODS and receiver_type is not None:
            if method == "shift":
                fill_node = next((kw.value for kw in node.keywords if kw.arg == "fill_value"), None)
                if fill_node is not None:
                    fill_result = self._shift_fill_dtype(receiver_type, fill_node)
                    if fill_result is not None:
                        return receiver_name, fill_result
                    # A null fill (fill_value=None / pl.lit(None)) behaves
                    # like no fill (probed) — fall through to the wrap.
            inner = receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
            if method == "diff":
                # ``diff`` is a subtraction, not a dtype-preserving window
                # (issue #46). Probed (polars 1.41.2): a temporal receiver
                # (Date / Datetime — any tz — / Time / Duration) yields
                # Duration, keeping the receiver's time unit (issue #66;
                # Date is us-based and Time ns-based, both probed); an
                # unsigned-int receiver widens to the signed dtype of the
                # next width (UInt8 -> Int16, UInt16 -> Int32,
                # UInt32 -> Int64, UInt64 -> Int64; UInt128 has no wider
                # signed dtype and stays UInt128). Signed ints and floats
                # keep their dtype — the generic wrap below.
                if isinstance(inner, (Datetime, Duration)):
                    return receiver_name, Nullable(Duration(unit=inner.unit))
                if isinstance(inner, Date):
                    return receiver_name, Nullable(Duration(unit="us"))
                if isinstance(inner, Time):
                    return receiver_name, Nullable(Duration(unit="ns"))
                widened = _DIFF_UNSIGNED_WIDENING.get(type(inner))
                if widened is not None:
                    return receiver_name, Nullable(widened)
            return receiver_name, Nullable(inner)

        # Rolling reductions returning Float64 — except a Float32 receiver,
        # which keeps Float32 (probed 1.41.2; backlog N-2 — Float16 is NOT
        # width-preserved: it widens to Float64). Rows whose window holds
        # fewer than ``min_samples`` (default: window_size) values are null
        # (probed 1.41.2; issue #57), so the result is Nullable unless the
        # call provably fills every window (see _rolling_min_samples_total).
        # rolling_std/rolling_var additionally need an explicit ``ddof=0``:
        # the default ddof=1 is null on 1-sample windows. String / Null
        # receivers are accepted by polars but yield an ALL-null Float64
        # column (probed), so they stay Nullable regardless; a Nullable
        # receiver stays Nullable too (an all-null window is null even with
        # min_samples=1).
        if method in self._ROLLING_FLOAT_METHODS and receiver_type is not None:
            inner = receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
            rolling_float: DataType = Float32() if isinstance(inner, Float32) else Float64()
            total = _rolling_min_samples_total(method, node, self._const_int)
            if method in ("rolling_std", "rolling_var"):
                total = total and _rolling_ddof_zero(node, self._const_int)
            if (
                total
                and not isinstance(receiver_type, Nullable)
                and not isinstance(inner, (Utf8, Null))
            ):
                return receiver_name, rolling_float
            return receiver_name, Nullable(rolling_float)

        # Dtype-carrying rolling reducers (rolling_sum/min/max): strictly
        # typed receivers — the probed matrix on
        # ``_ROLLING_INVALID_RECEIVERS`` flags PLY016 and degrades the
        # output to Unknown (issue #57; the family was deferred in #49).
        # Valid cells follow the same windowed-nullability rule as the
        # Float64 family above; rolling_sum upcasts narrow ints to Int64
        # and Boolean to UInt32 (probed, mirrors cum_sum). Unknown
        # receivers stay silent.
        if method in self._ROLLING_INVALID_RECEIVERS and receiver_type is not None:
            inner = receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
            if isinstance(inner, Unknown):
                return receiver_name, receiver_type
            if isinstance(inner, self._ROLLING_INVALID_RECEIVERS[method]):
                self.errors.append(
                    tag(
                        PLY016,
                        f"{method}: operation not supported for dtype {inner} — "
                        f"polars raises InvalidOperationError at runtime",
                    )
                )
                return receiver_name, None
            result_inner: DataType = inner
            if method == "rolling_sum":
                if isinstance(inner, self._ROLLING_SUM_INT64_RECEIVERS):
                    result_inner = Int64()
                elif isinstance(inner, Boolean):
                    result_inner = UInt32()
            if isinstance(receiver_type, Nullable) or not _rolling_min_samples_total(
                method, node, self._const_int
            ):
                return receiver_name, Nullable(result_inner)
            return receiver_name, result_inner

        # ``rank(method=...)`` — the ranking method decides the dtype: the
        # default "average" returns Float64; the count-based methods return
        # polars' IDX dtype (UInt32). The receiver's Nullable wrapper is
        # preserved on the result.
        if method == "rank" and receiver_type is not None:
            rank_method = "average"
            if node.args:
                cand = _str_constant(node.args[0])
                if cand is not None:
                    rank_method = cand
            for kw in node.keywords:
                if kw.arg == "method":
                    cand = _str_constant(kw.value)
                    if cand is not None:
                        rank_method = cand
            if rank_method == "average":
                return receiver_name, _wrap_like(receiver_type, Float64())
            if rank_method in ("min", "max", "dense", "ordinal", "random"):
                return receiver_name, _wrap_like(receiver_type, UInt32())
            return None

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
            # ``Expr.len()`` counts nulls too; same UInt32 dtype (issue #23).
            "len": AggFunction.COUNT,
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
                result_type = infer_agg_result_type(
                    agg_map[method],
                    receiver_type,
                    # Expression-level aggs are whole-frame reductions unless
                    # the agg chain fallback re-entered us (backlog N-5).
                    context="agg" if self._in_agg_chain else "select",
                )
            except GroupByTypeError as e:
                self.errors.append(tag(PLY011, str(e)))
                return receiver_name, receiver_type
            # std/var with an explicit literal ``ddof=0`` are total on
            # non-empty input (probed 1.41.2: singleton group -> 0.0), so
            # the Nullable wrap from ``infer_agg_result_type`` is undone;
            # the receiver's own nullability still propagates (an all-null
            # window stays null even with ddof=0). The inferred width
            # (Float64, or Float32 for a Float32 receiver — backlog N-2)
            # is kept as-is.
            if (
                method in ("std", "var")
                and _stdvar_ddof_zero(node)
                and isinstance(result_type, Nullable)
                and isinstance(result_type.inner, (Float32, Float64))
            ):
                result_type = _wrap_like(receiver_type, result_type.inner)
            return receiver_name, result_type

        # ``cast(pl.<dtype>)`` chained directly on column. A structurally
        # impossible source -> target pair flags PLY013 (issue #34) and the
        # output degrades to Unknown — fabricating the target dtype would
        # hide declared-type mismatches downstream. ``strict=False`` exempts
        # nothing: the probed-invalid pairs fail in both modes.
        if method == "cast" and node.args:
            target = _resolve_pl_dtype(node.args[0])
            if target is not None and receiver_type is not None:
                receiver_inner = (
                    receiver_type.inner if isinstance(receiver_type, Nullable) else receiver_type
                )
                if _cast_invalid(receiver_inner, target):
                    self.errors.append(
                        tag(
                            PLY013,
                            f"cast: {receiver_inner} cannot be cast to {target} — "
                            f"polars raises InvalidOperationError even with "
                            f"strict=False",
                        )
                    )
                    return receiver_name, None
                return receiver_name, _wrap_like(receiver_type, target)
            if target is not None:
                # Receiver dtype was uninferable (e.g. ``.interpolate()``)
                # but the explicit cast pins the result dtype. The cast is
                # exactly the repair PLW007 asks for, so retract any PLW007
                # the receiver chain just emitted (backlog B-4).
                self.warnings[warnings_before_receiver:] = [
                    w
                    for w in self.warnings[warnings_before_receiver:]
                    if not w.startswith(f"[{PLW007}]")
                ]
                return receiver_name, target

        # Unrecognised method on a resolved receiver: it IS a chain on this
        # column, the dtype just can't be inferred. Surface the name so the
        # column stays registered (as Unknown) instead of vanishing from the
        # tracked schema (issue #8). When the receiver dtype was precisely
        # known, the degradation is worth a warning (backlog B-4) — an
        # already-degraded receiver stays silent so one unmodeled call does
        # not cascade into a warning per chained method.
        if receiver_type is not None and not _base_is_unknown(receiver_type):
            self.warnings.append(_unmodeled_method_warning(f".{method}()"))
        return receiver_name, None

    def _extract_lit_type(self, node: ast.expr) -> DataType | None:
        """Extract type from pl.lit(value) expression."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "lit":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "pl":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            value = node.args[0].value
                            if value is None or isinstance(value, (bool, int, float, str, bytes)):
                                return infer_lit(value)
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
        module_consts: dict[str, str | list[str] | int] | None = None,
        reported_broken_schemas: set[str] | None = None,
        reported_degraded_schemas: set[str] | None = None,
    ):
        self.input_types = input_types
        self.errors = errors
        # Schema names already flagged PLY041 for this function (issue #69).
        # Shared with ``analyze_function`` (signature sites) so a broken
        # schema is reported once per function however many annotation /
        # validate sites reference it.
        self._reported_broken_schemas: set[str] = (
            reported_broken_schemas if reported_broken_schemas is not None else set()
        )
        # Same dedup contract for PLW011 (issue #77): schemas with
        # unrecognized field annotations, warned once per function.
        self._reported_degraded_schemas: set[str] = (
            reported_degraded_schemas if reported_degraded_schemas is not None else set()
        )
        # Non-fatal advisories. Owned externally so nested analyses can append.
        self.warnings: list[str] = warnings if warnings is not None else []
        self.registry = registry or FunctionRegistry()
        self.schema_registry = schema_registry or SchemaRegistry()
        self.class_registry = class_registry or ClassRegistry()
        # Name of the class enclosing the function being analysed (None for
        # module-level functions). Used to resolve ``self.method()`` /
        # ``cls.method()`` to a class-local method's return annotation.
        self.current_class_name = current_class_name
        # Module-level ``NAME = "lit"`` / ``NAME = ["a", "b"]`` / ``N = 1``
        # constants, collected by ``analyze_source``. Read-only here;
        # function-local constants in ``var_consts`` shadow them.
        self.module_consts: dict[str, str | list[str] | int] = module_consts or {}
        # Track variable -> FrameType mapping
        self.var_types: dict[str, FrameType] = dict(input_types)
        # Track variable -> FrameList element type (for partition_by results
        # and any other op that yields a list of frames).
        self.var_lists: dict[str, FrameType] = {}
        # Track variable -> class name for ``var = ClassName()`` assignments,
        # so a later ``var.method()`` call can resolve to the class's method.
        self.var_classes: dict[str, str] = {}
        # Track function-local ``var = "lit"`` / ``var = ["a", "b"]`` /
        # ``var = 1`` constants so column-spec arguments passed by name
        # (``join(on=key)``, ``unpivot(on=cols)``, ...) and int-valued call
        # args (rolling ``min_samples``/``window_size``/``ddof``; backlog
        # B-5) can be resolved. Reassigning the name to anything
        # non-constant drops the entry.
        self.var_consts: dict[str, str | list[str] | int] = {}
        self.return_type: FrameType | None = None
        # Bare ``Schema.validate(df)`` narrowing only fires at the function's
        # top level. We toggle this off when descending into if/for/while/try.
        self._narrowing_enabled = True
        # Source positions (lineno, col_offset, method) that already got the
        # frame-level PLW007 — some call paths analyze the same node twice
        # (``.group_by(...).agg(...)`` infers the grouped receiver both for
        # its laziness and inside ``_infer_agg_call``), and the warning must
        # fire once per source call, not once per analysis.
        self._warned_frame_calls: set[tuple[int, int, str]] = set()

    def _note_schema_use(self, schema_name: str) -> None:
        """Flag PLY041 when a body site references a runtime-broken schema.

        Issue #69: ``x: DataFrame[Broken] = ...`` and ``Broken.validate(df)``
        crash at runtime even when the function signature is healthy. Once
        per schema per function (shared dedup set with the signature sites).

        Issue #77: the same body sites surface PLW011 when the schema has
        fields whose annotation degraded to Unknown dtype.
        """
        if schema_name not in self._reported_broken_schemas:
            error = _schema_definition_error(schema_name, self.schema_registry)
            if error is not None:
                self._reported_broken_schemas.add(schema_name)
                self.errors.append(error)
        if schema_name not in self._reported_degraded_schemas:
            warning = _schema_definition_warning(schema_name, self.schema_registry)
            if warning is not None:
                self._reported_degraded_schemas.add(schema_name)
                self.warnings.append(warning)

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
        if schema_ft is not None:
            self._note_schema_use(schema_node.id)
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
            # Any reassignment invalidates a previously recorded constant;
            # it is re-recorded below if the new RHS is again a literal.
            self.var_consts.pop(var_name, None)
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
                else:
                    # ``key = "id"`` / ``cols = ["a", "b"]`` / ``ms = 1`` —
                    # record the constant for column-spec / int-arg
                    # resolution.
                    const_val: str | list[str] | int | None = _str_constant(node.value)
                    if const_val is None:
                        const_val = _str_list_or_tuple(node.value)
                    if const_val is None:
                        const_val = _int_literal(node.value)
                    if const_val is not None:
                        self.var_consts[var_name] = const_val
                        self.var_types.pop(var_name, None)
                        self.var_lists.pop(var_name, None)
                        self.var_classes.pop(var_name, None)
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

    # -- constant resolution ----------------------------------------------

    def _lookup_const(self, name: str) -> str | list[str] | int | None:
        """Look up a constant binding: function-locals shadow module-level."""
        if name in self.var_consts:
            return self.var_consts[name]
        return self.module_consts.get(name)

    def _const_str(self, node: ast.expr) -> str | None:
        """Resolve ``node`` to a string: a literal, or a ``Name`` bound to a
        string constant (function-local, falling back to module-level)."""
        s = _str_constant(node)
        if s is not None:
            return s
        if isinstance(node, ast.Name):
            val = self._lookup_const(node.id)
            if isinstance(val, str):
                return val
        return None

    def _const_str_list(self, node: ast.expr) -> list[str] | None:
        """Resolve ``node`` to a list of strings: a list/tuple literal, or a
        ``Name`` bound to a string-list constant (locals shadow module)."""
        lst = _str_list_or_tuple(node)
        if lst is not None:
            return lst
        if isinstance(node, ast.Name):
            val = self._lookup_const(node.id)
            if isinstance(val, list):
                return val
        return None

    def _int_consts(self) -> dict[str, int]:
        """Snapshot of the int-constant bindings visible at this statement
        (function locals shadow module-level; backlog B-5). Handed to
        ``ExpressionAnalyzer`` so int-valued call args resolve like
        literals. ``type(...) is int`` keeps bools out, mirroring
        ``_int_literal``."""
        merged: dict[str, str | list[str] | int] = {**self.module_consts, **self.var_consts}
        return {name: val for name, val in merged.items() if type(val) is int}

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
        # Non-strict return schemas bind open at call sites (issue #81).
        return _call_result_frame(method_info.signature.return_type, method_name)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments like: df: DataFrame[Schema] = expr."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # Any reassignment invalidates a previously recorded constant;
            # it is re-recorded below if the new RHS is again a literal.
            self.var_consts.pop(var_name, None)
            # Try to get type from a Pandera DataFrame[Schema] annotation
            frame_type, _ = _resolve_declared_type(node.annotation, self.schema_registry)
            if frame_type is not None:
                annotated_schema = frame_annotation_schema_name(node.annotation)
                if annotated_schema is not None:
                    self._note_schema_use(annotated_schema)
                self.var_types[var_name] = frame_type
                # Walk the RHS so expression-level warnings (e.g. PLW005
                # from pivot) and column-resolution errors surface, and
                # check the annotation against the inferred frame
                # (ADR-0005). The annotation wins for the variable's
                # static type either way — one clear diagnostic at the
                # assignment, no downstream cascade.
                if node.value:
                    inferred = self._infer_expr_type(node.value)
                    if inferred is not None:
                        self._check_annotation_against_inferred(
                            node.annotation, frame_type, inferred
                        )
                return
            # Fall back to inference from value
            if node.value:
                inferred = self._infer_expr_type(node.value)
                if inferred:
                    self.var_types[var_name] = inferred
                else:
                    # ``key: str = "id"`` / ``ms: int = 1`` — record
                    # constants the same way un-annotated assignments do.
                    const_val: str | list[str] | int | None = _str_constant(node.value)
                    if const_val is None:
                        const_val = _str_list_or_tuple(node.value)
                    if const_val is None:
                        const_val = _int_literal(node.value)
                    if const_val is not None:
                        self.var_consts[var_name] = const_val
        self.generic_visit(node)

    def _check_annotation_against_inferred(
        self,
        annotation: ast.expr,
        declared: FrameType,
        inferred: FrameType,
    ) -> None:
        """ADR-0005 two-direction rule: check a variable annotation against
        the inferred RHS schema.

        The forward comparison reuses the checker's verdict engine, so every
        leniency rule (Unknown compatibility, open-frame skips, ``coerce``)
        applies — a pivot/Unknown RHS never contradicts its annotation.
        Forward failures are then classified by the REVERSE direction:

        - ``declared <: inferred`` holds — a pure *narrowing assertion*
          (non-null over nullable, required over optional): allowed,
          surfaced as PLW008 with the runtime-backed upgrade.
        - neither direction holds (unrelated dtype, provably-absent
          column, strict extras, eager/lazy mismatch): PLY033 error — the
          annotation re-interprets the frame as something it provably is
          not.
        """
        # Function-level import: checker imports analyzer at module level
        # (check_source -> analyze_source), so the reverse import must be
        # deferred to call time to avoid the cycle.
        from polypolarism.checker import _is_coercible_difference, _subtype_verdict

        narrowings: list[str] = []
        contradictions: list[str] = []
        if declared.is_lazy != inferred.is_lazy:
            declared_kind = "LazyFrame" if declared.is_lazy else "DataFrame"
            inferred_kind = "LazyFrame" if inferred.is_lazy else "DataFrame"
            contradictions.append(
                f"annotation declares {declared_kind} but the expression is {inferred_kind}"
            )
        for col_name, declared_spec in declared.columns.items():
            inferred_spec = inferred.columns.get(col_name)
            if inferred_spec is None:
                if declared_spec.required and inferred.rest is None:
                    if inferred.strict:
                        # Only a STRICT inferred frame proves absence; a
                        # non-strict schema tolerates extra runtime columns,
                        # so declaring one is a narrowing assertion
                        # (issue #63).
                        contradictions.append(
                            f"column '{col_name}' ({declared_spec.dtype}) is provably "
                            f"absent from the inferred frame"
                        )
                    else:
                        narrowings.append(
                            f"column '{col_name}' ({declared_spec.dtype}) is declared "
                            f"but not guaranteed by the (non-strict) inferred frame"
                        )
                continue
            if declared_spec.required and not inferred_spec.required:
                # Asserting an optional column always-present is the
                # column-existence flavor of narrowing.
                narrowings.append(
                    f"column '{col_name}': declared always-present but inferred optional"
                )
                continue
            verdict = _subtype_verdict(inferred_spec.dtype, declared_spec.dtype)
            if verdict.ok:
                continue
            if declared.coerce and _is_coercible_difference(
                inferred_spec.dtype, declared_spec.dtype
            ):
                # Sound at return positions (check_types really coerces at
                # runtime), but annotations are runtime-inert — relying on
                # coerce here is an unbacked re-type (issue #64). The
                # validate remedy in the warning text WOULD coerce.
                narrowings.append(
                    f"column '{col_name}': declared {declared_spec.dtype} but "
                    f"inferred {inferred_spec.dtype} relies on coerce=True, "
                    f"which annotations do not run"
                )
                continue
            detail = (
                f"column '{col_name}': declared {declared_spec.dtype} but "
                f"inferred {inferred_spec.dtype}"
            )
            if _subtype_verdict(declared_spec.dtype, inferred_spec.dtype).ok:
                narrowings.append(detail)
            else:
                contradictions.append(detail)
        if declared.strict:
            for col_name, inferred_spec in inferred.columns.items():
                if col_name not in declared.columns:
                    contradictions.append(
                        f"extra column '{col_name}' ({inferred_spec.dtype}) "
                        f"contradicts the strict annotation"
                    )
        if contradictions:
            self.errors.append(
                tag(
                    PLY033,
                    f"annotation `{ast.unparse(annotation)}` re-interprets the "
                    "inferred frame as an unrelated type: "
                    + "; ".join(contradictions)
                    + ". Fix the annotation or the expression.",
                )
            )
        if narrowings:
            self.warnings.append(
                tag(
                    PLW008,
                    f"annotation `{ast.unparse(annotation)}` narrows the "
                    "inferred schema without runtime backing: "
                    + "; ".join(narrowings)
                    + ". Assert it with `Schema.validate(...)` (which retypes "
                    "the value), or widen the annotation.",
                )
            )

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
            single = self._const_str(arg)
            multi = self._const_str_list(arg)
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

            # ``pl.DataFrame({...})`` / ``pl.LazyFrame({...})`` literal
            # constructors — the schema is read off the dict literal /
            # explicit ``schema=`` (issue #25).
            if (
                isinstance(receiver, ast.Name)
                and receiver.id == "pl"
                and method_name in ("DataFrame", "LazyFrame")
            ):
                return self._infer_frame_literal(node, lazy=method_name == "LazyFrame")

            # ``pl.read_parquet(...)`` / ``pl.scan_csv(...)`` — IO readers
            # whose schema only the data source knows (ADR-0006). The
            # result is an empty OPEN frame: column references resolve to
            # Unknown, shape-determining calls downstream close it, and
            # the eager/lazy bit is enforced.
            if isinstance(receiver, ast.Name) and receiver.id == "pl":
                if method_name in EAGER_READ_FUNCTIONS:
                    return FrameType({}, rest=RowVar(method_name), is_lazy=False)
                if method_name in LAZY_SCAN_FUNCTIONS:
                    return FrameType({}, rest=RowVar(method_name), is_lazy=True)

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
                elif method_name == "sort":
                    # Identity-shaped like the _IDENTITY_FRAME_METHODS
                    # fallback below, but the sort keys are validated
                    # (issue #29). Must be dispatched before that fallback —
                    # ``sort`` stays in the compat identity set.
                    return _lazy_like(self._infer_sort_call(receiver_type, node), receiver_type)
                elif method_name == "unique":
                    # Identity-shaped, but the ``subset=`` columns are
                    # validated (issue #35). Like ``sort``, dispatched before
                    # the identity fallback — ``unique`` stays in the compat
                    # identity set.
                    return _lazy_like(self._infer_unique_call(receiver_type, node), receiver_type)
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
                elif method_name == "to_dummies":
                    return self._infer_to_dummies_call(receiver_type, node)
                elif method_name == "null_count":
                    return self._infer_null_count_call(receiver_type)
                elif method_name == "upsample":
                    return self._infer_upsample_call(receiver_type, node)
                elif method_name == "join_where":
                    return self._infer_join_where_call(receiver_type, node)
                elif method_name in _IDENTITY_FRAME_METHODS:
                    return receiver_type
                elif method_name in (
                    _LAZY_FRAME_RETURNING_METHODS
                    if receiver_type.is_lazy
                    else _EAGER_FRAME_RETURNING_METHODS
                ):
                    # Unmodeled frame-returning method on a tracked receiver:
                    # the variable silently untracks and every downstream
                    # check dies quietly — warn (backlog N-3). The probed
                    # frame-returning gate keeps terminal methods
                    # (``to_dicts``, ``write_*``, ``height``, ...) and
                    # unknown names (typos, plugin namespaces) silent; the
                    # receiver's laziness picks the probe set so a
                    # wrong-side call (eager-only method on a LazyFrame)
                    # keeps its precise PLY030/PLY031 without a warning
                    # piled on top. Deduped per source call: ``.agg()``
                    # chains analyze the grouped receiver twice (laziness
                    # probe + _infer_agg_call).
                    key = (node.lineno, node.col_offset, method_name)
                    if key not in self._warned_frame_calls:
                        self._warned_frame_calls.add(key)
                        self.warnings.append(
                            _unmodeled_method_warning(f".{method_name}()", frame=True)
                        )
                    return None

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
        if ft is not None:
            self._note_schema_use(schema_node.id)
        if ft is None or not node.args:
            return ft
        # The validation retypes the result to the schema — exactly the
        # repair PLW007 recommends — so a PLW007 emitted while analyzing the
        # wrapped argument (``Schema.validate(df.interpolate())``) is
        # retracted: the frame-level analog of the expression-level cast
        # retraction (backlog N-3). A warning fired on an EARLIER statement
        # stands — between that call and this validate the variable really
        # was untracked.
        warnings_before_arg = len(self.warnings)
        arg_type = self._infer_expr_type(node.args[0])
        self.warnings[warnings_before_arg:] = [
            w for w in self.warnings[warnings_before_arg:] if not w.startswith(f"[{PLW007}]")
        ]
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
                if ft is not None:
                    self._note_schema_use(callable_arg.value.id)
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
                    # Deferred import — see _is_frame_subtype.
                    from polypolarism.checker import _is_coercible_difference

                    # Generate detailed error
                    for col_name, expected_col_spec in expected_type.columns.items():
                        if col_name not in arg_type.columns:
                            self.errors.append(
                                f"Argument '{param_name}' is missing column '{col_name}'"
                            )
                        else:
                            actual_col_dtype = arg_type.columns[col_name].dtype
                            expected_col_dtype = expected_col_spec.dtype
                            if _is_column_subtype(actual_col_dtype, expected_col_dtype):
                                continue
                            # Mirror _is_frame_subtype: coercible dtype
                            # differences are not errors under coerce.
                            if expected_type.coerce and _is_coercible_difference(
                                actual_col_dtype, expected_col_dtype
                            ):
                                continue
                            self.errors.append(
                                f"Argument '{param_name}' column '{col_name}' has type "
                                f"{actual_col_dtype} but expected {expected_col_dtype}"
                            )
                    # Strict parameter schemas reject undeclared columns at
                    # runtime (issue #82) — a REQUIRED pinned extra is
                    # provable even on an open argument frame. Optional
                    # (required=False) columns MAY be absent (issue #84) and
                    # unknown open-frame extras are unenumerable — both stay
                    # lenient (not provable).
                    if expected_type.strict:
                        for col_name, actual_spec in arg_type.columns.items():
                            if col_name not in expected_type.columns and actual_spec.required:
                                self.errors.append(
                                    f"Argument '{param_name}' has extra column "
                                    f"'{col_name}' ({actual_spec.dtype}) but the "
                                    f"parameter schema is strict"
                                )
            return _call_result_frame(sig.return_type, func_name)

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
            module_consts=self.module_consts,
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

        # Extract keyword arguments. Key columns may be a string literal,
        # a list/tuple of strings, or a name bound to such a constant.
        on: str | list[str] | None = None
        left_on: str | list[str] | None = None
        right_on: str | list[str] | None = None
        how: str = "inner"
        suffix: str = "_right"
        coalesce: bool | None = None

        for kw in node.keywords:
            if kw.arg in ("on", "left_on", "right_on"):
                keys: str | list[str] | None = self._const_str(kw.value)
                if keys is None:
                    keys = self._const_str_list(kw.value)
                if keys is None:
                    continue
                if kw.arg == "on":
                    on = keys
                elif kw.arg == "left_on":
                    left_on = keys
                else:
                    right_on = keys
            elif kw.arg == "how":
                cand = self._const_str(kw.value)
                if cand is not None:
                    how = cand
            elif kw.arg == "suffix":
                cand = self._const_str(kw.value)
                if cand is not None:
                    suffix = cand
            elif kw.arg == "coalesce":
                # Only a literal True/False is honored; anything dynamic
                # falls back to the how-specific polars default (None).
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                    coalesce = kw.value.value

        if how in ("inner", "left", "right", "full", "cross", "semi", "anti"):
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
                suffix=suffix,
                coalesce=coalesce,
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
        suffix: str = "_right"
        for kw in node.keywords:
            cand = self._const_str(kw.value)
            if cand is None:
                continue
            if kw.arg == "on":
                on = cand
            elif kw.arg == "left_on":
                left_on = cand
            elif kw.arg == "right_on":
                right_on = cand
            elif kw.arg == "suffix":
                suffix = cand
        try:
            # join_asof coalesces the key only when ``on`` names both sides:
            # differently-named left_on/right_on keys are both kept.
            return infer_join(
                left_type,
                right_type,
                on=on,
                left_on=left_on,
                right_on=right_on,
                how="left",
                suffix=suffix,
                coalesce=on is not None,
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
                # Names bound to such constants resolve too.
                single = self._const_str(arg)
                if single is not None:
                    keys.append(single)
                    continue
                multi = self._const_str_list(arg)
                if multi is not None:
                    keys.extend(multi)
        else:
            # group_by_dynamic / rolling: first positional / ``index_column`` is the time axis.
            index_col: str | None = None
            if groupby_receiver.args:
                index_col = self._const_str(groupby_receiver.args[0])
            for kw in groupby_receiver.keywords:
                if kw.arg == "index_column":
                    cand = self._const_str(kw.value)
                    if cand is not None:
                        index_col = cand
                if kw.arg == "by" or kw.arg == "group_by":
                    extra = self._const_str_list(kw.value)
                    single = self._const_str(kw.value)
                    if extra is not None:
                        keys.extend(extra)
                    elif single is not None:
                        keys.append(single)
            if index_col is not None:
                keys.insert(0, index_col)

        # Extract aggregation expressions
        expr_analyzer = ExpressionAnalyzer(
            input_frame,
            warnings=self.warnings,
            registry=self.registry,
            int_consts=self._int_consts(),
        )
        agg_exprs: list[AggExpr] = []
        for arg in node.args:
            # A plural ``pl.col`` inside an aggregation expression produces
            # one output per column (issue #42): ``agg(pl.col("a", "b").sum())``
            # yields columns ``a`` and ``b``. Analyze one clone per name.
            for expr in self._expand_plural_expr(arg) or [arg]:
                agg_expr = expr_analyzer.analyze_agg_expr(expr)
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
            else:
                # Un-inferable aggregation expression — the output column
                # still exists at runtime, so register it as Unknown
                # (issue #8). Expression-level column errors are collected
                # separately via the expression analyser.
                agg_exprs.append(
                    AggExpr(column=kw.arg, function=None, alias=kw.arg, dtype=Unknown())
                )

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

    def _expand_plural_expr(self, arg: ast.expr) -> list[ast.expr] | None:
        """Expand an expression containing exactly one nested plural ``pl.col``.

        polars runs ``select(pl.col("a", "b") * 10)`` once per column
        (issue #42) — model that by returning one deep copy of ``arg`` per
        name, with the plural node swapped for single-name ``pl.col(name)``.
        Each clone then flows through the unchanged single-expression path,
        so output names follow each column (polars keeps per-column names
        through elementwise ops; probed on 1.41.2).

        Returns ``None`` when the tree contains zero plural nodes (regular
        path) or two-plus (polars' pairwise semantics there are out of
        scope — stay silent). The bare-plural fast path in the callers
        consumes ``pl.col("a", "b")`` arguments before this runs, so the
        plural node is usually nested; a bare plural reaching here (the
        ``agg`` loop has no fast path) expands to bare ``pl.col(name)``.
        """
        walked = list(ast.walk(arg))
        found: list[tuple[ast.AST, list[str]]] = []
        for sub in walked:
            if isinstance(sub, ast.expr):
                names = self._resolve_plural_col(sub)
                if names is not None:
                    found.append((sub, names))
        if len(found) != 1:
            return None
        target, names = found[0]
        return self._clone_per_column(arg, target, names)

    def _expand_selector_chain(
        self, arg: ast.expr, input_frame: FrameType
    ) -> list[ast.expr] | None:
        """Expand a method chain ROOTED at a multi-column selector (issue #56).

        polars runs ``select(pl.all().name.prefix("p_"))`` /
        ``select(cs.numeric().sum())`` once per selected column — model
        that by cloning the chain per column with the selector swapped for
        single-name ``pl.col(name)``, exactly like the plural-``pl.col``
        expansion. Only the receiver spine is walked: a selector in
        ARGUMENT position (``.over(cs.by_name("g"))``, ``pl.struct(cs.*)``)
        is a column-set argument, not a per-column fan-out, and must NOT
        expand the surrounding expression. Bare selector args never reach
        this helper (the callers' fast paths consume them first).
        """
        node: ast.expr = arg
        while True:
            nxt: ast.expr | None = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                nxt = node.func.value
            elif isinstance(node, ast.Attribute):
                nxt = node.value
            if nxt is None:
                return None
            names = _resolve_selector(nxt, input_frame)
            if names is not None:
                return self._clone_per_column(arg, nxt, names)
            node = nxt

    @staticmethod
    def _clone_per_column(arg: ast.expr, target: ast.AST, names: list[str]) -> list[ast.expr]:
        """One deep copy of ``arg`` per column name, with ``target`` (a node
        of ``arg``, matched by identity) swapped for ``pl.col(name)``."""
        walked = list(ast.walk(arg))
        target_idx = next(i for i, sub in enumerate(walked) if sub is target)
        clones: list[ast.expr] = []
        for name in names:
            clone = copy.deepcopy(arg)
            # ``ast.walk`` order is structural, so the clone's walk list
            # lines up index-for-index with the original's.
            clone_target = list(ast.walk(clone))[target_idx]
            replacement = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pl", ctx=ast.Load()), attr="col", ctx=ast.Load()
                ),
                args=[ast.Constant(value=name)],
                keywords=[],
            )
            ast.copy_location(replacement, clone_target)
            ast.fix_missing_locations(replacement)
            expanded = _ReplaceNode(clone_target, replacement).visit(clone)
            assert isinstance(expanded, ast.expr)
            clones.append(expanded)
        return clones

    def _register_string_selection(
        self,
        name: str,
        input_frame: FrameType,
        result_columns: dict[str, DataType],
        output_name: str | None = None,
    ) -> str | None:
        """Resolve a bare string column name in ``select`` — equivalent to
        ``pl.col(name)``. Missing names error on closed frames (PLY001);
        on an open frame the column may exist among the unknown extras,
        so it is selected as ``Unknown``.

        ``output_name`` overrides the result column name for the kwarg
        form: ``select(x="a")`` selects column ``a`` under the name ``x``.

        Returns the output column name actually registered, or ``None``
        when the name was missing on a closed frame (PLY001 emitted)."""
        spec = input_frame.columns.get(name)
        if spec is not None:
            result_columns[output_name or name] = spec.dtype
            return output_name or name
        if input_frame.rest is not None:
            if name in input_frame.absent:
                # Negative knowledge (issue #78): removed by an earlier
                # drop/rename — a guaranteed runtime miss.
                self.errors.append(
                    tag(
                        PLY001,
                        f"Column '{name}' not found — it was removed earlier "
                        f"in this chain (drop/rename)",
                    )
                )
                return None
            result_columns[output_name or name] = Unknown()
            return output_name or name
        code, msg = _missing_column_diag(input_frame, name)
        self.errors.append(tag(code, msg))
        return None

    def _track_output_name(self, name: str, seen: set[str], call_name: str) -> None:
        """Record one output column produced by a ``select`` / ``with_columns``
        call; a repeated name WITHIN the same call is a guaranteed runtime
        duplicate-name error (issue #36). ``with_columns`` overwriting a
        pre-existing input column is legal, so callers seed ``seen`` with
        only the names this call produced — never the input columns.
        """
        if name in seen:
            self.errors.append(
                tag(
                    PLY015,
                    f"duplicate output column '{name}' in {call_name} — "
                    f"polars rejects duplicate output names at runtime; "
                    f"rename one with `.alias(...)`",
                )
            )
        else:
            seen.add(name)

    def _infer_select_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """Infer type of .select() call."""
        expr_analyzer = ExpressionAnalyzer(
            input_frame,
            warnings=self.warnings,
            registry=self.registry,
            int_consts=self._int_consts(),
        )
        result_columns: dict[str, DataType] = {}
        # Output names produced by THIS call — a repeat is a runtime
        # DuplicateError (issue #36). Registration keeps the last dtype.
        seen_outputs: set[str] = set()
        # A ``.name.*`` output whose result name is unknowable (issue #56)
        # exists at runtime under SOME name — opens the result frame.
        has_opaque_outputs = False

        for arg in node.args:
            sel = _resolve_selector(arg, input_frame)
            if sel is not None:
                # A selector on an OPEN frame can also match unknown extra
                # columns we cannot enumerate (ADR-0006) — the pinned
                # matches register normally but the result stays open.
                if input_frame.rest is not None:
                    has_opaque_outputs = True
                for c in sel:
                    spec = input_frame.columns.get(c)
                    if spec is not None:
                        result_columns[c] = spec.dtype
                        self._track_output_name(c, seen_outputs, "select")
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
                            self._track_output_name(c, seen_outputs, "select")
                            continue
                        code, msg = _missing_column_diag(input_frame, c)
                        self.errors.append(tag(code, msg))
                        continue
                    result_columns[c] = spec.dtype
                    self._track_output_name(c, seen_outputs, "select")
                continue
            # Bare string / list-of-strings column names — the most common
            # polars idiom: ``select("a", "b")`` / ``select(["a", "b"])``.
            # Constant-bound names (``KEY = "a"; select(KEY)``) resolve the
            # same way (issue #22); an unknown bare ``ast.Name`` is NOT a
            # constant and falls through to expression analysis (it could
            # be a frame variable).
            single = self._const_str(arg)
            if single is not None:
                registered = self._register_string_selection(single, input_frame, result_columns)
                if registered is not None:
                    self._track_output_name(registered, seen_outputs, "select")
                continue
            str_list = self._const_str_list(arg)
            if str_list is not None:
                for c in str_list:
                    registered = self._register_string_selection(c, input_frame, result_columns)
                    if registered is not None:
                        self._track_output_name(registered, seen_outputs, "select")
                continue
            # A plural ``pl.col`` nested inside an expression (issue #42) or
            # a method chain rooted at a selector (issue #56) runs the
            # expression once per column — analyze one clone per name
            # through the unchanged single-expression path.
            expanded = self._expand_plural_expr(arg)
            if expanded is None:
                expanded = self._expand_selector_chain(arg, input_frame)
            for expr in expanded if expanded is not None else [arg]:
                name, dtype = expr_analyzer.analyze_select_expr(expr)
                if name and dtype:
                    result_columns[name] = dtype
                    self._track_output_name(name, seen_outputs, "select")
                elif name:
                    # Named output whose dtype is uninferable — register it as
                    # Unknown so later references resolve (issue #8).
                    result_columns[name] = Unknown()
                    self._track_output_name(name, seen_outputs, "select")
                elif dtype is not None and _contains_name_accessor(expr):
                    # A ``.name.*`` output whose name is unknowable (issue
                    # #56): the column exists at runtime under some name —
                    # open the frame instead of losing it.
                    has_opaque_outputs = True

        # Kwarg form ``select(name=expr)`` — polars treats it as
        # ``expr.alias("name")``. Same for ``with_columns``.
        for kw in node.keywords:
            if kw.arg is None:
                continue
            # ``select(x="a")`` — a bare string in expression position is a
            # column reference (not a Utf8 literal), renamed to the kwarg.
            # Constant-bound names (``select(x=KEY)``) resolve too (#22).
            col_ref = self._const_str(kw.value)
            if col_ref is not None:
                registered = self._register_string_selection(
                    col_ref, input_frame, result_columns, output_name=kw.arg
                )
                if registered is not None:
                    self._track_output_name(registered, seen_outputs, "select")
                continue
            _, dtype = expr_analyzer.analyze_select_expr(kw.value)
            if dtype is not None:
                result_columns[kw.arg] = dtype
            else:
                result_columns[kw.arg] = Unknown()
            self._track_output_name(kw.arg, seen_outputs, "select")

        self.errors.extend(expr_analyzer.errors)

        if result_columns or has_opaque_outputs:
            return FrameType(
                columns=result_columns,
                rest=RowVar("name") if has_opaque_outputs else None,
            )
        return None

    def _infer_with_columns_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """Infer type of .with_columns() call."""
        # Start with all existing columns
        result_columns: dict[str, ColumnSpec | DataType] = dict(input_frame.columns)

        expr_analyzer = ExpressionAnalyzer(
            input_frame,
            warnings=self.warnings,
            registry=self.registry,
            int_consts=self._int_consts(),
        )
        # Output names produced by THIS call — a repeat within the call is a
        # runtime duplicate-name error (issue #36). Collision with a
        # pre-existing input column is NOT an error (overwrite semantics),
        # so the set starts empty rather than seeded with the input columns.
        seen_outputs: set[str] = set()
        # A ``.name.*`` output whose result name is unknowable (issue #56)
        # exists at runtime under SOME name — opens the result frame.
        has_opaque_outputs = False

        for arg in node.args:
            sel = _resolve_selector(arg, input_frame)
            if sel is not None:
                # cs.* selectors in with_columns are a no-op type-wise (re-include
                # existing columns) — but each re-included name is still an
                # output of this call for duplicate detection.
                for c in sel:
                    self._track_output_name(c, seen_outputs, "with_columns")
                continue
            plural = self._resolve_plural_col(arg)
            if plural is not None:
                # ``with_columns(pl.col("a", "b"))`` is equivalent to keeping
                # the existing columns; nothing to add. On an open frame a
                # missing name may exist among the unknown extras — no error.
                for c in plural:
                    if c not in input_frame.columns and input_frame.rest is None:
                        code, msg = _missing_column_diag(input_frame, c)
                        self.errors.append(tag(code, msg))
                        continue
                    self._track_output_name(c, seen_outputs, "with_columns")
                continue
            # Bare string / list-of-strings — equivalent to ``pl.col(name)``,
            # a re-selection of existing columns. Validate existence (PLY001
            # on closed frames) but leave the schema unchanged. Constant-bound
            # names resolve the same way (issue #22); an unknown bare
            # ``ast.Name`` falls through to expression analysis.
            single = self._const_str(arg)
            str_list = self._const_str_list(arg)
            if single is not None or str_list is not None:
                names = [single] if single is not None else str_list
                assert names is not None
                for c in names:
                    if c not in input_frame.columns and input_frame.rest is None:
                        code, msg = _missing_column_diag(input_frame, c)
                        self.errors.append(tag(code, msg))
                        continue
                    self._track_output_name(c, seen_outputs, "with_columns")
                continue
            # A plural ``pl.col`` nested inside an expression (issue #42) or
            # a method chain rooted at a selector (issue #56) runs the
            # expression once per column — analyze one clone per name
            # through the unchanged single-expression path.
            expanded = self._expand_plural_expr(arg)
            if expanded is None:
                expanded = self._expand_selector_chain(arg, input_frame)
            for expr in expanded if expanded is not None else [arg]:
                name, dtype = expr_analyzer.analyze_select_expr(expr)
                if name and dtype:
                    result_columns[name] = dtype
                    self._track_output_name(name, seen_outputs, "with_columns")
                elif name:
                    # Named output whose dtype is uninferable — register it as
                    # Unknown so later references resolve (issue #8).
                    result_columns[name] = Unknown()
                    self._track_output_name(name, seen_outputs, "with_columns")
                elif dtype is not None and _contains_name_accessor(expr):
                    # A ``.name.*`` output whose name is unknowable (issue
                    # #56): the column exists at runtime under some name —
                    # open the frame instead of losing it.
                    has_opaque_outputs = True

        # Kwarg form ``with_columns(name=expr)`` — polars treats it as
        # ``expr.alias("name")``.
        for kw in node.keywords:
            if kw.arg is None:
                continue
            # ``with_columns(x="a")`` — a bare string in expression position
            # is a column reference (not a Utf8 literal), renamed to the kwarg.
            # Constant-bound names (``with_columns(x=KEY)``) resolve too (#22).
            col_ref = self._const_str(kw.value)
            if col_ref is not None:
                ref_spec = input_frame.columns.get(col_ref)
                if ref_spec is not None:
                    result_columns[kw.arg] = ref_spec.dtype
                elif input_frame.rest is not None:
                    result_columns[kw.arg] = Unknown()
                else:
                    code, msg = _missing_column_diag(input_frame, col_ref)
                    self.errors.append(tag(code, msg))
                    continue
                self._track_output_name(kw.arg, seen_outputs, "with_columns")
                continue
            _, dtype = expr_analyzer.analyze_select_expr(kw.value)
            if dtype is not None:
                result_columns[kw.arg] = dtype
            else:
                result_columns[kw.arg] = Unknown()
            self._track_output_name(kw.arg, seen_outputs, "with_columns")

        self.errors.extend(expr_analyzer.errors)

        return FrameType(
            columns=result_columns,
            strict=input_frame.strict,
            rest=input_frame.rest or (RowVar("name") if has_opaque_outputs else None),
            # The constructor subtracts pinned names, so columns this call
            # (re)introduced clear their absence marks (issue #78).
            absent=input_frame.absent,
            nonstrict_schema=input_frame.nonstrict_schema,
        )

    # -- frame methods --------------------------------------------------

    def _collect_drop_targets(
        self, node: ast.Call, input_frame: FrameType | None = None
    ) -> list[str]:
        """Resolve column-name args for ``drop`` / ``drop_nulls(subset=...)``.

        Supports ``drop("a", "b")``, ``drop(["a", "b"])``, ``drop(cs.numeric())``
        when ``input_frame`` is supplied (selectors need a frame to resolve),
        and names bound to string(-list) constants.
        Returns column names in argument order (lists / selectors flattened in).
        """
        names: list[str] = []
        for arg in node.args:
            s = self._const_str(arg)
            if s is not None:
                names.append(s)
                continue
            lst = self._const_str_list(arg)
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
                if input_frame.rest is None:
                    self.errors.append(tag(PLY002, f"drop: column '{name}' not found"))
                elif name in input_frame.absent:
                    # Negative knowledge (issue #78): the column was
                    # already removed — polars drop (strict by default)
                    # raises ColumnNotFoundError on every execution.
                    self.errors.append(
                        tag(
                            PLY002,
                            f"drop: column '{name}' not found — it was removed "
                            f"earlier in this chain (drop/rename)",
                        )
                    )
                # Otherwise: the column may exist among the open frame's
                # unknown extras — dropping it is a no-op for the tracked
                # schema (but it is provably gone afterwards, see below).
                continue
            del result_columns[name]
        # Issue #78: every enumerable drop target is provably absent
        # afterwards — even targets that were never pinned (they may have
        # existed among the extras; either way the name is gone).
        absent = input_frame.absent | set(targets) if input_frame.rest is not None else None
        return FrameType(
            columns=result_columns,
            strict=input_frame.strict,
            rest=input_frame.rest,
            absent=absent,
            nonstrict_schema=input_frame.nonstrict_schema,
        )

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
        for old, new in mapping.items():
            if old not in input_frame.columns:
                if input_frame.rest is not None:
                    if old in input_frame.absent:
                        # Negative knowledge (issue #78): renaming a
                        # provably removed column always raises.
                        self.errors.append(
                            tag(
                                PLY003,
                                f"rename: column '{old}' not found — it was "
                                f"removed earlier in this chain (drop/rename)",
                            )
                        )
                        continue
                    # The source may exist among the open frame's unknown
                    # extras — assume the rename succeeded (ADR-0006) and
                    # pin the target name.
                    result_columns.setdefault(new, ColumnSpec(dtype=Unknown()))
                    continue
                self.errors.append(tag(PLY003, f"rename: column '{old}' not found"))
        # Issue #78: a renamed-away old name is provably absent afterwards
        # — unless some other entry renames INTO it (a swap). Rename
        # targets are provably present, clearing any stale marks.
        absent = None
        if input_frame.rest is not None:
            gone = set(mapping.keys()) - set(mapping.values())
            absent = (input_frame.absent | gone) - set(mapping.values())
        return FrameType(
            columns=result_columns,
            strict=input_frame.strict,
            rest=input_frame.rest,
            absent=absent,
            nonstrict_schema=input_frame.nonstrict_schema,
        )

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
                if input_frame.rest is not None:
                    if col in input_frame.absent:
                        # Negative knowledge (issue #78): casting a
                        # provably removed column always raises.
                        self.errors.append(
                            tag(
                                PLY004,
                                f"cast: column '{col}' not found — it was removed "
                                f"earlier in this chain (drop/rename)",
                            )
                        )
                        continue
                    # The column may exist among the open frame's unknown
                    # extras — assume the cast succeeded (ADR-0006); its
                    # dtype is now exactly the target.
                    result_columns[col] = ColumnSpec(dtype=target)
                    continue
                self.errors.append(tag(PLY004, f"cast: column '{col}' not found"))
                continue
            source_inner = spec.dtype.inner if isinstance(spec.dtype, Nullable) else spec.dtype
            if _cast_invalid(source_inner, target):
                # Issue #34: structurally impossible cast — flag and keep
                # the source spec rather than fabricating the target dtype.
                # ``strict=False`` exempts nothing (probed).
                self.errors.append(
                    tag(
                        PLY013,
                        f"cast: column '{col}' with dtype {source_inner} cannot be "
                        f"cast to {target} — polars raises InvalidOperationError "
                        f"even with strict=False",
                    )
                )
                continue
            result_columns[col] = ColumnSpec(
                dtype=_wrap_like(spec.dtype, target),
                required=spec.required,
            )
        return FrameType(
            columns=result_columns,
            strict=input_frame.strict,
            rest=input_frame.rest,
            absent=input_frame.absent,
            nonstrict_schema=input_frame.nonstrict_schema,
        )

    def _infer_drop_nulls_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        # subset can be passed positionally or as keyword
        subset: list[str] | None = None
        if node.args:
            single = self._const_str(node.args[0])
            cand = self._const_str_list(node.args[0]) or (None if single is None else [single])
            if cand is not None:
                subset = cand
        for kw in node.keywords:
            if kw.arg == "subset":
                single_kw = self._const_str(kw.value)
                cand2 = self._const_str_list(kw.value) or (
                    None if single_kw is None else [single_kw]
                )
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
        if subset is not None and input_frame.rest is None:
            # On an open frame a missing subset column may exist among the
            # unknown extras (ADR-0006) — not provably absent, stay silent.
            for s in subset:
                if s not in input_frame.columns:
                    self.errors.append(tag(PLY005, f"drop_nulls: column '{s}' not found"))
        return FrameType(
            columns=result_columns,
            strict=input_frame.strict,
            rest=input_frame.rest,
            absent=input_frame.absent,
            nonstrict_schema=input_frame.nonstrict_schema,
        )

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

    def _infer_frame_literal(self, node: ast.Call, lazy: bool) -> FrameType | None:
        """Infer the schema of ``pl.DataFrame({...})`` / ``pl.LazyFrame({...})``
        literal constructors (issue #25).

        Column names come from the data dict-literal keys; per-column dtypes
        from ``_frame_literal_value_dtype``. An explicit ``schema=`` wins
        entirely (it defines names + dtypes — even renaming data columns),
        whether the data argument is present, opaque, or missing; the
        list-of-names form gives those names ``Unknown`` dtypes.
        ``schema_overrides=`` patches individual data-inferred columns.

        Returns ``None`` when neither a readable data dict nor a readable
        ``schema=`` exists (``pl.DataFrame(some_var)`` is uninferable).
        The result is a closed, non-strict frame; ``is_lazy`` reflects the
        constructor used.
        """
        schema_kw: ast.expr | None = None
        overrides_kw: ast.expr | None = None
        for kw in node.keywords:
            if kw.arg == "schema":
                schema_kw = kw.value
            elif kw.arg == "schema_overrides":
                overrides_kw = kw.value

        columns: dict[str, DataType] = {}

        if schema_kw is not None:
            if isinstance(schema_kw, ast.Dict):
                for key_node, val_node in zip(schema_kw.keys, schema_kw.values, strict=False):
                    if key_node is None:
                        return None
                    name = _str_constant(key_node)
                    if name is None:
                        return None
                    columns[name] = _resolve_schema_dtype(val_node)
                return FrameType(columns=columns, is_lazy=lazy)
            names = _str_list_or_tuple(schema_kw)
            if names is not None:
                return FrameType(columns={n: Unknown() for n in names}, is_lazy=lazy)
            # Opaque ``schema=`` (a variable) — even the column names are
            # unknowable, so the whole literal is uninferable.
            return None

        if not node.args or not isinstance(node.args[0], ast.Dict):
            return None
        data = node.args[0]
        for key_node, val_node in zip(data.keys, data.values, strict=False):
            if key_node is None:
                # ``{**spread}`` — names unknowable.
                return None
            name = _str_constant(key_node)
            if name is None:
                return None
            columns[name] = _frame_literal_value_dtype(val_node, self._lookup_const)

        if isinstance(overrides_kw, ast.Dict):
            for key_node, val_node in zip(overrides_kw.keys, overrides_kw.values, strict=False):
                if key_node is None:
                    continue
                name = _str_constant(key_node)
                if name is None or name not in columns:
                    continue
                columns[name] = _resolve_schema_dtype(val_node)

        return FrameType(columns=columns, is_lazy=lazy)

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
            s = self._const_str(arg)
            if s is not None:
                targets.append(s)
                continue
            lst = self._const_str_list(arg)
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
            # Both containers explode to their element dtype (probed:
            # Array(Int64, 3) explodes to Int64; issue #53).
            if not isinstance(inner, (ListT, Array)):
                self.errors.append(
                    tag(PLY021, f"explode: column '{col}' is {spec.dtype}, not List/Array")
                )
                continue
            elem_dtype: DataType = inner.inner
            if outer_nullable:
                elem_dtype = Nullable(elem_dtype)
            result_columns[col] = ColumnSpec(dtype=elem_dtype, required=spec.required)
        return FrameType(
            columns=result_columns,
            strict=input_frame.strict,
            rest=input_frame.rest,
            absent=input_frame.absent,
            nonstrict_schema=input_frame.nonstrict_schema,
        )

    def _infer_unpivot_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        index: list[str] = []
        on: list[str] = []
        variable_name = "variable"
        value_name = "value"
        for kw in node.keywords:
            if kw.arg == "index":
                lst = self._const_str_list(kw.value)
                single = self._const_str(kw.value)
                if lst is not None:
                    index = lst
                elif single is not None:
                    index = [single]
            elif kw.arg == "on":
                lst = self._const_str_list(kw.value)
                single = self._const_str(kw.value)
                if lst is not None:
                    on = lst
                elif single is not None:
                    on = [single]
            elif kw.arg == "variable_name":
                cand = self._const_str(kw.value)
                if cand is not None:
                    variable_name = cand
            elif kw.arg == "value_name":
                cand = self._const_str(kw.value)
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

    def _infer_to_dummies_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """``df.to_dummies(columns)`` — value-dependent, like pivot (issue #74).

        The output columns are one UInt8 indicator per *runtime value* of
        each dummied column (``c`` -> ``c_a``, ``c_b``, ...; probed on
        polars 1.41.2, identical on 1.37.0), which polypolarism cannot
        see. Same treatment as ``pivot``: emit ``PLW005`` with an
        annotation suggestion instead of silently failing inference.
        """
        # ``columns`` is the only positional parameter (probed signature:
        # ``to_dummies(columns=None, *, separator, drop_first, drop_nulls)``).
        cols_node: ast.expr | None = node.args[0] if node.args else None
        for kw in node.keywords:
            if kw.arg == "columns":
                cols_node = kw.value
        dummied: list[str] | None = None
        if cols_node is None:
            # No selection: every column is dummied (only enumerable on a
            # closed frame).
            if input_frame.rest is None:
                dummied = list(input_frame.columns)
        else:
            single = _str_constant(cols_node)
            multi = _str_list_or_tuple(cols_node)
            if single is not None:
                dummied = [single]
            elif multi is not None:
                dummied = multi

        # Build a copy-pasteable hint when the call shape is readable and
        # the receiver is closed: passthrough columns keep their dtype,
        # dummied columns expand to one UInt8 indicator per value.
        hint = ""
        if dummied is not None and input_frame.rest is None:
            passthrough_lines = [
                f"{name}: pl.{spec.dtype}"
                for name, spec in input_frame.columns.items()
                if name not in dummied
            ]
            dummied_lines = [
                f"# one pl.UInt8 column per value of '{c}' (named '{c}_<value>')" for c in dummied
            ]
            hint = (
                " Suggested annotation:\n"
                "      class DummiesOut(pa.DataFrameModel):\n"
                + "".join(f"          {line}\n" for line in passthrough_lines)
                + "".join(f"          {line}\n" for line in dummied_lines)
                + "      result: DataFrame[DummiesOut] = df.to_dummies(...)"
            )

        self.warnings.append(
            tag(
                PLW005,
                "to_dummies: output column names depend on the runtime "
                "values of the dummied columns, so polypolarism cannot "
                "infer the schema. Assign the result to a "
                "`DataFrame[Schema]`-annotated variable to give the "
                "analyser a schema to check against." + hint,
            )
        )
        return None

    def _infer_null_count_call(self, input_frame: FrameType) -> FrameType:
        """``df.null_count()`` — same column names, every dtype UInt32 (issue #74).

        Probed (polars 1.41.2, identical on 1.37.0): one row holding the
        per-column null tally; every column — Nullable or not — maps to a
        non-null UInt32 of the same name, on both DataFrame and LazyFrame.
        Optional columns stay optional (absent input column, absent tally);
        an open receiver stays open (unknown extras are tallied too).
        """
        columns = {
            name: ColumnSpec(dtype=UInt32(), required=spec.required)
            for name, spec in input_frame.columns.items()
        }
        return FrameType(
            columns, rest=input_frame.rest, is_lazy=input_frame.is_lazy, absent=input_frame.absent
        )

    def _infer_upsample_call(self, input_frame: FrameType, node: ast.Call) -> FrameType:
        """``df.upsample(time_column, every=..., group_by=...)`` (issue #74).

        Identity schema, but the inserted gap rows are null-filled in every
        column except the keys — so non-key columns become Nullable.
        Probed (polars 1.41.2, identical on 1.37.0): the time column keeps
        its dtype (Datetime unit included) and stays non-null; ``group_by``
        columns are filled per group and stay non-null; everything else
        gains nulls. Eager-only — ``pl.LazyFrame`` has no ``upsample``
        (PLY030 is raised by the eager/lazy gate before this runs).
        """
        # Keys: ``time_column`` (the only positional parameter; probed
        # signature ``upsample(time_column, *, every, group_by=None,
        # maintain_order=False)``) plus the ``group_by`` column(s).
        keys: set[str] = set()
        if node.args:
            cand = _str_constant(node.args[0])
            if cand is not None:
                keys.add(cand)
        for kw in node.keywords:
            if kw.arg == "time_column":
                cand = _str_constant(kw.value)
                if cand is not None:
                    keys.add(cand)
            elif kw.arg == "group_by":
                single = _str_constant(kw.value)
                multi = _str_list_or_tuple(kw.value)
                if single is not None:
                    keys.add(single)
                elif multi is not None:
                    keys.update(multi)
        columns: dict[str, ColumnSpec] = {}
        for name, spec in input_frame.columns.items():
            if name in keys or isinstance(spec.dtype, Nullable):
                columns[name] = spec
            else:
                columns[name] = ColumnSpec(dtype=Nullable(spec.dtype), required=spec.required)
        return FrameType(
            columns, rest=input_frame.rest, is_lazy=input_frame.is_lazy, absent=input_frame.absent
        )

    def _infer_join_where_call(self, input_frame: FrameType, node: ast.Call) -> FrameType:
        """``df.join_where(other, *predicates)`` — degrade, don't guess (issue #74).

        polars documents the method as experimental ("It may be changed at
        any point without it being considered a breaking change"), so
        encoding its schema would couple polypolarism to an API polars
        reserves the right to change silently. Instead of the old hard
        "Could not infer return type" we return an OPEN frame (correct
        code passes via the open-frame leniency, visibly) and surface a
        PLW007 so the degradation is reviewable. The observed schema —
        left + right columns with ``_right`` suffix on collisions, probed
        identical on polars 1.37.0–1.41.2 — is a candidate for precise
        inference if/when polars stabilizes the API.
        """
        # Walk the ``other`` frame argument so diagnostics nested in it
        # (e.g. ``a.join_where(b.select(...), ...)``) still surface.
        if node.args:
            self._infer_expr_type(node.args[0])
        # Deduped per source call: ``.agg()`` chains analyze the grouped
        # receiver twice (laziness probe + _infer_agg_call).
        key = (node.lineno, node.col_offset, "join_where")
        if key not in self._warned_frame_calls:
            self._warned_frame_calls.add(key)
            self.warnings.append(
                tag(
                    PLW007,
                    "`.join_where()` is experimental in polars (its schema "
                    "may change without notice), so polypolarism does not "
                    "track its result schema — downstream checks weaken. "
                    "Validate the result against a schema "
                    "(`Schema.validate(...)`) to keep checking precise.",
                )
            )
        return FrameType({}, rest=RowVar("join_where"), is_lazy=input_frame.is_lazy)

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
            if inner.open:
                # OPEN struct (backlog C-9): fields beyond the pinned ones
                # are unknown — the result frame opens, like the Unknown
                # branch above, but the pinned fields still register.
                result_rest = RowVar("unnest")
            for field_name, field_dtype in inner.fields.items():
                wrapped: DataType = field_dtype
                if outer_nullable and not isinstance(wrapped, Nullable):
                    wrapped = Nullable(wrapped)
                result_columns[field_name] = ColumnSpec(dtype=wrapped, required=spec.required)
        return FrameType(columns=result_columns, strict=input_frame.strict, rest=result_rest)

    def _infer_filter_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """Identity-typed, but validate every predicate (issue #28).

        Each positional predicate is walked through the expression analyzer
        (so missing-column refs keep surfacing PLY001) and its resolved
        dtype is checked: a known non-Boolean predicate is PLY008. A bare
        string constant (or a constant-bound name) is a column reference,
        not a Utf8 literal. Unresolved / Unknown dtypes are never flagged.
        Kwarg constraints (``filter(a=1)``) are equality comparisons —
        boolean by construction, so only their value expressions are walked.
        """
        expr_analyzer = ExpressionAnalyzer(
            input_frame,
            warnings=self.warnings,
            registry=self.registry,
            int_consts=self._int_consts(),
        )
        for arg in node.args:
            dtype: DataType | None
            col_ref = self._const_str(arg)
            if col_ref is not None:
                # ``filter("flag")`` ≡ ``filter(pl.col("flag"))``: resolve
                # against the frame — PLY001 on closed frames, Unknown on
                # open ones (the column may be among the unseen extras).
                try:
                    dtype = infer_col(col_ref, input_frame)
                except ColumnNotFoundError as e:
                    expr_analyzer.errors.append(tag(PLY001, str(e)))
                    continue
            else:
                _, dtype = expr_analyzer.analyze_select_expr(arg)
            pred_error = _nonboolean_predicate_error(dtype)
            if pred_error is not None:
                expr_analyzer.errors.append(pred_error)
        for kw in node.keywords:
            expr_analyzer.analyze_select_expr(kw.value)
        self.errors.extend(expr_analyzer.errors)
        return input_frame

    def _infer_sort_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """Identity-typed, but validate that every sort key exists (issue #29).

        Keys come from positional args and the ``by=`` kwarg: string
        constants, list/tuple-of-string literals, constant-bound names,
        selectors (which resolve against the frame, so they can't name a
        missing column) and ``pl.col(...)`` expressions (walked through the
        expression analyzer so PLY001 fires). A string key missing from a
        closed frame is PLY007; open frames stay error-free. The modifier
        kwargs (``descending=`` / ``nulls_last=`` / ``maintain_order=`` /
        ``multithreaded=``) are ignored.
        """
        key_nodes: list[ast.expr] = list(node.args)
        for kw in node.keywords:
            if kw.arg == "by":
                key_nodes.append(kw.value)

        expr_analyzer = ExpressionAnalyzer(
            input_frame,
            warnings=self.warnings,
            registry=self.registry,
            int_consts=self._int_consts(),
        )
        for key_node in key_nodes:
            if _resolve_selector(key_node, input_frame) is not None:
                continue
            single = self._const_str(key_node)
            names = [single] if single is not None else self._const_str_list(key_node)
            if names is None:
                # Expression key (``pl.col(...)`` chains etc.) — walk it for
                # its PLY001 side effects; anything unrecognised stays silent.
                expr_analyzer.analyze_select_expr(key_node)
                continue
            for name in names:
                if name not in input_frame.columns and input_frame.rest is None:
                    self.errors.append(tag(PLY007, f"sort: column '{name}' not found"))
        self.errors.extend(expr_analyzer.errors)
        return input_frame

    def _infer_unique_call(self, input_frame: FrameType, node: ast.Call) -> FrameType | None:
        """Identity-typed, but validate the ``subset=`` columns (issue #35).

        ``unique(subset=None, *, keep, maintain_order)`` — the subset can be
        passed positionally or as the ``subset=`` kwarg: a string constant, a
        list/tuple of strings, a constant-bound name, or a selector (which
        resolves against the frame, so it can't name a missing column). A
        subset column missing from a closed frame is PLY014 — polars raises
        ColumnNotFoundError at runtime; open frames stay lenient. The
        ``keep=`` / ``maintain_order=`` modifiers are ignored.
        """
        subset_nodes: list[ast.expr] = list(node.args[:1])
        for kw in node.keywords:
            if kw.arg == "subset":
                subset_nodes.append(kw.value)

        for subset_node in subset_nodes:
            if _resolve_selector(subset_node, input_frame) is not None:
                continue
            single = self._const_str(subset_node)
            names = [single] if single is not None else self._const_str_list(subset_node)
            if names is None:
                # Expression subset (``pl.col(...)`` etc.) or an unresolvable
                # variable — stay silent rather than guess.
                continue
            for name in names:
                if name not in input_frame.columns and input_frame.rest is None:
                    self.errors.append(tag(PLY014, f"unique: subset column '{name}' not found"))
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
        return FrameType(
            columns=result_columns,
            strict=input_frame.strict,
            rest=input_frame.rest,
            absent=input_frame.absent,
            nonstrict_schema=input_frame.nonstrict_schema,
        )


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
            if frame_type is None:
                # Bare ``pl.DataFrame`` / ``pl.LazyFrame`` params (ADR-0006)
                # register as empty open frames so call sites type-check
                # (any frame satisfies them) and laziness is enforced.
                bare_head = bare_frame_annotation(arg.annotation)
                if bare_head is not None:
                    frame_type = FrameType(
                        {}, rest=RowVar(arg.arg), is_lazy=bare_head == "LazyFrame"
                    )
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
    module_consts: dict[str, str | list[str] | int] | None = None,
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

    # Schema names already flagged PLY041 (issue #69): a schema whose
    # ``Annotated`` field arity provably crashes pandera is reported once
    # per function, however many annotation sites reference it. Shared with
    # the body analyzer below. ``degraded_schemas_reported`` is the same
    # dedup contract for PLW011 (issue #77: unrecognized field annotations
    # degraded to Unknown-dtype columns).
    broken_schemas_reported: set[str] = set()
    degraded_schemas_reported: set[str] = set()

    def _note_schema_reference(annotation: ast.expr) -> None:
        schema_name = frame_annotation_schema_name(annotation)
        if schema_name is None:
            return
        if schema_name not in broken_schemas_reported:
            error = _schema_definition_error(schema_name, schema_registry)
            if error is not None:
                broken_schemas_reported.add(schema_name)
                errors.append(error)
        if schema_name not in degraded_schemas_reported:
            warning = _schema_definition_warning(schema_name, schema_registry)
            if warning is not None:
                degraded_schemas_reported.add(schema_name)
                warnings.append(warning)

    # Extract input parameter types
    for arg in func_node.args.args:
        if arg.annotation:
            if _annotation_declares_frame(arg.annotation, schema_registry):
                has_df_annotation = True
                _note_schema_reference(arg.annotation)
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
                else:
                    # Bare ``pl.DataFrame`` / ``pl.LazyFrame`` (ADR-0006):
                    # the user declared a frame without a schema — bind an
                    # empty OPEN frame and check what the body determines.
                    bare_head = bare_frame_annotation(arg.annotation)
                    if bare_head is not None:
                        has_df_annotation = True
                        input_types[arg.arg] = FrameType(
                            {},
                            rest=RowVar(arg.arg),
                            is_lazy=bare_head == "LazyFrame",
                        )

    # Extract return type
    if func_node.returns:
        if _annotation_declares_frame(func_node.returns, schema_registry):
            has_df_annotation = True
            _note_schema_reference(func_node.returns)
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
            elif bare_frame_annotation(func_node.returns) is not None:
                # A bare frame return opts the function in but makes no
                # schema claim — nothing to check against the inferred
                # return (ADR-0006; the eager/lazy bit is future work).
                has_df_annotation = True

    # If no DataFrame annotations found, skip this function
    if not has_df_annotation:
        return None

    # Surface unresolved-schema names so the user sees the file isn't
    # being silently skipped because of a missing import. Deduplicate to
    # one warning per name.
    for name in dict.fromkeys(unresolved_schemas):
        if "." in name:
            # Module-qualified reference (``DataFrame[mod.Schema]``) that
            # didn't resolve through a project-local plain import.
            module, attr = name.rsplit(".", 1)
            hint = (
                f"schema '{name}' referenced in annotation but not found. "
                f"Qualified references resolve through a top-level "
                f"`import {module}` of a project-local module defining "
                f"'{attr}'. Stdlib/third-party imports aren't followed, "
                f"and nested classes aren't supported."
            )
        else:
            hint = (
                f"schema '{name}' referenced in annotation but not found. "
                f"Define it in this module, or import it from a project-local "
                f"module via `from <module> import {name}`. "
                f"Stdlib/third-party imports aren't followed."
            )
        warnings.append(tag(PLW006, hint))

    # Analyze function body with registry
    body_analyzer = FunctionBodyAnalyzer(
        input_types,
        errors,
        registry,
        schema_registry,
        warnings=warnings,
        class_registry=class_registry,
        current_class_name=current_class_name,
        module_consts=module_consts,
        reported_broken_schemas=broken_schemas_reported,
        reported_degraded_schemas=degraded_schemas_reported,
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


def _collect_module_consts(tree: ast.Module) -> dict[str, str | list[str] | int]:
    """Collect top-level ``NAME = "lit"`` / ``NAME = ["a", "b"]`` / ``N = 1``
    constants.

    Only single-``Name``-target ``Assign`` / ``AnnAssign`` statements whose
    value is a string constant, a list/tuple of string constants, or an int
    literal (bools excluded) are recorded. These feed constant resolution
    for column-spec arguments (``join(on=KEY)``, ``unpivot(on=ON_COLS)``,
    ...) and int-valued call args (rolling ``min_samples``/``window_size``/
    ``ddof``; backlog B-5).
    """
    consts: dict[str, str | list[str] | int] = {}
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        const_val: str | list[str] | int | None = _str_constant(value)
        if const_val is None:
            const_val = _str_list_or_tuple(value)
        if const_val is None:
            const_val = _int_literal(value)
        if const_val is not None:
            consts[target.id] = const_val
        else:
            # Rebinding the name to a non-constant at module level drops
            # any earlier constant — we can't tell which value wins.
            consts.pop(target.id, None)
    return consts


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

    # Module-level string(-list) constants for column-spec resolution.
    module_consts = _collect_module_consts(tree)

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
            module_consts=module_consts,
        )
        if analysis:
            results.append(analysis)

    return results
