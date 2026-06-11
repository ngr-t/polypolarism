"""Polars surface tables consumed by the analyzer.

Single source of truth for the polars 1.x surface that polypolarism
recognizes:

- ``DTYPE_NAME_MAP`` — bare-attribute dtype names (``pl.Int64``,
  ``pl.String``, ``pl.Decimal`` …). Parametrized dtypes
  (``pl.Datetime("us", "UTC")``, ``pl.Decimal(20, 4)``) are matched
  by Call form in dispatch sites; ``DECIMAL_DEFAULT`` is the bare
  ``pl.Decimal`` fallback.
- ``IDENTITY_FRAME_METHODS`` / ``LAZY_ONLY_METHODS`` /
  ``EAGER_ONLY_METHODS`` / ``EAGER_FRAME_RETURNING_METHODS`` /
  ``LAZY_FRAME_RETURNING_METHODS`` — frame method classification.
- ``STR_NAMESPACE_RETURN`` / ``DT_NAMESPACE_RETURN`` /
  ``DT_NAMESPACE_PRESERVING`` / ``LIST_NAMESPACE_PRESERVING`` /
  ``LIST_NAMESPACE_ELEMENT_RETURN`` / ``BIN_NAMESPACE_RETURN`` /
  ``ARR_NAMESPACE_*`` / ``CAT_NAMESPACE_RETURN`` /
  ``container_agg_return`` — sub-namespace return tables.
- ``JOIN_HOW_VALUES`` / ``JOIN_HOW_INFERRED`` and the
  ``join_left_nullable`` / ``join_right_nullable`` predicates.
- ``agg_function_for(name)`` / ``AGG_SHORTHAND_NAMES`` — polars-side
  aggregation name lookup. (The actual signature/inference logic lives
  in ``ops/groupby.py``.)
- ``METHOD_ALIASES`` + ``canonicalize_method`` — rename / variant
  canonicalization applied before dispatch (``select_seq`` → ``select``).
- ``PolarsProfile`` / ``POLARS_1_X`` — version-conditional behavior
  scaffold (no fields yet).

See ADR-0001 (``docs/adr/0001-polars-pandera-version-support.md``) for
the policy that motivates the centralization.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polypolarism.types import (
    Array,
    Binary,
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Null,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Utf8,
)
from polypolarism.types import List as ListT

if TYPE_CHECKING:
    from polypolarism.ops.groupby import AggFunction


@dataclass(frozen=True)
class PolarsProfile:
    """Version-conditional behavior knobs.

    ADR-0001 ships this as a name-only scaffold. Fields get added when a
    real fixture exposes a divergence between supported minors that the
    analyzer cares about — speculative fields rot. The selection
    mechanism (``--polars-version`` / ``[tool.polypolarism]``) is wired in
    `version_check.py`; the *consumption* path (analyzer dispatch
    branching on profile fields) doesn't exist yet because there are no
    fields to branch on.
    """

    name: str


POLARS_1_X = PolarsProfile(name="1.x")
DEFAULT_POLARS_PROFILE = POLARS_1_X


# Polars' own ``pl.Decimal()`` defaults (precision=38, scale=0). Used when
# the analyzer sees a bare ``pl.Decimal`` reference; explicit
# ``pl.Decimal(p, s)`` is parsed via ``parse_decimal_call`` to preserve p / s.
DECIMAL_DEFAULT = Decimal(38, 0)


def _decimal_call_arg(call: ast.Call, *, position: int, name: str, default: int) -> int | None:
    """One ``pl.Decimal(...)`` argument: positional or keyword.

    Returns ``default`` when the argument is omitted (or an explicit
    ``None`` literal — polars substitutes its own default), the value for
    an integer literal, and ``None`` when the argument is present but not
    statically readable (a variable, an expression, ...).
    """
    node: ast.expr | None = None
    if position < len(call.args):
        node = call.args[position]
    for kw in call.keywords:
        if kw.arg == name:
            node = kw.value
    if node is None:
        return default
    if isinstance(node, ast.Constant):
        if node.value is None:
            return default
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
    return None


def parse_decimal_call(node: ast.Call) -> Decimal | None:
    """Parse a ``pl.Decimal(...)`` call form into a ``Decimal`` dtype.

    Shared by the analyzer (``cast`` targets, ``schema=`` dicts) and
    ``pandera_dtype`` (schema field annotations). Omitted arguments take
    polars' own defaults — ground truth on 1.41.2: ``pl.Decimal()`` →
    ``Decimal(38, 0)``, ``pl.Decimal(10)`` → ``Decimal(10, 0)``,
    ``pl.Decimal(scale=2)`` → ``Decimal(38, 2)``. Returns ``None`` when an
    argument is present but not an integer literal: the precision/scale are
    unknowable and each caller picks its own fallback (the analyzer
    degrades to unresolved/Unknown; pandera_dtype keeps the bare default).
    """
    precision = _decimal_call_arg(
        node, position=0, name="precision", default=DECIMAL_DEFAULT.precision
    )
    scale = _decimal_call_arg(node, position=1, name="scale", default=DECIMAL_DEFAULT.scale)
    if precision is None or scale is None:
        return None
    return Decimal(precision, scale)


def parse_int_shape(node: ast.expr) -> int | None:
    """An int literal, or a 1-tuple of one, as an Array width (backlog C-7).

    Multi-dimensional tuples and non-literal expressions are statically
    unknowable -> ``None`` (the checker treats an unknown width as a
    wildcard).
    """
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Tuple) and len(node.elts) == 1:
        elt = node.elts[0]
        if isinstance(elt, ast.Constant) and type(elt.value) is int:
            return elt.value
    return None


def parse_array_shape(node: ast.Call) -> int | None:
    """Width of a ``pl.Array(inner, shape)`` call (backlog C-7).

    Shared by the analyzer (``cast`` targets) and ``pandera_dtype``
    (schema field annotations). The shape is the second positional
    argument or the ``shape=`` keyword; see :func:`parse_int_shape` for
    the accepted literal forms.
    """
    arg: ast.expr | None = node.args[1] if len(node.args) > 1 else None
    if arg is None:
        for kw in node.keywords:
            if kw.arg == "shape":
                arg = kw.value
                break
    if arg is None:
        return None
    return parse_int_shape(arg)


# The three time units polars 1.x accepts for Datetime / Duration, ordered
# coarse -> fine. Mixed-unit operations (arithmetic, when/then, concat)
# resolve to the COARSER operand unit (probed 1.41.2: us + ns -> us,
# ms + ns -> ms, in every operand order). Casting toward a coarser unit is
# value-independent (a division); casting toward a finer unit multiplies
# and overflows for extreme values (probed: Datetime[us] year 9999 -> ns
# raises InvalidOperationError).
TIME_UNITS: tuple[str, ...] = ("ms", "us", "ns")

_TIME_UNIT_RANK: dict[str, int] = {u: i for i, u in enumerate(TIME_UNITS)}


def coarser_time_unit(left: str, right: str) -> str:
    """The coarser of two time units (the probed mixed-unit result)."""
    return left if _TIME_UNIT_RANK[left] <= _TIME_UNIT_RANK[right] else right


def time_unit_refines(source: str, target: str) -> bool:
    """True when casting ``source -> target`` moves to a finer unit
    (multiplication — overflows for extreme values, hence value-dependent)."""
    return _TIME_UNIT_RANK[target] > _TIME_UNIT_RANK[source]


def parse_time_unit(node: ast.expr | None) -> str | None:
    """One time_unit argument: ``None`` literal / omitted -> polars' "us"
    default; a literal in TIME_UNITS -> itself; anything else (a variable,
    an invalid literal) -> ``None`` for "not statically readable"."""
    if node is None:
        return "us"
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "us"
        if isinstance(node.value, str) and node.value in TIME_UNITS:
            return node.value
    return None


def parse_datetime_call(node: ast.Call) -> Datetime | None:
    """Parse a ``pl.Datetime(...)`` call form into a ``Datetime`` dtype.

    Shared by the analyzer (``cast`` targets, ``schema=`` dicts) and
    ``pandera_dtype`` (schema field annotations) — issues #50 / #66. The
    polars signature is ``pl.Datetime(time_unit="us", time_zone=None)``:
    the first positional argument / ``time_unit=`` keyword sets the unit
    (omitted or an explicit ``None`` literal -> polars' "us" default), the
    second positional argument / ``time_zone=`` keyword sets the tz (a
    string literal; omitted or ``None`` -> tz-naive).

    Returns ``None`` when either argument is present but not statically
    readable (a variable, a ``timezone`` object, an invalid unit literal):
    the dtype is unknowable and each caller picks its own fallback (the
    analyzer degrades to unresolved; ``pandera_dtype`` uses ``Unknown``).
    Claiming the default would be a false-positive trap now that tz and
    unit mismatches are flagged.
    """
    unit_node: ast.expr | None = node.args[0] if node.args else None
    tz_node: ast.expr | None = None
    if len(node.args) >= 2:
        tz_node = node.args[1]
    for kw in node.keywords:
        if kw.arg == "time_unit":
            unit_node = kw.value
        if kw.arg == "time_zone":
            tz_node = kw.value
    unit = parse_time_unit(unit_node)
    if unit is None:
        return None
    if tz_node is None:
        return Datetime(unit=unit)
    if isinstance(tz_node, ast.Constant):
        if tz_node.value is None:
            return Datetime(unit=unit)
        if isinstance(tz_node.value, str):
            return Datetime(tz=tz_node.value, unit=unit)
    return None


def parse_duration_call(node: ast.Call) -> Duration | None:
    """Parse a ``pl.Duration(...)`` call form into a ``Duration`` dtype
    (issue #66; signature ``pl.Duration(time_unit="us")``). ``None`` when
    the unit argument is present but not statically readable."""
    unit_node: ast.expr | None = node.args[0] if node.args else None
    for kw in node.keywords:
        if kw.arg == "time_unit":
            unit_node = kw.value
    unit = parse_time_unit(unit_node)
    if unit is None:
        return None
    return Duration(unit=unit)


def parse_enum_call(node: ast.Call) -> Enum:
    """Parse a ``pl.Enum([...])`` call form into an ``Enum`` dtype (issue #67).

    The category list is the first positional argument or the
    ``categories=`` keyword; a list/tuple literal of string constants
    yields the ordered category tuple. Anything else (a variable, a
    ``pl.Series``, mixed elements) is statically unreadable — the result
    is ``Enum(categories=None)``, the "some Enum, categories unknown"
    wildcard (the call still provably constructs an Enum, so degrading
    all the way to Unknown would lose precision). Total — never ``None``.
    """
    cats_node: ast.expr | None = node.args[0] if node.args else None
    for kw in node.keywords:
        if kw.arg == "categories":
            cats_node = kw.value
    if isinstance(cats_node, (ast.List, ast.Tuple)):
        cats: list[str] = []
        for elt in cats_node.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                return Enum()
            cats.append(elt.value)
        return Enum(categories=tuple(cats))
    return Enum()


# Single source of truth for ``pl.<Name>`` attribute → DataType. Both the
# analyzer (for column / cast inference) and pandera_dtype (for schema
# field annotations) consume this mapping.
#
# Versioning notes:
# - ``String`` is the polars 1.x canonical name for what was historically
#   ``Utf8``; both are accepted and resolve to ``Utf8()`` (polars itself
#   ships ``Utf8`` as an alias of ``String``).
# - ``Int128`` is polars 1.18+, ``Enum`` stabilized 1.25+, ``UInt128``
#   1.34+, ``Float16`` 1.36+. They appear here unconditionally — the
#   analyzer doesn't differentiate between supported minors today, and
#   recognizing newer names on older polars is harmless (the user's code
#   wouldn't have run anyway).
# - The container dtypes ``List`` / ``Array`` / ``Struct`` are deliberately
#   absent: a bare ``pl.List`` / ``pl.Array`` carries no element dtype, so
#   the analyzer treats it as unresolved in cast / ``schema=`` positions
#   (the schema-annotation parser in ``pandera_dtype`` has its own bare-
#   container fallbacks). Their call forms are matched in dispatch sites.
DTYPE_NAME_MAP: dict[str, DataType] = {
    "Int8": Int8(),
    "Int16": Int16(),
    "Int32": Int32(),
    "Int64": Int64(),
    "Int128": Int128(),
    "UInt8": UInt8(),
    "UInt16": UInt16(),
    "UInt32": UInt32(),
    "UInt64": UInt64(),
    "UInt128": UInt128(),
    "Float16": Float16(),
    "Float32": Float32(),
    "Float64": Float64(),
    "Utf8": Utf8(),
    "String": Utf8(),
    "Boolean": Boolean(),
    "Binary": Binary(),
    "Date": Date(),
    "Time": Time(),
    "Datetime": Datetime(),
    "Duration": Duration(),
    "Categorical": Categorical(),
    "Decimal": DECIMAL_DEFAULT,
    "Enum": Enum(),
    "Null": Null(),
}


# Join ``how`` literal values accepted by polars 1.x. ``outer`` was renamed
# to ``full`` at 1.0; we don't accept the legacy spelling.
JOIN_HOW_VALUES: frozenset[str] = frozenset(
    {"inner", "left", "right", "full", "cross", "semi", "anti"}
)

# ``how`` values where the analyzer infers a column-shape result via
# ``ops/join.infer_join`` — ``semi`` / ``anti`` return the left frame's
# schema unchanged, ``cross`` concatenates both schemas. Every polars
# ``how`` is now inferred, so this mirrors ``JOIN_HOW_VALUES``. Kept as a
# tuple for use in ``Literal[...]`` typing.
JOIN_HOW_INFERRED: tuple[str, ...] = (
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "semi",
    "anti",
)


def join_left_nullable(how: str) -> bool:
    """Whether the left-side columns should be wrapped in ``Nullable`` for
    a given join type. Right and full joins introduce nulls on the left."""
    return how in ("right", "full")


def join_right_nullable(how: str) -> bool:
    """Whether the right-side columns should be wrapped in ``Nullable``."""
    return how in ("left", "full")


# Aggregation method-name → AggFunction enum. Single source of truth for
# the analyzer's recognition of ``.sum()`` / ``.mean()`` / etc. on
# expressions and ``pl.sum("col")`` / ``pl.mean("col")`` top-level calls.
#
# ``AGG_NAME_MAP`` covers every aggregation polypolarism currently models;
# ``AGG_SHORTHAND_NAMES`` is the strict subset that polars exposes as a
# top-level shorthand (``pl.<name>("col")``) — ``list``, ``quantile``, and
# ``product`` are method-only.
def _build_agg_name_map() -> dict[str, AggFunction]:
    # Imported lazily to avoid the compat -> ops/groupby -> compat cycle
    # that would otherwise form (ops/groupby imports compat for join, but
    # compat needs the AggFunction enum). The enum lives in ops/groupby
    # for now; a future refactor may pull it here too.
    from polypolarism.ops.groupby import AggFunction

    return {
        "sum": AggFunction.SUM,
        "mean": AggFunction.MEAN,
        "count": AggFunction.COUNT,
        # ``Expr.len()`` is the count-including-nulls variant of
        # ``Expr.count()`` — same UInt32 result dtype (issue #23). Method
        # form only: zero-arg ``pl.len()`` is handled separately and
        # ``len`` is deliberately NOT in AGG_SHORTHAND_NAMES.
        "len": AggFunction.COUNT,
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


AGG_SHORTHAND_NAMES: frozenset[str] = frozenset(
    {"sum", "mean", "min", "max", "first", "last", "count", "n_unique", "median", "std", "var"}
)


# Top-level ``pl.<name>(...)`` IO readers whose return annotation is
# unconditionally ``DataFrame`` / ``LazyFrame`` (probed on polars 1.41.2 via
# ``inspect.signature``; ADR-0006). Union-typed readers are deliberately
# absent, mirroring the EAGER_FRAME_RETURNING_METHODS policy:
# ``read_database`` (-> DataFrame | Iterator), ``read_excel`` / ``read_ods``
# (-> DataFrame | dict), ``read_csv_batched`` (a reader object), and the
# ``read_*_schema`` / ``read_parquet_metadata`` dict returns. The analyzer
# infers an EMPTY OPEN frame for these — the file's schema is unknown and
# deliberately not read at check time (hermetic checks; C-12b).
EAGER_READ_FUNCTIONS: frozenset[str] = frozenset(
    {
        "read_avro",
        "read_clipboard",
        "read_csv",
        "read_database_uri",
        "read_delta",
        "read_ipc",
        "read_ipc_stream",
        "read_json",
        "read_lines",
        "read_ndjson",
        "read_parquet",
    }
)

LAZY_SCAN_FUNCTIONS: frozenset[str] = frozenset(
    {
        "scan_csv",
        "scan_delta",
        "scan_iceberg",
        "scan_ipc",
        "scan_lines",
        "scan_ndjson",
        "scan_parquet",
        "scan_pyarrow_dataset",
    }
)


# Method names that dispatch as another (canonical) method. Two uses:
#
# 1. Renames within polars 1.x — none in the supported window today. The
#    day a polars minor renames a method, the fix is one entry here rather
#    than touching every dispatch site. An entry ``"new_name":
#    "canonical_name"`` means "treat new_name as canonical_name during
#    dispatch." See ADR-0001 § Decision item 3 for the policy.
# 2. Schema-equivalent variants — ``select_seq`` / ``with_columns_seq``
#    differ from ``select`` / ``with_columns`` only in evaluation order
#    (sequential instead of parallel); the resulting schema is identical,
#    so they share one inference path (issue #21). Both exist on DataFrame
#    AND LazyFrame, so no eager/lazy-only classification applies.
METHOD_ALIASES: dict[str, str] = {
    "select_seq": "select",
    "with_columns_seq": "with_columns",
}


# ``pl.col("x").str.<method>(...)`` return types. Boolean predicates,
# Utf8-returning transformations, length / count integer returns, and a
# couple of parsing-into-temporal methods.
STR_NAMESPACE_RETURN: dict[str, DataType] = {
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
    # Parsing into temporal types. ``to_datetime`` is argument-dependent
    # (tz from ``time_zone=`` / ``%z`` format, issue #50) and dispatched
    # via ``analyzer._str_to_datetime_dtype`` before this table is
    # consulted; the entry documents the no-tz-argument default.
    "to_date": Date(),
    "to_datetime": Datetime(),
    "to_time": Time(),
    # Parsing into numeric types (issue #19). ``to_decimal`` is absent:
    # its scale comes from the required ``scale=`` argument
    # (``analyzer._str_to_decimal_dtype``, issue #61) — a fixed entry here
    # was exactly the #61 false positive.
    "to_integer": Int64(),
}


# ``pl.col("ts").dt.<method>()`` returning a fixed dtype (calendar parts,
# epoch / total_* integers).
DT_NAMESPACE_RETURN: dict[str, DataType] = {
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
    # Formatting into strings — ``strftime`` is an alias of ``to_string``.
    "strftime": Utf8(),
    "to_string": Utf8(),
    # ``epoch`` is argument-dependent (issue #73): ``epoch("d")`` returns
    # Int32, the sub-second units Int64 — dispatched via
    # ``analyzer._dt_epoch_dtype`` before this table is consulted; the
    # entry documents the no-argument default (time_unit="us").
    "epoch": Int64(),
    "timestamp": Int64(),
    "total_days": Int64(),
    "total_hours": Int64(),
    "total_minutes": Int64(),
    "total_seconds": Int64(),
    "total_milliseconds": Int64(),
    "total_nanoseconds": Int64(),
    "total_microseconds": Int64(),
}

# ``pl.col("ts").dt.<method>()`` methods that preserve the receiver dtype
# (timezone / truncation / window-shift methods).
DT_NAMESPACE_PRESERVING: frozenset[str] = frozenset(
    {
        "truncate",
        "round",
        "offset_by",
        "replace_time_zone",
        "convert_time_zone",
        "month_start",
        "month_end",
    }
)

# ``pl.col("xs").list.<method>()`` methods that preserve the receiver list dtype.
LIST_NAMESPACE_PRESERVING: frozenset[str] = frozenset(
    {
        "unique",
        "sort",
        "reverse",
        "head",
        "tail",
        "slice",
        "drop_nulls",
        "sample",
        "shift",
    }
)

# ``pl.col("xs").list.<method>()`` methods that return the element dtype
# (de-listing operations). The reducers ``sum`` / ``mean`` / ``median`` /
# ``std`` / ``var`` / ``min`` / ``max`` are NOT here — they are strictly
# typed per element dtype via ``container_agg_return`` below (probed:
# ``list.mean`` on List(Int64) is Float64, and ``list.min`` on
# List(Struct) raises at runtime).
LIST_NAMESPACE_ELEMENT_RETURN: frozenset[str] = frozenset(
    {
        "get",
        "first",
        "last",
        "explode",
    }
)


# Element-wise reductions of the ``list`` and ``arr`` namespaces, strictly
# typed per (namespace, method, element dtype) — issue #55. Probed on
# polars 1.41.2 (full matrix; the two namespaces DIVERGE on several cells):
#
#   sum:  Int8/Int16/UInt8/UInt16 -> Int64 (overflow guard);
#         other int/uint widths and floats -> element; Boolean -> UInt32.
#         list-only extras: Decimal(p, s) -> Decimal(p, s), Duration ->
#         Duration, Null -> Null. INVALID (InvalidOperationError) on list:
#         Utf8 / Date / Datetime / Time / List / Array / Struct /
#         Categorical / Enum / Binary; on arr ALL of the non-core cells
#         (Duration / Decimal / Null included) raise ComputeError.
#   mean/median: Float32/Float16 keep their width (except list.mean on
#         Float16 — a probed rust panic); ints, Boolean and Decimal ->
#         Float64; Duration -> Duration. list-only: Date/Datetime ->
#         Datetime (tz kept), Time -> Time; the same arr cells are probed
#         to return Float64 (epoch numbers) and stay unclaimed.
#   std:  like mean but temporal cells beyond Duration stay unclaimed.
#   var:  like std; INVALID on list for Date / Datetime / Time / Duration
#         (InvalidOperationError) but only Duration on arr.
#   min/max: element dtype for every numeric width on both namespaces.
#         list-only: Boolean / Utf8 / temporal / Decimal / Categorical /
#         Enum / Binary / Null -> element (lexicographic etc., probed);
#         INVALID on list for List / Array / Struct; on arr EVERY
#         non-numeric cell is a probed rust panic.
#
# Degenerate probed-valid cells (e.g. ``list.mean`` over strings -> an
# all-null Float64) are deliberately unclaimed: they look like polars
# accepting-by-accident and are likelier to drift across minors.
CONTAINER_AGG_METHODS: frozenset[str] = frozenset(
    {"sum", "mean", "median", "std", "var", "min", "max"}
)


class ContainerAggInvalid:
    """Sentinel type for probed-invalid container-reduction cells.

    The (namespace, method, element) combination is a probed runtime error
    — InvalidOperationError, ComputeError, or a rust panic, depending on
    the cell. The analyzer flags PLY016 and degrades the output to Unknown.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "CONTAINER_AGG_INVALID"


CONTAINER_AGG_INVALID = ContainerAggInvalid()

_CONTAINER_SUM_WIDENS_TO_INT64 = (Int8, Int16, UInt8, UInt16)
_CONTAINER_SUM_KEEPS_ELEMENT = (
    Int32,
    Int64,
    Int128,
    UInt32,
    UInt64,
    UInt128,
    Float16,
    Float32,
    Float64,
)
_LIST_SUM_KEEPS_ELEMENT_EXTRA = (Decimal, Duration, Null)
_LIST_SUM_INVALID = (Utf8, Date, Datetime, Time, ListT, Array, Struct, Categorical, Enum, Binary)
_ARR_SUM_INVALID = (
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
    Binary,
    Null,
)
_CONTAINER_FLOAT_AGG_FLOAT64 = (
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
    Float64,
    Boolean,
    Decimal,
)
_LIST_VAR_INVALID = (Date, Datetime, Time, Duration)
_ARR_VAR_INVALID = (Duration,)
_CONTAINER_MINMAX_NUMERIC = (
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
_LIST_MINMAX_ELEMENT_EXTRA = (
    Boolean,
    Utf8,
    Date,
    Datetime,
    Duration,
    Time,
    Decimal,
    Categorical,
    Enum,
    Binary,
    Null,
)
_LIST_MINMAX_INVALID = (ListT, Array, Struct)
_ARR_MINMAX_INVALID = (
    Boolean,
    Utf8,
    Date,
    Datetime,
    Duration,
    Time,
    Decimal,
    ListT,
    Array,
    Struct,
    Categorical,
    Enum,
    Binary,
    Null,
)


def container_agg_return(
    namespace: str, method: str, element: DataType
) -> DataType | ContainerAggInvalid | None:
    """Result dtype of ``list.<agg>()`` / ``arr.<agg>()`` for one element dtype.

    Three-way verdict (issue #55):

    - a ``DataType``: the probed result dtype;
    - ``CONTAINER_AGG_INVALID``: a probed runtime error (the caller flags
      PLY016 and degrades the output to Unknown);
    - ``None``: unprobed or deliberately unclaimed — the caller degrades
      to Unknown silently.
    """
    if method == "sum":
        if isinstance(element, _CONTAINER_SUM_WIDENS_TO_INT64):
            return Int64()
        if isinstance(element, Boolean):
            return UInt32()
        if isinstance(element, _CONTAINER_SUM_KEEPS_ELEMENT):
            return element
        if namespace == "list":
            if isinstance(element, _LIST_SUM_KEEPS_ELEMENT_EXTRA):
                return element
            if isinstance(element, _LIST_SUM_INVALID):
                return CONTAINER_AGG_INVALID
        elif isinstance(element, _ARR_SUM_INVALID):
            return CONTAINER_AGG_INVALID
        return None
    if method in ("mean", "median"):
        if namespace == "list" and isinstance(element, Float16):
            # Probed oddity: ``list.mean`` on Float16 panics in rust while
            # ``list.median`` (and every arr cell) keeps Float16.
            return CONTAINER_AGG_INVALID if method == "mean" else element
        if isinstance(element, (Float16, Float32)):
            return element
        if isinstance(element, _CONTAINER_FLOAT_AGG_FLOAT64):
            return Float64()
        if isinstance(element, Duration):
            return element
        if namespace == "list":
            if isinstance(element, Date):
                return Datetime()
            if isinstance(element, Datetime):
                return element
            if isinstance(element, Time):
                return element
        return None
    if method == "std":
        if isinstance(element, (Float16, Float32)):
            return element
        if isinstance(element, _CONTAINER_FLOAT_AGG_FLOAT64):
            return Float64()
        if isinstance(element, Duration):
            return element
        return None
    if method == "var":
        invalid = _LIST_VAR_INVALID if namespace == "list" else _ARR_VAR_INVALID
        if isinstance(element, invalid):
            return CONTAINER_AGG_INVALID
        if isinstance(element, (Float16, Float32)):
            return element
        if isinstance(element, _CONTAINER_FLOAT_AGG_FLOAT64):
            return Float64()
        return None
    if method in ("min", "max"):
        if isinstance(element, _CONTAINER_MINMAX_NUMERIC):
            return element
        if namespace == "list":
            if isinstance(element, _LIST_MINMAX_ELEMENT_EXTRA):
                return element
            if isinstance(element, _LIST_MINMAX_INVALID):
                return CONTAINER_AGG_INVALID
        elif isinstance(element, _ARR_MINMAX_INVALID):
            return CONTAINER_AGG_INVALID
        return None
    return None


# ``pl.col("q").arr.<method>()`` tables (issue #53). Probed on polars
# 1.41.2 with Array receivers; the polars arr namespace is close to —
# but not identical with — the list namespace:
# - ``sort`` / ``reverse`` / ``shift`` preserve the Array dtype;
# - ``unique`` / ``head`` / ``tail`` / ``slice`` / ``to_list`` return a
#   *List* of the element dtype (the fixed width is lost);
# - ``get`` / ``first`` / ``last`` / ``explode`` return the element dtype
#   (``min`` / ``max`` are strictly typed via ``container_agg_return`` —
#   they panic in rust for every non-numeric element, issue #55);
# - fixed returns below; ``join`` (element-dependent error) and
#   ``to_struct`` / ``agg`` (shape-dependent) fall through to Unknown.
ARR_NAMESPACE_PRESERVING: frozenset[str] = frozenset({"sort", "reverse", "shift"})

ARR_NAMESPACE_ELEMENT_RETURN: frozenset[str] = frozenset(
    {
        "get",
        "first",
        "last",
        "explode",
    }
)

ARR_NAMESPACE_TO_LIST: frozenset[str] = frozenset(
    {
        "unique",
        "head",
        "tail",
        "slice",
        "to_list",
    }
)

ARR_NAMESPACE_RETURN: dict[str, DataType] = {
    "len": UInt32(),
    "n_unique": UInt32(),
    "arg_min": UInt32(),
    "arg_max": UInt32(),
    "count_matches": UInt32(),
    "contains": Boolean(),
    "any": Boolean(),
    "all": Boolean(),
}


# ``pl.col("c").cat.<method>(...)`` return types (issue #54). Probed on
# polars 1.41.2 with Categorical AND Enum receivers (identical results):
# ``get_categories`` -> String, ``len_bytes`` / ``len_chars`` -> UInt32,
# ``starts_with`` / ``ends_with`` -> Boolean, ``slice`` -> String.
# ``get_categories`` is length-changing (one row per category) and its
# output has no nulls even for a nullable receiver (probed) — the
# dispatcher skips the receiver-nullability wrap for it.
CAT_NAMESPACE_RETURN: dict[str, DataType] = {
    "get_categories": Utf8(),
    "len_bytes": UInt32(),
    "len_chars": UInt32(),
    "starts_with": Boolean(),
    "ends_with": Boolean(),
    "slice": Utf8(),
}

# ``pl.col("b").bin.<method>(...)`` return types (issue #51). Probed on
# polars 1.41.2: ``encode("hex"/"base64")`` -> String, ``decode`` ->
# Binary, ``size()`` -> UInt32, the predicates -> Boolean. Unlisted
# methods (e.g. ``reinterpret``, whose dtype is argument-dependent) fall
# through to Unknown as usual.
BIN_NAMESPACE_RETURN: dict[str, DataType] = {
    "encode": Utf8(),
    "decode": Binary(),
    "size": UInt32(),
    "contains": Boolean(),
    "starts_with": Boolean(),
    "ends_with": Boolean(),
}


def canonicalize_method(name: str) -> str:
    """Return the canonical form of a polars method name. The dispatch
    layer in analyzer.py calls this before looking up methods in any of
    the classification frozensets so renames are absorbed in one place."""
    return METHOD_ALIASES.get(name, name)


# Frame methods whose return shape is identical to the receiver. Includes
# both lazy and eager identity-shape methods (eager/lazy validation is
# handled separately via EAGER_ONLY_METHODS / LAZY_ONLY_METHODS).
IDENTITY_FRAME_METHODS: frozenset[str] = frozenset(
    {
        "sort",
        "head",
        "tail",
        "limit",
        "slice",
        "gather_every",
        "reverse",
        "sample",
        "unique",
        "clone",
        "set_sorted",
        "shrink_to_fit",
        "rechunk",
        "cache",
        "first",
        "last",
        "inspect",
        "top_k",
        "bottom_k",
        "sink_csv",
        "sink_parquet",
        "sink_ipc",
        "sink_ndjson",
        "sink_batches",
    }
)


# Methods that exist only on LazyFrame. Calling them on a DataFrame
# raises AttributeError at runtime — statically we surface PLY031.
LAZY_ONLY_METHODS: frozenset[str] = frozenset(
    {
        "collect",
        "collect_async",
        "collect_batches",
        "cache",
        "inspect",
        "explain",
        "show_graph",
        "profile",
        "fetch",
        "with_context",
        "sink_csv",
        "sink_parquet",
        "sink_ipc",
        "sink_ndjson",
        "sink_batches",
    }
)


# Methods that exist only on DataFrame. Calling them on a LazyFrame
# triggers PLY030 with a ``.collect()`` hint.
EAGER_ONLY_METHODS: frozenset[str] = frozenset(
    {
        "to_pandas",
        "to_numpy",
        "to_arrow",
        "to_dict",
        "to_dicts",
        "to_struct",
        "to_init_repr",
        "to_jax",
        "to_torch",
        "to_series",
        "to_dummies",
        "write_csv",
        "write_parquet",
        "write_ipc",
        "write_ipc_stream",
        "write_json",
        "write_ndjson",
        "write_avro",
        "write_excel",
        "write_database",
        "write_delta",
        "write_iceberg",
        "write_clipboard",
        "get_column",
        "get_column_index",
        "get_columns",
        "iter_columns",
        "iter_rows",
        "iter_slices",
        "row",
        "rows",
        "rows_by_key",
        "item",
        "n_chunks",
        "estimated_size",
        "shape",
        "height",
        "flags",
        "glimpse",
        "describe",
        "transpose",
        "partition_by",
        "n_unique",
    }
)


# Frame methods probed to RETURN a frame (DataFrame or LazyFrame), one set
# per receiver class. Consumed by the analyzer's frame-method dispatch
# fall-through (backlog N-3): an unmodeled method in the receiver's set
# means schema tracking silently dies — PLW007. Names NOT in the set stay
# silent: terminal methods (``to_dicts``, ``write_*``, ``item``,
# ``height``, ...) legitimately return non-frames, and unknown names
# (typos, plugin namespaces) are unknowable (conservative). Modeled
# methods also appear here — they are dispatched before the fall-through,
# so the sets stay a pure probe artifact that can be regenerated
# mechanically.
#
# Probed (polars 1.41.2): enumerated the public callables of
# ``pl.DataFrame`` / ``pl.LazyFrame`` (``dir()`` minus underscore names,
# properties skipped) and kept exactly those whose
# ``inspect.signature`` return annotation is the bare ``DataFrame`` /
# ``LazyFrame`` / ``pl.DataFrame`` token. Union-typed returns
# (``sink_csv -> LazyFrame | None``, ``collect -> DataFrame |
# InProcessQuery``, ``glimpse -> str | DataFrame | None``) and generics
# (``pipe -> T``) are excluded — they are not unconditionally
# frame-returning. Spot-checked by execution: ``df.interpolate`` /
# ``df.transpose`` / ``df.sql`` / ``df.count`` / ``df.clear`` /
# ``lf.describe`` (a DataFrame!) / ``lf.fetch`` / ``lf.pipe_with_schema``
# all return the annotated frame class; ``df.glimpse`` returns str.
EAGER_FRAME_RETURNING_METHODS: frozenset[str] = frozenset(
    {
        "approx_n_unique",
        "bottom_k",
        "cast",
        "clear",
        "clone",
        "corr",
        "count",
        "describe",
        "deserialize",
        "drop",
        "drop_nans",
        "drop_nulls",
        "explode",
        "extend",
        "fill_nan",
        "fill_null",
        "filter",
        "gather",
        "gather_every",
        "head",
        "hstack",
        "insert_column",
        "interpolate",
        "join",
        "join_asof",
        "join_where",
        "lazy",
        "limit",
        "map_columns",
        "map_rows",
        "match_to_schema",
        "max",
        "mean",
        "median",
        "melt",
        "merge_sorted",
        "min",
        "null_count",
        "pivot",
        "product",
        "quantile",
        "rechunk",
        "remove",
        "rename",
        "replace_column",
        "reverse",
        "sample",
        "select",
        "select_seq",
        "set_sorted",
        "shift",
        "shrink_to_fit",
        "slice",
        "sort",
        "sql",
        "std",
        "sum",
        "tail",
        "to_dummies",
        "top_k",
        "transpose",
        "unique",
        "unnest",
        "unpivot",
        "unstack",
        "update",
        "upsample",
        "var",
        "vstack",
        "with_columns",
        "with_columns_seq",
        "with_row_count",
        "with_row_index",
    }
)

LAZY_FRAME_RETURNING_METHODS: frozenset[str] = frozenset(
    {
        "approx_n_unique",
        "bottom_k",
        "cache",
        "cast",
        "clear",
        "clone",
        "count",
        "describe",
        "deserialize",
        "drop",
        "drop_nans",
        "drop_nulls",
        "explode",
        "fetch",
        "fill_nan",
        "fill_null",
        "filter",
        "first",
        "gather",
        "gather_every",
        "head",
        "inspect",
        "interpolate",
        "join",
        "join_asof",
        "join_where",
        "last",
        "lazy",
        "limit",
        "map_batches",
        "match_to_schema",
        "max",
        "mean",
        "median",
        "melt",
        "merge_sorted",
        "min",
        "null_count",
        "pipe_with_schema",
        "pivot",
        "quantile",
        "remove",
        "rename",
        "reverse",
        "select",
        "select_seq",
        "set_sorted",
        "shift",
        "sink_iceberg",
        "slice",
        "sort",
        "sql",
        "std",
        "sum",
        "tail",
        "top_k",
        "unique",
        "unnest",
        "unpivot",
        "update",
        "var",
        "with_columns",
        "with_columns_seq",
        "with_context",
        "with_row_count",
        "with_row_index",
    }
)


def agg_function_for(name: str):
    """Look up the ``AggFunction`` enum value for a polars aggregation method
    name. Returns ``None`` if the name isn't a known aggregation."""
    return _AGG_NAME_MAP_CACHE().get(name)


_AGG_NAME_MAP_INSTANCE: dict[str, AggFunction] | None = None


def _AGG_NAME_MAP_CACHE() -> dict[str, AggFunction]:  # noqa: N802
    global _AGG_NAME_MAP_INSTANCE
    if _AGG_NAME_MAP_INSTANCE is None:
        _AGG_NAME_MAP_INSTANCE = _build_agg_name_map()
    return _AGG_NAME_MAP_INSTANCE
