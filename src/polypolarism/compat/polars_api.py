"""Polars surface tables consumed by the analyzer.

Single source of truth for the polars 1.x surface that polypolarism
recognizes:

- ``DTYPE_NAME_MAP`` — bare-attribute dtype names (``pl.Int64``,
  ``pl.String``, ``pl.Decimal`` …). Parametrized dtypes
  (``pl.Datetime("us", "UTC")``, ``pl.Decimal(20, 4)``) are matched
  by Call form in dispatch sites; ``DECIMAL_DEFAULT`` is the bare
  ``pl.Decimal`` fallback.
- ``IDENTITY_FRAME_METHODS`` / ``LAZY_ONLY_METHODS`` /
  ``EAGER_ONLY_METHODS`` — frame method classification.
- ``STR_NAMESPACE_RETURN`` / ``DT_NAMESPACE_RETURN`` /
  ``DT_NAMESPACE_PRESERVING`` / ``LIST_NAMESPACE_PRESERVING`` /
  ``LIST_NAMESPACE_ELEMENT_RETURN`` — sub-namespace return tables.
- ``JOIN_HOW_VALUES`` / ``JOIN_HOW_INFERRED`` and the
  ``join_left_nullable`` / ``join_right_nullable`` predicates.
- ``agg_function_for(name)`` / ``AGG_SHORTHAND_NAMES`` — polars-side
  aggregation name lookup. (The actual signature/inference logic lives
  in ``ops/groupby.py``.)
- ``METHOD_ALIASES`` + ``canonicalize_method`` — empty rename scaffold
  for future intra-1.x renames.
- ``PolarsProfile`` / ``POLARS_1_X`` — version-conditional behavior
  scaffold (no fields yet).

See ADR-0001 (``docs/adr/0001-polars-pandera-version-support.md``) for
the policy that motivates the centralization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polypolarism.types import (
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
# ``pl.Decimal(p, s)`` is parsed at the call site to preserve p / s.
DECIMAL_DEFAULT = Decimal(38, 0)

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
    "Date": Date(),
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

# Subset of ``how`` values where the analyzer infers a column-shape result
# (vs. ``cross`` / ``semi`` / ``anti`` which have specific shape rules
# handled elsewhere). Kept as a tuple for use in ``Literal[...]`` typing.
JOIN_HOW_INFERRED: tuple[str, ...] = ("inner", "left", "right", "full")


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


# Method-name renames within polars 1.x. Empty today — the supported window
# is the latest two 1.x minors and no rename has happened across that
# window. The slot exists so that the day a polars minor renames a method,
# the fix is one entry here rather than touching every dispatch site.
#
# When a rename does ship, an entry like ``"new_name": "canonical_name"``
# means "if the user wrote new_name (or if it's the rename target —
# direction depends on which side we want as canonical), treat it as
# canonical_name during dispatch." See ADR-0001 § Decision item 3 for the
# policy.
METHOD_ALIASES: dict[str, str] = {}


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
    # Parsing into temporal types
    "to_date": Date(),
    "to_datetime": Datetime(),
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
# (de-listing operations).
LIST_NAMESPACE_ELEMENT_RETURN: frozenset[str] = frozenset(
    {
        "get",
        "first",
        "last",
        "sum",
        "mean",
        "min",
        "max",
        "median",
    }
)


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
