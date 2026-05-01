"""Polars surface tables consumed by the analyzer.

Currently centralizes the bare-attribute dtype name map. Other tables
(agg shorthand, namespace tables, method classifications, join `how`
literals, aggregation signatures) will move here per ADR-0001 steps 2-5.

Attribute references like ``pl.Int64`` resolve through ``DTYPE_NAME_MAP``;
parametrized dtypes (``pl.Datetime("us", "UTC")``, ``pl.Decimal(20, 4)``,
``pl.Datetime``) are handled via Call form in the dispatch sites.
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from polypolarism.ops.groupby import AggFunction

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


def canonicalize_method(name: str) -> str:
    """Return the canonical form of a polars method name. The dispatch
    layer in analyzer.py calls this before looking up methods in any of
    the classification frozensets so renames are absorbed in one place."""
    return METHOD_ALIASES.get(name, name)


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
