"""Polars surface tables consumed by the analyzer.

Currently centralizes the bare-attribute dtype name map. Other tables
(agg shorthand, namespace tables, method classifications, join `how`
literals, aggregation signatures) will move here per ADR-0001 steps 2-5.

Attribute references like ``pl.Int64`` resolve through ``DTYPE_NAME_MAP``;
parametrized dtypes (``pl.Datetime("us", "UTC")``, ``pl.Decimal(20, 4)``,
``pl.Datetime``) are handled via Call form in the dispatch sites.
"""

from __future__ import annotations

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
