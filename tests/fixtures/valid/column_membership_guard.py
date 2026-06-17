"""Column-membership guards narrow the frame schema (issue #109).

A runtime ``if "a" in df.columns:`` test proves column ``a`` is present in the
matching branch, so referencing / returning it there is no longer a false
positive. Presence is learned at ``Unknown`` dtype (gradual): enough to
satisfy an open / ``coerce=True`` consumer, NOT enough to satisfy a strict,
concretely-dtyped column under ``coerce=False`` (see the invalid twin
``column_membership_guard_no_coerce.py``).

The guarded columns (``a`` / ``b``) are deliberately ABSENT from the input
schema ``KVopen`` — that is what makes the guard *do* something: without the
narrowing each function would fail with a missing-column error. Because the
column is absent from the annotated input, the runtime-differential harness
cannot synthesize an input that reaches the guarded branch (it synthesizes
from the schema, which has no ``a``); the fixture is therefore SKIPped at
runtime for the same reason as the validate-narrowing fixtures — "body
requires columns beyond the annotated input schema".

Scope (per the issue): presence is learned, dtype is NOT — a guard on a
column the schema already declares ``Optional`` is not promoted to required
here; only genuinely-absent columns are introduced (at ``Unknown``).
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class KVopen(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = False  # extra columns allowed
        coerce = True


class KVa(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True
        coerce = True


class KVab(pa.DataFrameModel):
    k: str
    v: float
    a: float
    b: float

    class Config:
        strict = True
        coerce = True


@pa.check_types
def positive_guard(df: DataFrame[KVopen]) -> DataFrame[KVa]:
    # True branch: 'a' is provably present (Unknown dtype); coerce=True on the
    # consumer satisfies the declared Float64.
    if "a" in df.columns:
        return df
    raise ValueError("missing a")


@pa.check_types
def negated_early_raise(df: DataFrame[KVopen]) -> DataFrame[KVa]:
    # not-in branch exits, so the fall-through flow knows 'a' is present.
    if "a" not in df.columns:
        raise ValueError("missing a")
    return df


@pa.check_types
def set_subset_guard(df: DataFrame[KVopen]) -> DataFrame[KVab]:
    # {"a", "b"} <= set(df.columns): both columns are present in the branch.
    if {"a", "b"} <= set(df.columns):
        return df
    raise ValueError("missing a/b")


@pa.check_types
def issubset_guard(df: DataFrame[KVopen]) -> DataFrame[KVab]:
    if {"a", "b"}.issubset(df.columns):
        return df
    raise ValueError("missing a/b")


@pa.check_types
def schema_membership_guard(df: DataFrame[KVopen]) -> DataFrame[KVa]:
    if "a" in df.schema:
        return df
    raise ValueError("missing a")
