"""Stable diagnostic codes for analyzer errors.

The codes are intended for IDE / CI consumption — never repurpose a code,
only add new ones. Format: ``[PLY###] message``.
"""

from __future__ import annotations

import re

# Expression / column lookup
PLY001 = "PLY001"  # Column not found in expression (pl.col / cs.*)

# Frame reshape — column reference errors
PLY002 = "PLY002"  # drop: column not found
PLY003 = "PLY003"  # rename: source column not found
PLY004 = "PLY004"  # cast: column not found
PLY005 = "PLY005"  # drop_nulls: subset column not found
PLY006 = "PLY006"  # with_row_index: name collides with existing column
PLY007 = "PLY007"  # sort: column not found
PLY008 = "PLY008"  # filter predicate / when condition dtype is not Boolean
PLY009 = "PLY009"  # binary operation between incompatible dtypes (arithmetic, comparison, is_in)
PLY013 = "PLY013"  # cast between structurally incompatible dtypes
PLY014 = "PLY014"  # unique: subset column not found
PLY015 = "PLY015"  # duplicate output column name in select/with_columns
PLY016 = "PLY016"  # numeric-only operation applied to a non-numeric column
PLY017 = "PLY017"  # list literal mixed with other positional expression arguments

# Join
PLY010 = "PLY010"  # join key error (missing or dtype mismatch)

# GroupBy / aggregation
PLY011 = "PLY011"  # group_by key missing or aggregation type error

# Expression namespaces
PLY012 = "PLY012"  # namespace accessor (.str/.dt/.list/.arr/.struct/.bin/.cat) on a wrong dtype

# Concat / explode / unpivot
PLY020 = "PLY020"  # concat schema mismatch / horizontal overlap
PLY021 = "PLY021"  # explode: column missing or not List/Array
PLY022 = "PLY022"  # unpivot: column missing / value dtype unification failure

# Eager / lazy distinction
PLY030 = "PLY030"  # eager-only method called on a LazyFrame (suggest .collect())
PLY031 = "PLY031"  # lazy-only method called on a DataFrame (suggest .lazy())
PLY032 = "PLY032"  # function expected DataFrame[S] but got LazyFrame[S] (or vice versa)
PLY033 = (
    "PLY033"  # variable annotation re-interprets the inferred frame as an unrelated type (ADR-0005)
)

# Declared vs inferred return type comparison (checker.py). One shared code
# for the whole family (issue #70): missing column, extra column, dtype
# difference, and could-not-infer all describe the same declared-return-type
# check; the message distinguishes the kind.
PLY040 = "PLY040"  # declared return type does not match the inferred return type

# Schema definition
PLY041 = "PLY041"  # schema field annotation provably crashes pandera at runtime (Annotated arity, issue #69)

# Declared-schema interface ("checked island", issue #83)
PLY042 = "PLY042"  # column not declared in the function's (non-strict) schema — an undeclared dependency, not a provable runtime failure


# Warnings (PLW###): inference is imprecise here, but the user can usually
# fix it by adding a type annotation or an explicit dtype argument.
PLW001 = "PLW001"  # map_elements / map_batches without ``return_dtype=`` keyword
PLW002 = "PLW002"  # ``df.pipe(callable)`` where the callable can't be resolved
PLW003 = "PLW003"  # function call to a name that isn't in the analysed module
PLW004 = "PLW004"  # lambda / inline callable used where its return dtype is unknowable
PLW005 = "PLW005"  # pivot / to_dummies result schema is data-dependent; user should annotate
PLW006 = "PLW006"  # DataFrame[X] / LazyFrame[X] annotation references an unknown schema
PLW007 = "PLW007"  # method not modeled (or experimental polars API); result degrades to Unknown
PLW008 = "PLW008"  # variable annotation narrows the inferred RHS without runtime backing (ADR-0005)

# Environment / version
PLW010 = "PLW010"  # detected polars or pandera version below polypolarism's supported floor

PLW011 = "PLW011"  # schema field annotation unrecognized; column degrades to Unknown dtype (#77)


def tag(code: str, message: str) -> str:
    """Return ``"[CODE] message"``; idempotent if message is already tagged."""
    if message.startswith(f"[{code}]"):
        return message
    return f"[{code}] {message}"


_TAGGED_MESSAGE = re.compile(r"^\[(PL[YW]\d{3})\]")


def extract_code(message: str) -> str | None:
    """Return the leading ``PLY###`` / ``PLW###`` code of a tagged message.

    ``None`` for untagged diagnostics (e.g. parse / read failures). Used by
    JSON output to expose the code structurally (issue #70) — consumers
    should not have to regex the message themselves.
    """
    match = _TAGGED_MESSAGE.match(message)
    return match.group(1) if match else None
