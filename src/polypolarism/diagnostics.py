"""Stable diagnostic codes for analyzer errors.

The codes are intended for IDE / CI consumption — never repurpose a code,
only add new ones. Format: ``[PLY###] message``.
"""

from __future__ import annotations

# Expression / column lookup
PLY001 = "PLY001"  # Column not found in expression (pl.col / cs.*)

# Frame reshape — column reference errors
PLY002 = "PLY002"  # drop: column not found
PLY003 = "PLY003"  # rename: source column not found
PLY004 = "PLY004"  # cast: column not found
PLY005 = "PLY005"  # drop_nulls: subset column not found
PLY006 = "PLY006"  # with_row_index: name collides with existing column

# Join
PLY010 = "PLY010"  # join key error (missing or dtype mismatch)

# GroupBy / aggregation
PLY011 = "PLY011"  # group_by key missing or aggregation type error

# Concat / explode / unpivot
PLY020 = "PLY020"  # concat schema mismatch / horizontal overlap
PLY021 = "PLY021"  # explode: column missing or not List[T]
PLY022 = "PLY022"  # unpivot: column missing / value dtype unification failure

# Eager / lazy distinction
PLY030 = "PLY030"  # eager-only method called on a LazyFrame (suggest .collect())
PLY031 = "PLY031"  # lazy-only method called on a DataFrame (suggest .lazy())
PLY032 = "PLY032"  # function expected DataFrame[S] but got LazyFrame[S] (or vice versa)


# Warnings (PLW###): inference is imprecise here, but the user can usually
# fix it by adding a type annotation or an explicit dtype argument.
PLW001 = "PLW001"  # map_elements / map_batches without ``return_dtype=`` keyword
PLW002 = "PLW002"  # ``df.pipe(callable)`` where the callable can't be resolved
PLW003 = "PLW003"  # function call to a name that isn't in the analysed module
PLW004 = "PLW004"  # lambda / inline callable used where its return dtype is unknowable
PLW005 = "PLW005"  # pivot result schema is data-dependent; user should annotate
PLW006 = "PLW006"  # DataFrame[X] / LazyFrame[X] annotation references an unknown schema

# Environment / version
PLW010 = "PLW010"  # detected polars or pandera version below polypolarism's supported floor


def tag(code: str, message: str) -> str:
    """Return ``"[CODE] message"``; idempotent if message is already tagged."""
    if message.startswith(f"[{code}]"):
        return message
    return f"[{code}] {message}"
