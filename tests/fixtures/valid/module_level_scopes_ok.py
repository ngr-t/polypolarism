"""Out-of-function scopes are analyzed soundly (issue #110, valid twin).

Correct column references on a statically-known frame pass at module level,
in a frame-untyped ``def main()`` body, and in an ``if __name__`` guard — and,
crucially, the new coverage stays SILENT whenever the receiver schema is NOT
provably known-and-closed:

* an OPEN frame (non-strict schema) admits extra runtime columns, so a
  "missing" reference is not a provable static error (it is a gradual-typing
  leniency, not a claim the code is runtime-correct);
* a local that can't be pinned to a known frame has no static type, so
  nothing is checked.

The open-frame / unknown-receiver demonstrations live inside ``checks`` (a
frame-untyped function that is defined but never called) so that importing
this module — which the runtime-differential harness does — cannot raise. Only
runtime-safe statements run at module top level.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = True


class Loose(pa.DataFrameModel):
    """Non-strict: the call result is an OPEN frame."""

    k: str


@pa.check_types
def get_kv() -> DataFrame[KV]:
    return pl.DataFrame({"k": ["a"], "v": [1.0]})


@pa.check_types
def get_loose() -> DataFrame[Loose]:
    return pl.DataFrame({"k": ["a"]})


# Module top level: valid references on a known closed frame (runtime-safe).
src = get_kv()
fine_module = src.select(pl.col("k"), pl.col("v"))


def checks() -> None:
    """Soundness boundary cases — analyzed by the issue #110 pass, all silent.

    Defined but never called, so the operations that would raise at runtime
    (open-frame extra, unknown receiver) are never executed.
    """
    df = get_kv()
    fine_closed = df.select(pl.col("k"))

    # OPEN frame: 'maybe_extra' may exist among the non-strict schema's runtime
    # extras, so referencing it must NOT be flagged.
    loose = get_loose()
    open_extra = loose.select(pl.col("maybe_extra"))

    # Unpinnable local: the source function is undefined in this module, so the
    # variable has no known schema — stay silent.
    mystery = some_undefined_loader()  # noqa: F821
    mystery_extra = mystery.select(pl.col("whatever"))


if __name__ == "__main__":
    fine_guard = src.select(pl.col("k"))
