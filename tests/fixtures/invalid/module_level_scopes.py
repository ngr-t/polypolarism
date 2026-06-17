"""Statements outside a frame-typed function signature are analyzed when the
receiver schema is statically known (issue #110).

The same provable missing-column reference that fires PLY001 inside a typed
function must also fire at module top level, inside a frame-untyped
``def main() -> None:`` body, and inside an ``if __name__`` guard — wherever
the receiver frame's schema is statically known (here, the closed return of
``get_kv() -> DataFrame[KV]``).
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


@pa.check_types
def get_kv() -> DataFrame[KV]:
    return pl.DataFrame({"k": ["a"], "v": [1.0]})


# (1) module top level: src has KV's closed schema, so 'nope_module' is a
# provable missing-column reference.
src = get_kv()
boom = src.select(pl.col("nope_module"))


# (2) frame-untyped function: df is seeded from get_kv()'s closed return.
def main() -> None:
    df = get_kv()
    boom2 = df.select(pl.col("nope_main"))


# (3) __main__ guard: src (closed KV) is in scope from the module top level.
if __name__ == "__main__":
    boom3 = src.select(pl.col("nope_guard"))
