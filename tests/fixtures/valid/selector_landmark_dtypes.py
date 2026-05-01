"""Valid fixture: ``cs.integer`` / ``cs.float`` / ``cs.numeric`` pick up
the polars 1.18+/1.34+/1.36+ landmark dtypes.

If a future change drops Int128 / UInt128 / Float16 from the selector
classifier tuples, this fixture's ``select`` will under-match and the
declared schemas will fail to type-check — surfacing the regression.
"""

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    name: str
    a: pl.Int128
    b: pl.UInt128
    c: pl.Float16


def pick_integers(df: DataFrame[S]):
    # Should match a (Int128) and b (UInt128); omit c (Float16) and name.
    return df.select(cs.integer())


def pick_floats(df: DataFrame[S]):
    # Should match c (Float16).
    return df.select(cs.float())


def pick_numerics(df: DataFrame[S]):
    # Should match a (Int128), b (UInt128), c (Float16).
    return df.select(cs.numeric())
