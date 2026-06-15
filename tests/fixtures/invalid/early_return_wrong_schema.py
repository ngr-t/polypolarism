"""All return paths are checked, not just the last one (issue #95).

The early branch returns a frame missing column 'a' while the final
return is correct.  polypolarism must catch the early wrong path.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = True


class KVa(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True
        coerce = True


@pa.check_types
def wrong_early_return(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
    if flag:
        return df.with_columns(b=pl.col("v"))  # wrong: no 'a', has 'b'
    return df.with_columns(a=pl.col("v") * 2)
