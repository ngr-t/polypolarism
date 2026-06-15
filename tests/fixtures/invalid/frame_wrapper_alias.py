"""DataFrame / LazyFrame alias imports must be recognised (issue #96).

When the user writes ``from pandera.typing.polars import DataFrame as DF``
and annotates the return type with ``DF[KVa]``, polypolarism must apply
the same schema checks as for the canonical ``DataFrame[KVa]`` form.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame as DF


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = False


class KVa(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True
        coerce = False


@pa.check_types
def bug_alias_return(df: DF[KV]) -> DF[KVa]:
    return df.select(pl.col("k"))
