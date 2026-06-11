"""Invalid: scalar declaration for a list-aggregated UDF result (issue #86).

``agg(x=pl.col("v").map_elements(f, return_dtype=pl.Float64))`` is
implicitly list-aggregated in grouped context — the runtime schema is
``List(Float64)`` (probed on polars 1.41.2 and 1.37.0) — so the scalar
``Float64`` declaration is rejected. Previously this was a false negative
that only ``@pa.check_types`` caught at runtime.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


def double(v: float) -> float:
    return v * 2.0


class Src(pa.DataFrameModel):
    g: str
    v: int


class ScalarOut(pa.DataFrameModel):
    g: str
    x: pl.Float64

    class Config:
        strict = True


@pa.check_types
def wrong_scalar_declaration(df: DataFrame[Src]) -> DataFrame[ScalarOut]:
    return df.group_by("g").agg(x=pl.col("v").map_elements(double, return_dtype=pl.Float64))
