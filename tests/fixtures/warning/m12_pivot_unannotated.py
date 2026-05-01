"""pivot without an annotated assignment emits PLW005."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    region: str
    metric: str
    value: pl.Float64


def widen(df: DataFrame[In]):
    # No annotation, no inference — the user is nudged toward declaring the
    # output shape with a Pandera schema.
    return df.pivot(on="metric", index=["region"], values="value")
