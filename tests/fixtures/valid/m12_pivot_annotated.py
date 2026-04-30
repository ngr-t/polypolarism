"""M12: pivot() output schema is data-dependent — annotation wins."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    region: str
    metric: str
    value: pl.Float64


class WideOut(pa.DataFrameModel):
    region: str
    revenue: pl.Float64
    cost: pl.Float64


def widen(df: DataFrame[In]) -> DataFrame[WideOut]:
    # The pivot result is data-dependent so polypolarism falls back to the
    # variable annotation. PLW005 is emitted as a guidance warning.
    result: DataFrame[WideOut] = df.pivot(on="metric", index=["region"], values="value")
    return result
