"""The pivot variable annotation must actually flow to the return check.

False-negative twin of ``valid/m12_pivot_annotated.py``. The pivot output
is data-dependent, so the ``DataFrame[WidePivoted]`` variable annotation
itself is *trusted* (pplw-data-dependent-schema documents why) and cannot contradict the pivot
statically. What IS checkable is that the annotated type flows downstream:
here the declared return schema contradicts the annotation.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    region: str
    metric: str
    value: pl.Float64


class WidePivoted(pa.DataFrameModel):
    region: str
    revenue: pl.Float64
    cost: pl.Float64


class WideOut(pa.DataFrameModel):
    region: str
    revenue: pl.Int64  # WRONG: the pivot annotation declares revenue Float64
    cost: pl.Float64


def widen(df: DataFrame[In]) -> DataFrame[WideOut]:
    result: DataFrame[WidePivoted] = df.pivot(on="metric", index=["region"], values="value")
    return result
