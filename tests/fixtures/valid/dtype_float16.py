"""Valid fixture: ``pl.Float16`` (polars 1.36+) as a Pandera schema dtype.

Exercises landmark version 1.36 — half-precision float. Symmetric with
``Float32`` / ``Float64`` but cannot be confused with them at the type
level.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class FeaturesSchema(pa.DataFrameModel):
    feat_id: int
    embedding_dim_0: pl.Float16
    embedding_dim_1: pl.Float16


def passthrough(df: DataFrame[FeaturesSchema]) -> DataFrame[FeaturesSchema]:
    return df.filter(pl.col("feat_id") > 0)


def cast_to_float16(df: DataFrame[FeaturesSchema]) -> DataFrame[FeaturesSchema]:
    return df.with_columns(pl.col("embedding_dim_0").cast(pl.Float16))
