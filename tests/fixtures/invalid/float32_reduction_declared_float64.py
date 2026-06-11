"""Invalid: Float32 reductions declared as Float64 (wrong width; N-2).

False-negative twin of ``valid/float32_reduction_width.py``: polars keeps
the Float32 width through ``rolling_mean`` / ``mean`` / ``median`` /
``std`` on a Float32 column (probed 1.41.2), so a Float64 declaration
mismatches — pandera rejects the Float32 data against a Float64 field at
runtime (no coerce). Each function below declares the old, wrongly
widened dtype.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Sensor(pa.DataFrameModel):
    sensor: str
    reading: pl.Float32


class Rolled(pa.DataFrameModel):
    sensor: str
    reading: pl.Float32
    # WRONG: rolling_mean(Float32) -> Float32, not Float64
    reading_rolling_mean: pl.Float64 = pa.Field(nullable=True)

    class Config:
        strict = True


def rolling_mean_declared_float64(df: DataFrame[Sensor]) -> DataFrame[Rolled]:
    return df.with_columns(
        pl.col("reading").rolling_mean(window_size=3).alias("reading_rolling_mean"),
    )


class PerSensor(pa.DataFrameModel):
    sensor: str
    avg_reading: pl.Float64  # WRONG: mean(Float32) -> Float32, not Float64
    med_reading: pl.Float64  # WRONG: median(Float32) -> Float32, not Float64

    class Config:
        strict = True


def agg_declared_float64(df: DataFrame[Sensor]) -> DataFrame[PerSensor]:
    return df.group_by("sensor").agg(
        pl.col("reading").mean().alias("avg_reading"),
        pl.col("reading").median().alias("med_reading"),
    )


class Spread(pa.DataFrameModel):
    spread: pl.Float64  # WRONG: std(Float32, ddof=0) -> Float32, not Float64

    class Config:
        strict = True


def select_std_ddof0_declared_float64(df: DataFrame[Sensor]) -> DataFrame[Spread]:
    return df.select(pl.col("reading").std(ddof=0).alias("spread"))
