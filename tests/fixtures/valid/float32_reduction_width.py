"""Float32 receivers keep Float32 through float-family reductions (N-2).

Probed (polars 1.41.2): ``rolling_mean``/``rolling_std``/``rolling_var``/
``rolling_median``/``rolling_quantile`` on a Float32 column return
**Float32**, and so do ``mean``/``std``/``var``/``median``/``quantile`` in
``select`` and ``group_by().agg()`` contexts. Each declaration below pins
the preserved width; the false-negative twin is
``invalid/float32_reduction_declared_float64``.
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
    reading_rolling_mean: pl.Float32 = pa.Field(nullable=True)

    class Config:
        strict = True


def rolling_mean_keeps_float32(df: DataFrame[Sensor]) -> DataFrame[Rolled]:
    return df.with_columns(
        pl.col("reading").rolling_mean(window_size=3).alias("reading_rolling_mean"),
    )


class PerSensor(pa.DataFrameModel):
    sensor: str
    avg_reading: pl.Float32
    med_reading: pl.Float32

    class Config:
        strict = True


def agg_keeps_float32(df: DataFrame[Sensor]) -> DataFrame[PerSensor]:
    return df.group_by("sensor").agg(
        pl.col("reading").mean().alias("avg_reading"),
        pl.col("reading").median().alias("med_reading"),
    )


class Spread(pa.DataFrameModel):
    spread: pl.Float32

    class Config:
        strict = True


def select_std_ddof0_keeps_float32(df: DataFrame[Sensor]) -> DataFrame[Spread]:
    return df.select(pl.col("reading").std(ddof=0).alias("spread"))
