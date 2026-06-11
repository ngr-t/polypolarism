"""False-negative twin of ``valid/upsample_nullable_gaps``.

upsample's inserted gap rows null-fill every non-key column (probed,
polars 1.37.0-1.41.2), so declaring a non-key column non-nullable must
fail — exactly like the left-join nullability rule.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Readings(pa.DataFrameModel):
    t: pl.Datetime
    g: str
    v: int


class UpsampledWrong(pa.DataFrameModel):
    t: pl.Datetime
    g: str = pa.Field(nullable=True)
    v: int  # WRONG: gap rows null-fill non-key columns -> Nullable[Int64]

    class Config:
        strict = True


def half_hourly(data: DataFrame[Readings]) -> DataFrame[UpsampledWrong]:
    return data.sort("t").upsample(time_column="t", every="30m")
