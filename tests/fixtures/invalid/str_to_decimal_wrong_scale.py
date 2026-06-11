"""Invalid: ``str.to_decimal(scale=2)`` declared with the wrong scale (issue #61).

False-negative twin of ``valid/m3_str_namespace.py``: ``to_decimal`` yields
``Decimal(38, scale)`` from its required keyword argument — before #61 the
inferred dtype was a fixed ``Decimal(38, 0)``, so this wrong declaration
passed statically while pandera rejects the runtime ``Decimal(38, 2)``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Raw(pa.DataFrameModel):
    price: str


class ParsedWrong(pa.DataFrameModel):
    price: pl.Decimal(38, 0)  # WRONG: to_decimal(scale=2) yields Decimal(38, 2)


def parse(df: DataFrame[Raw]) -> DataFrame[ParsedWrong]:
    return df.select(pl.col("price").str.to_decimal(scale=2).alias("price"))
