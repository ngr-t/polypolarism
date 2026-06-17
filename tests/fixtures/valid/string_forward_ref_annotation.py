"""Valid test case: string (forward-ref) annotations resolve cleanly (C-13).

Quoted ``DataFrame[Schema]`` annotations on both the parameter and the return
are parsed (``ast.parse(..., mode="eval")``) and feed the normal detection
path, so a correct body type-checks exactly like the unquoted form.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    user_id: int
    name: str


def take_users(df: "DataFrame[InSchema]") -> "DataFrame[InSchema]":
    return df.select(pl.col("user_id"), pl.col("name"))
