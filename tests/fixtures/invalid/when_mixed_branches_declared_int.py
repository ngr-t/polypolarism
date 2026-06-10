"""when/then/otherwise with mixed branches declared as int (issue #40).

Probed (polars 1.41.2):
``pl.when(pl.col("a") > 0).then(pl.lit(1)).otherwise(pl.lit("x"))``
produces ``Schema({'literal': String})`` — the declared ``x: int`` (with
coerce=False) used to pass silently while polypolarism registered the
column as Unknown; it must now fail with a TypeDifference.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int


class Out(pa.DataFrameModel):
    a: int
    x: int


@pa.check_types
def bug_mixed_branches_declared_int(df: DataFrame[In]) -> DataFrame[Out]:
    return df.with_columns(x=pl.when(pl.col("a") > 0).then(pl.lit(1)).otherwise(pl.lit("x")))
