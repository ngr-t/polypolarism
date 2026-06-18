"""`pplw-unmodeled-method`: a method polypolarism does not model degrades the column
dtype to Unknown — the warning makes the silent degradation visible.

Probed (polars 1.41.2): ``peak_max`` returns Boolean; polypolarism does
not model it, so ``flag`` passes only via Unknown-leniency (the ``via:``
note in the golden pins that). The repair pplw-unmodeled-method recommends is shown by
``valid/unmodeled_method_pinned`` (cast right after the call — no
warning).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    v: int


class Out(pa.DataFrameModel):
    v: int
    flag: bool


def peaks(df: DataFrame[In]) -> DataFrame[Out]:
    return df.with_columns(flag=pl.col("v").peak_max())
