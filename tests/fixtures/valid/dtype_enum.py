"""Valid fixture: ``pl.Enum`` (polars 1.25+ stabilized) as a Pandera schema dtype.

Exercises landmark version 1.25 — ``Enum`` becoming a stable dtype
distinct from ``Categorical``. The analyzer treats every ``pl.Enum`` as
structurally equal for now; the variant list is not yet inspected (a
future enhancement when we model `Enum(["a", "b"])` calls explicitly).
``priority`` keeps the bare class form; ``status`` carries categories
because a *cast* target must — ``cast(pl.Enum)`` without categories is
``Enum([])`` at runtime and raises ``InvalidOperationError`` for every
non-null value (probed on polars 1.41.2).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class OrderSchema(pa.DataFrameModel):
    order_id: int
    status: pl.Enum(["new", "paid"])
    priority: pl.Enum


def passthrough(df: DataFrame[OrderSchema]) -> DataFrame[OrderSchema]:
    return df.filter(pl.col("order_id") > 0)


def cast_status_to_enum(df: DataFrame[OrderSchema]) -> DataFrame[OrderSchema]:
    return df.with_columns(pl.col("status").cast(pl.Enum(["new", "paid"])))
