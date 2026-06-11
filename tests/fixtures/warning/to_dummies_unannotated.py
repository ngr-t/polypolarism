"""to_dummies without an annotated assignment emits PLW005 (issue #74).

Like pivot, the output columns are value-dependent (``color`` ->
``color_red``, ``color_blue``, ... UInt8; probed on polars 1.41.2,
identical on 1.37.0), so instead of silently failing inference the user
is nudged toward declaring the output shape with a Pandera schema.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    color: str
    qty: int


def widen(df: DataFrame[In]):
    # No annotation, no inference — the dummied column names exist only
    # at runtime.
    return df.to_dummies("color")
