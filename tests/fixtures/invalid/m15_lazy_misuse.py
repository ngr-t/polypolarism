"""M15 invalid: write_csv on LazyFrame triggers PLY030."""

import pandera.polars as pa
from pandera.typing.polars import LazyFrame


class S(pa.DataFrameModel):
    id: int


def f(lf: LazyFrame[S]):
    return lf.write_csv("out.csv")
