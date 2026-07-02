"""Warning: positional selectors degrade loudly in select (issue #142).

``pl.nth(i)`` / ``pl.first()`` / ``pl.last()`` select a column by POSITION — the
output name isn't inferable when column order isn't pinned. This used to
hard-fail ``Could not infer return type`` with no diagnostic; it now degrades
loudly (``pplw-unmodeled-method`` + the result frame opens), so a correct
declaration is no longer a bare hard error.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class TwoInts(pa.DataFrameModel):
    a: int
    b: int


class Open(pa.DataFrameModel):  # non-strict — admits the opaque positional column
    pass


def positional_select(df: DataFrame[TwoInts]) -> DataFrame[Open]:
    return df.select(pl.nth(0))
