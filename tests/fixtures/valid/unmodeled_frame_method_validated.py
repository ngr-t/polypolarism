"""Valid: the two silent paths around the frame-level PLW007 (backlog N-3).

Remedy twin of ``warning/unmodeled_frame_method``:

- ``smooth_validated`` — wrapping the unmodeled call in
  ``Schema.validate`` retypes the result to the schema, exactly the
  repair the warning recommends, so the warning is retracted (the
  frame-level analog of ``valid/unmodeled_method_pinned``'s cast).
- ``export`` — ``to_dicts()`` returns ``list[dict]``: schema tracking
  past a terminal method is meaningless, so methods probed NOT to
  return a frame never warn (nor do unknown names: typos, plugin
  namespaces).
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class Smoothed(pa.DataFrameModel):
    v: float = pa.Field(nullable=True)


def smooth_validated(df: DataFrame[Smoothed]) -> DataFrame[Smoothed]:
    return Smoothed.validate(df.interpolate())


def export(df: DataFrame[Smoothed]) -> list[dict]:
    return df.to_dicts()
