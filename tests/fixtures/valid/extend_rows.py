"""Valid: DataFrame.extend appends rows, schema-preserving (issue #133).

``extend`` is a memory-level vertical concat (like ``vstack``): it appends the
other frame's rows and keeps the schema. It must route through the vertical
handler, not the horizontal-concat one (which rejected the shared columns as
duplicates).
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class Wide(pa.DataFrameModel):
    id: str
    q1: int


def extend_same_schema(df: DataFrame[Wide]) -> DataFrame[Wide]:
    return df.clone().extend(df)
