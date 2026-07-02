"""Warning: rename(callable) loud-degrades instead of silent identity (issue #131).

``df.rename(lambda c: ...)`` (the documented callable form) cannot be evaluated
statically. Previously it was silently treated as identity — the frame kept its
OLD names, and a correct rename into a new schema was rejected with phantom
missing/extra-column errors. Now it loud-degrades: the variable untracks and a
``pplw-unmodeled-method`` warning fires (same convention as an unmodeled frame
method), so no phantom errors are manufactured. With a declared frame return the
loss surfaces instead as a "could not infer return type" error — the accepted
loud-degrade tradeoff.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class Wide(pa.DataFrameModel):
    id: str
    q1: int


def rename_callable(df: DataFrame[Wide]):
    return df.rename(lambda c: c + "_x")
