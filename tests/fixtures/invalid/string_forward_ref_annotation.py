"""Invalid test case: string (forward-ref) annotations are now analyzed (C-13).

``def f(df: "DataFrame[S]") -> "DataFrame[S]":`` carries its annotations as
``ast.Constant(str)`` nodes rather than the ``ast.Subscript`` the detectors
expect. Before C-13 the function was silently skipped ("no functions found");
now the quoted annotations are parsed and the body is checked like the bare
form — so this missing-column reference is caught.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    user_id: int
    name: str


def take_users(df: "DataFrame[InSchema]") -> "DataFrame[InSchema]":
    """ERROR: 'missing_col' is not a column of InSchema."""
    return df.select(pl.col("user_id"), pl.col("missing_col"))
