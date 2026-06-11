"""A variable annotation feeds the chain; the return declaration is wrong.

False-negative twin of ``valid/variable_annotation_basic.py`` and
``valid/variable_annotation_chain.py``.

Note (discovered false negative, 2026-06): the annotation itself is
TRUSTED — ``visit_AnnAssign`` lets a ``DataFrame[Schema]`` annotation win
unconditionally, so an annotation contradicting an *inferable* assigned
expression passes silently (the RHS inference is discarded). The wrong
declaration here therefore sits downstream, on the return schema; the
contradiction case stays in the README "Known gaps" until the checker
compares the RHS against the annotation.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    id: int
    value: int


class OutSchema(pa.DataFrameModel):
    id: int
    doubled: str  # WRONG: 'value' is Int64 via the annotation, so doubled is Int64


def process() -> DataFrame[OutSchema]:
    """The chain's input type comes only from the variable annotation."""
    df: DataFrame[InSchema] = get_external_data()
    result = df.select(
        pl.col("id"),
        (pl.col("value") * 2).alias("doubled"),
    )
    return result
