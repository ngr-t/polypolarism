"""A presence-only guard does NOT satisfy a strict dtyped column without
coerce (issue #109).

``if "a" in df.columns:`` proves ``a`` EXISTS but learns nothing about its
dtype (recorded as ``Unknown``, presence-only). When the declared return is
strict with a concrete dtype and ``coerce = False``, pandera will not cast
``a`` at validation time, so presence alone is insufficient — the dtype is
unproven and rejecting the return remains correct (the runtime guard passing
the column through does not make it a ``Float64``).

``a`` is absent from the input schema ``KVopen`` so the guard is what
introduces it (at ``Unknown``); contrast ``valid/column_membership_guard.py``,
where the same guard is accepted because the consumer is open / ``coerce =
True``. Runtime-SKIPped for the same reason as the valid twin: the guarded
branch needs a column the synthesized input cannot carry.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class KVopen(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = False
        coerce = True


class KVaStrictNoCoerce(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True
        coerce = False


@pa.check_types
def presence_only_not_enough(df: DataFrame[KVopen]) -> DataFrame[KVaStrictNoCoerce]:
    # 'a' is proven present by the guard (Unknown dtype), but coerce=False
    # won't cast it to Float64 — the strict dtyped slot stays unsatisfied.
    if "a" in df.columns:
        return df
    raise ValueError("missing a")
