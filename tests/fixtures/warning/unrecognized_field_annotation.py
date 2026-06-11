"""Warning fixture: an unrecognized field annotation degrades loudly (PLW011).

Issue #77: a field annotation polypolarism cannot translate used to silently
DROP the field from the schema — phantom "extra column" FPs on strict
schemas, vanished-column FNs on open ones. The column now registers with
Unknown dtype (it EXISTS, with an unknowable dtype) and every function
referencing the schema gets a PLW011 warning instead.

Why a warning and not a PLY041 error: the bare name may be a runtime alias
of a real dtype (``MyAlias = pl.Int64`` — pandera resolves the annotation
fine), so unresolvability is not provable statically. If the name genuinely
does not resolve, pandera raises TypeError the first time the schema is used
(probed, 0.31.1) — which is what the warning points at.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class CustomPayload:
    """Not a dtype — pandera would raise TypeError at first schema use."""


class Src(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


class WithMystery(pa.DataFrameModel):
    a: int
    blob: CustomPayload  # unrecognized -> Unknown dtype column + PLW011

    class Config:
        strict = True


def keeps_unknown_column(df: DataFrame[Src]) -> DataFrame[WithMystery]:
    # Before #77 the strict declaration lost 'blob' entirely, so this
    # correct-shape body was rejected with "Extra column 'blob'". Now the
    # declared slot exists (Unknown dtype, accepts anything) and the only
    # diagnostic is the PLW011 advisory on the schema definition.
    return df.with_columns(blob=pl.lit(1))
