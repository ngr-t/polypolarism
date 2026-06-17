"""A @rowpoly helper that pattern-drops the row variable (issue C-14 Tier 4, PLY043).

``@rowpoly("R")`` promises the helper preserves the caller's extra columns, so
a call site threads them into the result (Tier 3). This body breaks that
promise WITHOUT producing a closed frame: ``select(pl.exclude("^tmp_.*$"))``
keeps every column whose name does NOT match the regex, so a caller extra named
``tmp_junk`` would be excluded and dropped at runtime — yet the result frame
stays open (the row variable looks alive). The sentinel-injection probe cannot
see this: the injected ``\\x00__rowvar_0__`` sentinel does not match
``^tmp_.*$``, so it survives the exclude — a silent false negative before this
check. The pattern-drop guard instead detects the reduction STRUCTURALLY: a
``select``/``drop`` keyed by a regex/selector PREDICATE over an open frame could
fall on the dropped side of an unknown extra, so preservation is not provable.

Static-only: the property is relative to the caller (whether the helper kept
columns the *caller* supplied), so Pandera cannot check it at runtime — an
input synthesized from ``InId`` alone has no ``tmp_*`` extras to drop. The
runtime-differential harness SKIPs this fixture for that reason.

Contrast ``valid/rowpoly_regex_select_preserves.py`` (the ``^.*$`` match-all
regex keeps everything) and ``valid/rowpoly_select_all_preserves.py``
(``pl.all()`` / ``cs.all()``), which DO preserve and stay silent.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

from polypolarism import rowpoly


class InId(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class OutScore(pa.DataFrameModel):
    id: int
    score: float

    class Config:
        strict = False


@rowpoly("R")
def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
    # select(pl.exclude("^tmp_.*$")) drops any caller extra matching the regex
    # -> the row variable is not provably preserved.
    return df.select(pl.exclude("^tmp_.*$")).with_columns(score=pl.col("id").cast(pl.Float64))
