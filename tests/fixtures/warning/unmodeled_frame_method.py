"""`pplw-unmodeled-method` at FRAME level: a frame-returning method polypolarism does
not model silently untracks the variable — downstream checks die quietly
(backlog N-3).

Probed (polars 1.41.2): ``DataFrame.interpolate`` returns a DataFrame;
polypolarism does not model it, so past the call ``smooth``'s result is
invisible to every downstream check — the warning is the only trace
(with a declared frame return the loss would at least surface as a
"could not infer return type" error). The warning fires only for
methods probed to RETURN a frame; terminal methods and the repair are
shown by ``valid/unmodeled_frame_method_validated`` (``to_dicts()``
stays silent; wrapping the call in ``Schema.validate`` retracts the
warning).
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    v: float = pa.Field(nullable=True)


def smooth(df: DataFrame[In]):
    return df.interpolate()
