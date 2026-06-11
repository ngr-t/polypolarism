"""A bare ``pl.LazyFrame`` return annotation rejects an eager frame.

The missing direction of the bare-return laziness check
(``invalid/adr0006_amendment_proofs`` pins lazy-returned-as-DataFrame).

False-positive twin: ``valid/adr0006_amendment_flows``
(``matching_laziness``).
"""

from __future__ import annotations

import polars as pl


def eager_returned_as_bare_lazyframe(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.collect()  # WRONG: collect() makes it eager; annotation says LazyFrame
