"""@rowpoly("R", drops=<selector>) declares an intentional row-variable restriction.

Sometimes a row-polymorphic helper INTENTIONALLY removes a pattern-class of the
caller's extras while preserving everything else — a sanitizer that strips every
``_internal_*`` column. Without a declaration that is a pple-rowpoly-not-preserved pattern-drop
(``valid``/``invalid`` contrast below): ``select(~cs.starts_with("_internal_"))``
could drop a caller extra matching the regex, so preservation is not provable.

``drops=cs.starts_with("_internal_")`` DECLARES the intended restriction, so the
check verifies it precisely instead of rejecting it: the body's reduction
provably removes ONLY columns the declared selector matches (and keeps every
other caller extra), so the helper is ACCEPTED. The declaration is the escape
hatch — a body that dropped MORE than declared (a broader pattern, a fixed-name
select) would still be pple-rowpoly-not-preserved.

Static-only: like the other @rowpoly fixtures the preservation property is
caller-relative (it constrains which of the *caller's* columns survive), so
Pandera cannot check it at runtime — an input synthesized from ``InId`` alone
has no ``_internal_*`` extras to drop. The runtime-differential harness SKIPs
this fixture for that reason.
"""

import pandera.polars as pa
import polars.selectors as cs
from pandera.typing.polars import DataFrame

from polypolarism import rowpoly


class InId(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


@rowpoly("R", drops=cs.starts_with("_internal_"))
def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
    # Drops exactly the declared ``_internal_*`` pattern, keeps the rest —
    # accepted because the restriction was declared via ``drops=``.
    return df.select(~cs.starts_with("_internal_"))
