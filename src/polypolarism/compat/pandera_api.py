"""Pandera surface tables consumed by the analyzer.

Pandera's AST-relevant surface is much smaller than polars' — it boils
down to a couple of class-name sets (the schema base classes, the
DataFrame / LazyFrame heads in annotations) plus the ``Field``
constructor for nullable detection. This module is the single source
of truth for those.

Versioning notes:
- ``DataFrameModel`` is the post-0.20 canonical base; ``SchemaModel`` is
  the pre-0.20 legacy alias. Both are accepted indefinitely (see
  ADR-0001) and the analyzer makes no behavioral distinction.
- ``DataFrame`` / ``LazyFrame`` annotation heads are matched on the
  trailing attribute name, so qualified forms like
  ``pandera.typing.polars.DataFrame`` and ``pa.typing.LazyFrame`` resolve
  the same as bare ``DataFrame`` / ``LazyFrame``.
"""

from __future__ import annotations

from polypolarism.types import Decimal

# Class names treated as Pandera schema base classes. Qualified forms
# (``pa.DataFrameModel``, ``pandera.polars.DataFrameModel``) are matched
# on the trailing attribute name.
SCHEMA_BASE_NAMES: frozenset[str] = frozenset({"DataFrameModel", "SchemaModel"})

# What pandera builds for a BARE ``d: pl.Decimal`` class-name annotation
# (issue #75). Pandera's engine resolves the bare class through its own
# Decimal default — precision 28 (Python's ``decimal`` module default) —
# NOT polars' materialized (38, 0). Probed (pandera 0.31.1):
# ``to_schema()`` reports ``Decimal(precision=28, scale=0)``, ``validate``
# passes a (28, 0) column and rejects (38, 0), and the value is stable
# under ``decimal.getcontext()`` changes (pandera pins 28 at import time).
# Every CALL form (``pl.Decimal()``, omitted/None arguments) carries a
# polars instance instead and keeps polars' 38 — see
# ``polars_api.DECIMAL_DEFAULT``.
PANDERA_BARE_DECIMAL: Decimal = Decimal(28, 0)

# Annotation head names recognised in ``DataFrame[Schema]`` /
# ``LazyFrame[Schema]``. Qualified prefixes are stripped before matching.
FRAME_ANNOTATION_HEADS: frozenset[str] = frozenset({"DataFrame", "LazyFrame"})

# Pandera's ``Field`` constructor — recognized either as a bare ``Field``
# (when imported as ``from pandera import Field``) or as
# ``pa.Field`` / ``pandera.Field``.
FIELD_CALLABLE_NAME: str = "Field"
