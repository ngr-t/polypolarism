"""Patito surface tables consumed by the Patito frontend (ADR-0010).

Patito (``patito.Model``, a pydantic-for-polars layer) is the second schema
declaration frontend. Its AST-relevant surface is tiny: the schema base
class name and the ``Field`` constructor. The DataFrame / LazyFrame
annotation heads are shared with Pandera (``compat.pandera_api``) — Patito
spells them ``pt.DataFrame[Model]`` / ``pt.LazyFrame[Model]``, whose trailing
attribute name matches the same head set.

Unlike Pandera's ``DataFrameModel`` (a distinctive name matched on the
attribute tail anywhere), ``Model`` is collision-prone, so Patito detection
is import-anchored — see ``patito_schema.scan_patito_imports`` for the logic
that decides which ``Model`` / ``pt.Model`` spellings actually resolve to
Patito.
"""

from __future__ import annotations

# patito's schema base class: ``patito.Model`` / ``pt.Model`` / a bare
# ``Model`` imported ``from patito import Model``. Always import-anchored.
PATITO_MODEL_BASE: str = "Model"

# The patito package name (root segment), used to anchor ``import patito`` /
# ``from patito import Model`` spellings to the real library.
PATITO_PACKAGE: str = "patito"

# patito's ``Field`` constructor — ``pt.Field(dtype=..., unique=...)``. Only
# the ``dtype=`` keyword affects the static type (it forces a polars dtype);
# every other keyword (``unique``, value constraints) is runtime-only.
PATITO_FIELD_CALLABLE: str = "Field"

# The keyword on ``pt.Field(...)`` that overrides the annotation's dtype.
PATITO_FIELD_DTYPE_KW: str = "dtype"
