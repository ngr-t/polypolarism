"""Polars DataFrame static type checker, inspired by row polymorphism.

Schemas are declared with Pandera ``DataFrameModel`` classes and referenced
via ``DataFrame[Schema]`` / ``LazyFrame[Schema]`` annotations from
``pandera.typing.polars``.

``rowpoly`` is the runtime-inert decorator that names a static row variable
for column-preserving helpers (backlog C-14); it is a no-op at runtime so
Pandera validation is unaffected.
"""

from polypolarism.rowpoly import rowpoly

__all__ = ["rowpoly", "__version__"]

__version__ = "0.1.0"
