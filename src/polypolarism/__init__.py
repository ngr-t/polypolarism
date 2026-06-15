"""Polars DataFrame static type checker, inspired by row polymorphism.

Schemas are declared with Pandera ``DataFrameModel`` classes and referenced
via ``DataFrame[Schema]`` / ``LazyFrame[Schema]`` annotations from
``pandera.typing.polars``.
"""

__version__ = "0.1.0"
