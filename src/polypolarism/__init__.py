"""Polars DataFrame static type checker based on row polymorphism."""

__version__ = "0.1.0"

from typing import TypeAlias

# DF type alias for schema annotations
# Usage: def f(df: DF["{col:Type, ...}"]) -> DF["{...}"]: ...
DF: TypeAlias = "DF"
