"""Inherited frame-typed self.<attr> resolves along the MRO (issue #105).

Follow-up to #104: a ``self.<attr>`` stashed in a *parent* class's __init__ is
visible to subclass methods — whether the subclass inherits __init__ outright,
calls ``super().__init__(...)``, or sits two levels down. A subclass that
rebinds the attribute in its own __init__ shadows the inherited one.
"""

from __future__ import annotations

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = False


class KW(pa.DataFrameModel):
    k: str
    w: float

    class Config:
        strict = True
        coerce = False


class Base:
    def __init__(self, src: DataFrame[KV]) -> None:
        self.src = src


class ChildInherit(Base):
    @pa.check_types
    def use(self) -> DataFrame[KV]:
        return self.src


class ChildSuper(Base):
    def __init__(self, src: DataFrame[KV]) -> None:
        super().__init__(src)

    @pa.check_types
    def use(self) -> DataFrame[KV]:
        return self.src


class GrandChild(ChildInherit):
    @pa.check_types
    def use(self) -> DataFrame[KV]:
        return self.src


class ChildOverride(Base):
    def __init__(self, other: DataFrame[KW]) -> None:
        self.src = other

    @pa.check_types
    def use(self) -> DataFrame[KW]:
        return self.src
