"""Frame-typed attributes declared by annotation resolve (issue #108).

Follow-up to #104/#105: besides ``__init__``'s ``self.x = <frame-param>``
assignment, a frame-typed attribute declared via a ``@dataclass`` field, a
class-level annotation, or a ``@property`` return is now read from its
annotation — so a method returning the attribute no longer degrades to
"could not infer". (Attributes set only through a helper method called from
``__init__`` remain out of scope — dataflow-harder.)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = False


@dataclass
class DataclassField:
    src: DataFrame[KV]

    @pa.check_types
    def use(self) -> DataFrame[KV]:
        return self.src


class ClassLevelAnnotation:
    src: DataFrame[KV]

    @pa.check_types
    def use(self) -> DataFrame[KV]:
        return self.src


class PropertyReturn:
    def __init__(self, src: DataFrame[KV]) -> None:
        self._src = src

    @property
    def src(self) -> DataFrame[KV]:
        return self._src

    @pa.check_types
    def use(self) -> DataFrame[KV]:
        return self.src
