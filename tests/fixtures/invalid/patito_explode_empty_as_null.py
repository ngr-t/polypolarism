"""Regression for #123: Patito ``explode(empty_as_null=True)`` yields a
nullable element, so declaring the exploded column non-null must FAIL.

polars 1.42 turns an empty sub-list into a null element under
``empty_as_null=True``; exploding ``list[int]`` therefore produces a nullable
``Int64``. Declaring the result as a non-null field is unsound — the same way
``shift``/``diff`` null-injection is rejected against a non-null slot.
"""

from __future__ import annotations

import patito as pt


class ListIn(pt.Model):
    vals: list[int]


class ElemInt(pt.Model):
    vals: int  # non-null


def bug(df: pt.DataFrame[ListIn]) -> pt.DataFrame[ElemInt]:
    return df.explode("vals", empty_as_null=True)
