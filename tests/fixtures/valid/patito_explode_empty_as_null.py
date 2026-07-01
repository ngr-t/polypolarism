"""Regression for #123: Patito ``explode`` nullability tracks ``empty_as_null``.

``empty_as_null=True`` (polars 1.42) turns an empty sub-list into a null, so the
exploded element is nullable — declaring it ``int | None`` (Patito nullable
value, column still required) is sound. ``empty_as_null=False`` drops empties,
so the element stays non-null and a plain ``int`` field is satisfied.
"""

from __future__ import annotations

import patito as pt


class ListIn(pt.Model):
    vals: list[int]


class ElemNullable(pt.Model):
    vals: int | None  # nullable value, column required


class ElemInt(pt.Model):
    vals: int  # non-null


def keeps_null(df: pt.DataFrame[ListIn]) -> pt.DataFrame[ElemNullable]:
    return df.explode("vals", empty_as_null=True)


def drops_empties(df: pt.DataFrame[ListIn]) -> pt.DataFrame[ElemInt]:
    return df.explode("vals", empty_as_null=False)
