"""Tests for pandera_dtype.py — AST → ColumnSpec translator."""

from __future__ import annotations

import ast

from polypolarism.pandera_dtype import parse_field_annotation
from polypolarism.types import (
    Boolean,
    ColumnSpec,
    Float64,
    Int64,
    List,
    Nullable,
    Struct,
    Utf8,
)


def _parse(annotation_src: str, value_src: str | None = None) -> ColumnSpec | None:
    """Helper: build a class with a single field and parse it."""
    class_src = f"class S:\n    x: {annotation_src}"
    if value_src is not None:
        class_src += f" = {value_src}"
    tree = ast.parse(class_src)
    cls = tree.body[0]
    assert isinstance(cls, ast.ClassDef)
    stmt = cls.body[0]
    assert isinstance(stmt, ast.AnnAssign)
    return parse_field_annotation(stmt.annotation, stmt.value)


class TestPythonBuiltins:
    def test_int(self):
        assert _parse("int") == ColumnSpec(Int64(), required=True)

    def test_str(self):
        assert _parse("str") == ColumnSpec(Utf8(), required=True)

    def test_float(self):
        assert _parse("float") == ColumnSpec(Float64(), required=True)

    def test_bool(self):
        assert _parse("bool") == ColumnSpec(Boolean(), required=True)


class TestPolarsDtype:
    def test_pl_int64_bare(self):
        assert _parse("pl.Int64") == ColumnSpec(Int64(), required=True)

    def test_pl_int64_call(self):
        assert _parse("pl.Int64()") == ColumnSpec(Int64(), required=True)

    def test_pl_utf8(self):
        assert _parse("pl.Utf8") == ColumnSpec(Utf8(), required=True)

    def test_pl_string(self):
        assert _parse("pl.String") == ColumnSpec(Utf8(), required=True)

    def test_pl_float64(self):
        assert _parse("pl.Float64") == ColumnSpec(Float64(), required=True)


class TestPolarsLandmarkDtypes:
    """Dtypes added across the polars 1.x line. Pinning expectations here
    catches regressions in the centralized DTYPE_NAME_MAP and lets future
    landmark commits anchor at a specific minor."""

    def test_pl_int128_bare(self):
        from polypolarism.types import Int128

        # Landmark: polars 1.18 (Int128 introduced).
        assert _parse("pl.Int128") == ColumnSpec(Int128(), required=True)

    def test_pl_int128_call(self):
        from polypolarism.types import Int128

        assert _parse("pl.Int128()") == ColumnSpec(Int128(), required=True)

    def test_pl_int128_optional(self):
        from polypolarism.types import Int128

        assert _parse("Optional[pl.Int128]") == ColumnSpec(Int128(), required=False)

    def test_pl_int128_field_nullable(self):
        from polypolarism.types import Int128

        assert _parse("pl.Int128", "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(Int128()), required=True
        )

    def test_pl_enum_bare(self):
        from polypolarism.types import Enum

        # Landmark: polars 1.25 (Enum stabilized).
        assert _parse("pl.Enum") == ColumnSpec(Enum(), required=True)

    def test_pl_enum_call(self):
        from polypolarism.types import Enum

        assert _parse("pl.Enum()") == ColumnSpec(Enum(), required=True)

    def test_pl_enum_distinct_from_categorical(self):
        from polypolarism.types import Categorical, Enum

        assert _parse("pl.Enum") != _parse("pl.Categorical")
        # But each is internally consistent:
        assert _parse("pl.Enum") == ColumnSpec(Enum(), required=True)
        assert _parse("pl.Categorical") == ColumnSpec(Categorical(), required=True)

    def test_pl_uint128_bare(self):
        from polypolarism.types import UInt128

        # Landmark: polars 1.34 (UInt128 introduced).
        assert _parse("pl.UInt128") == ColumnSpec(UInt128(), required=True)

    def test_pl_uint128_distinct_from_int128(self):
        from polypolarism.types import Int128, UInt128

        assert _parse("pl.UInt128") == ColumnSpec(UInt128(), required=True)
        assert _parse("pl.Int128") == ColumnSpec(Int128(), required=True)
        assert _parse("pl.UInt128") != _parse("pl.Int128")


class TestOptional:
    def test_optional_int(self):
        assert _parse("Optional[int]") == ColumnSpec(Int64(), required=False)

    def test_optional_pl_int64(self):
        assert _parse("Optional[pl.Int64]") == ColumnSpec(Int64(), required=False)

    def test_typing_optional_qualified(self):
        assert _parse("typing.Optional[int]") == ColumnSpec(Int64(), required=False)

    def test_pep604_union_none(self):
        assert _parse("int | None") == ColumnSpec(Int64(), required=False)

    def test_pep604_union_none_reversed(self):
        assert _parse("None | int") == ColumnSpec(Int64(), required=False)


class TestFieldNullable:
    def test_field_nullable_true(self):
        assert _parse("int", "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(Int64()), required=True
        )

    def test_field_nullable_unqualified(self):
        assert _parse("str", "Field(nullable=True)") == ColumnSpec(Nullable(Utf8()), required=True)

    def test_field_without_nullable(self):
        assert _parse("int", "pa.Field(unique=True)") == ColumnSpec(Int64(), required=True)

    def test_optional_with_nullable_field(self):
        # Optional[T] (column may be absent) AND nullable values
        assert _parse("Optional[int]", "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(Int64()), required=False
        )

    def test_field_with_value_constraints_ignored(self):
        # Value constraints are runtime-only, ignored statically
        assert _parse("int", "pa.Field(gt=0, lt=100, unique=True)") == ColumnSpec(
            Int64(), required=True
        )


class TestAnnotatedList:
    def test_annotated_list_int(self):
        assert _parse("Annotated[pl.List, pl.Int64()]") == ColumnSpec(List(Int64()), required=True)

    def test_annotated_list_utf8(self):
        assert _parse("Annotated[pl.List, pl.Utf8()]") == ColumnSpec(List(Utf8()), required=True)

    def test_annotated_array_treated_as_list(self):
        # Width is ignored for our purposes
        assert _parse("Annotated[pl.Array, pl.Int64(), 3]") == ColumnSpec(
            List(Int64()), required=True
        )


class TestAnnotatedStruct:
    def test_annotated_struct(self):
        result = _parse('Annotated[pl.Struct, {"a": pl.Utf8(), "b": pl.Float64()}]')
        assert result == ColumnSpec(Struct({"a": Utf8(), "b": Float64()}), required=True)

    def test_annotated_struct_optional(self):
        result = _parse('Optional[Annotated[pl.Struct, {"a": pl.Utf8()}]]')
        assert result == ColumnSpec(Struct({"a": Utf8()}), required=False)


class TestStdlibTemporal:
    """Python stdlib ``date`` / ``datetime`` / ``timedelta`` types are
    accepted as field annotations the same way ``int`` / ``str`` are.
    Both ``from datetime import date`` (bare name) and
    ``import datetime; datetime.date`` (attribute) forms work."""

    def test_bare_date(self):
        from polypolarism.types import Date

        assert _parse("date") == ColumnSpec(Date(), required=True)

    def test_bare_datetime(self):
        from polypolarism.types import Datetime

        assert _parse("datetime") == ColumnSpec(Datetime(), required=True)

    def test_bare_timedelta(self):
        from polypolarism.types import Duration

        assert _parse("timedelta") == ColumnSpec(Duration(), required=True)

    def test_qualified_datetime_date(self):
        from polypolarism.types import Date

        assert _parse("datetime.date") == ColumnSpec(Date(), required=True)

    def test_qualified_datetime_datetime(self):
        from polypolarism.types import Datetime

        assert _parse("datetime.datetime") == ColumnSpec(Datetime(), required=True)

    def test_qualified_datetime_timedelta(self):
        from polypolarism.types import Duration

        assert _parse("datetime.timedelta") == ColumnSpec(Duration(), required=True)

    def test_aliased_module(self):
        # ``import datetime as dt`` — any non-``pl`` prefix should work.
        from polypolarism.types import Date

        assert _parse("dt.date") == ColumnSpec(Date(), required=True)

    def test_optional_date(self):
        from polypolarism.types import Date

        assert _parse("Optional[date]") == ColumnSpec(Date(), required=False)

    def test_pl_dtype_unchanged(self):
        # Make sure the existing ``pl.Date`` etc. path didn't regress.
        from polypolarism.types import Date

        assert _parse("pl.Date") == ColumnSpec(Date(), required=True)


class TestSeriesWrapper:
    """``Series[T]`` is the canonical pandera class-based form and should
    parse identically to bare ``T`` (issue #4)."""

    def test_series_str(self):
        assert _parse("Series[str]") == ColumnSpec(Utf8(), required=True)

    def test_series_pl_int32(self):
        from polypolarism.types import Int32

        assert _parse("Series[pl.Int32]") == ColumnSpec(Int32(), required=True)

    def test_series_pl_date(self):
        from polypolarism.types import Date

        assert _parse("Series[pl.Date]") == ColumnSpec(Date(), required=True)

    def test_qualified_series(self):
        # ``pa.typing.Series[T]`` / ``pandera.typing.polars.Series[T]``
        # — qualified forms should parse the same way.
        assert _parse("pa.typing.Series[int]") == ColumnSpec(Int64(), required=True)
        assert _parse("pandera.typing.polars.Series[str]") == ColumnSpec(Utf8(), required=True)

    def test_series_optional_inner(self):
        # ``Series[Optional[T]]`` should be ``required=False``.
        assert _parse("Series[Optional[int]]") == ColumnSpec(Int64(), required=False)

    def test_optional_series(self):
        # ``Optional[Series[T]]`` is the equivalent shape with the wrap
        # order reversed.
        assert _parse("Optional[Series[int]]") == ColumnSpec(Int64(), required=False)

    def test_series_with_field_nullable(self):
        # ``Series[T] = Field(nullable=True)`` should still wrap in Nullable.
        from polypolarism.types import Int32

        assert _parse("Series[pl.Int32]", "Field(nullable=True)") == ColumnSpec(
            Nullable(Int32()), required=True
        )


class TestUnknown:
    def test_unknown_name_returns_none(self):
        assert _parse("MyCustom") is None

    def test_unknown_pl_attr_returns_none(self):
        assert _parse("pl.NotARealType") is None

    def test_other_module_returns_none(self):
        assert _parse("np.int64") is None
