"""Tests for pandera_dtype.py — AST → ColumnSpec translator."""

from __future__ import annotations

import ast

from polypolarism.pandera_dtype import parse_field_annotation
from polypolarism.types import (
    Array,
    Boolean,
    ColumnSpec,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float64,
    Int64,
    List,
    Nullable,
    Struct,
    Unknown,
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

    def test_pl_time_bare(self):
        from polypolarism.types import Time

        # Issue #19: ``str.to_time()`` returns pl.Time, so the dtype must be
        # declarable in schemas too.
        assert _parse("pl.Time") == ColumnSpec(Time(), required=True)

    def test_pl_time_call(self):
        from polypolarism.types import Time

        assert _parse("pl.Time()") == ColumnSpec(Time(), required=True)

    def test_pl_time_field_nullable(self):
        from polypolarism.types import Time

        assert _parse("pl.Time", "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(Time()), required=True
        )

    def test_pl_decimal_bare_uses_pandera_default(self):
        from polypolarism.types import Decimal

        # Landmark: polars 1.35 (Decimal stabilized).
        # A bare ``pl.Decimal`` annotation resolves through PANDERA's
        # engine default — (28, 0), not polars' materialized (38, 0)
        # (issue #75; probed: ``validate`` rejects a (38, 0) column).
        assert _parse("pl.Decimal") == ColumnSpec(Decimal(28, 0), required=True)
        assert _parse("Optional[pl.Decimal]") == ColumnSpec(Decimal(28, 0), required=False)

    def test_pl_decimal_call_no_args(self):
        from polypolarism.types import Decimal

        # Call forms carry a polars instance — polars' 38 (probed).
        assert _parse("pl.Decimal()") == ColumnSpec(Decimal(38, 0), required=True)
        assert _parse("pl.Decimal(scale=2)") == ColumnSpec(Decimal(38, 2), required=True)

    def test_pl_decimal_nested_bare_is_runtime_wildcard(self):
        # Probed (issue #75): ``pl.List(pl.Decimal)`` validates BOTH
        # List(Decimal(38,0)) and List(Decimal(28,0)) — the nested bare
        # class is a runtime wildcard, so claiming any precision would be
        # a false-positive trap.
        assert _parse("pl.List(pl.Decimal)") == ColumnSpec(List(Unknown()), required=True)

    def test_pl_decimal_unreadable_call_args_degrade_to_unknown(self):
        assert _parse("pl.Decimal(P, S)") == ColumnSpec(Unknown(), required=True)

    def test_pl_decimal_positional_args_preserved(self):
        from polypolarism.types import Decimal

        # Crucial: precision and scale must round-trip — losing them would
        # make Decimal columns interchangeable with each other regardless
        # of declared shape.
        assert _parse("pl.Decimal(20, 4)") == ColumnSpec(Decimal(20, 4), required=True)

    def test_pl_decimal_keyword_args_preserved(self):
        from polypolarism.types import Decimal

        assert _parse("pl.Decimal(precision=10, scale=2)") == ColumnSpec(
            Decimal(10, 2), required=True
        )

    def test_pl_decimal_distinct_precision_distinct_type(self):
        assert _parse("pl.Decimal(20, 4)") != _parse("pl.Decimal(20, 2)")
        assert _parse("pl.Decimal(20, 4)") != _parse("pl.Decimal(10, 4)")

    def test_pl_float16_bare(self):
        from polypolarism.types import Float16

        # Landmark: polars 1.36 (Float16 introduced).
        assert _parse("pl.Float16") == ColumnSpec(Float16(), required=True)

    def test_pl_float16_distinct_from_float32(self):
        from polypolarism.types import Float16, Float32

        assert _parse("pl.Float16") == ColumnSpec(Float16(), required=True)
        assert _parse("pl.Float32") == ColumnSpec(Float32(), required=True)
        assert _parse("pl.Float16") != _parse("pl.Float32")

    def test_pl_binary_bare(self):
        from polypolarism.types import Binary

        # Issue #51: ``.bin`` namespace needs a Binary receiver dtype.
        assert _parse("pl.Binary") == ColumnSpec(Binary(), required=True)

    def test_pl_binary_call(self):
        from polypolarism.types import Binary

        assert _parse("pl.Binary()") == ColumnSpec(Binary(), required=True)

    def test_bytes_builtin_maps_to_binary(self):
        from polypolarism.types import Binary

        # Probed (pandera 0.x + polars 1.41.2): ``x: bytes`` validates a
        # Binary column and REJECTS a String column.
        assert _parse("bytes") == ColumnSpec(Binary(), required=True)


class TestPolarsDatetimeTimezone:
    """Issue #50: ``pl.Datetime(...)`` call arguments must not be dropped.

    Probed (pandera + polars 1.41.2): ``t: pl.Datetime(time_zone="UTC")``
    rejects a tz-naive frame, and bare ``pl.Datetime`` rejects a tz-aware
    frame — the tz is part of the dtype, both ways.
    """

    def test_bare_datetime_is_naive(self):
        assert _parse("pl.Datetime") == ColumnSpec(Datetime(), required=True)

    def test_call_no_args_is_naive(self):
        assert _parse("pl.Datetime()") == ColumnSpec(Datetime(), required=True)

    def test_positional_time_unit_and_zone(self):
        assert _parse('pl.Datetime("us", "UTC")') == ColumnSpec(Datetime(tz="UTC"), required=True)

    def test_time_zone_keyword(self):
        assert _parse('pl.Datetime(time_zone="UTC")') == ColumnSpec(
            Datetime(tz="UTC"), required=True
        )

    def test_time_unit_only_is_naive(self):
        # Issue #66: the unit participates in dtype identity (tz stays None).
        assert _parse('pl.Datetime("us")') == ColumnSpec(Datetime(), required=True)
        assert _parse('pl.Datetime(time_unit="ms")') == ColumnSpec(
            Datetime(unit="ms"), required=True
        )

    def test_explicit_none_time_zone_is_naive(self):
        assert _parse('pl.Datetime("us", None)') == ColumnSpec(Datetime(), required=True)
        assert _parse("pl.Datetime(time_zone=None)") == ColumnSpec(Datetime(), required=True)

    def test_series_wrapped(self):
        assert _parse('Series[pl.Datetime("us", "UTC")]') == ColumnSpec(
            Datetime(tz="UTC"), required=True
        )

    def test_optional_wrapped(self):
        assert _parse('Optional[pl.Datetime(time_zone="UTC")]') == ColumnSpec(
            Datetime(tz="UTC"), required=False
        )

    def test_field_nullable(self):
        assert _parse('pl.Datetime(time_zone="UTC")', "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(Datetime(tz="UTC")), required=True
        )

    def test_non_literal_time_zone_degrades_to_unknown(self):
        # The tz is unknowable — claiming naive would be a false-positive
        # trap now that tz mismatches are flagged.
        assert _parse("pl.Datetime(time_zone=TZ)") == ColumnSpec(Unknown(), required=True)

    def test_annotated_form_with_tz(self):
        # Probed: ``Annotated[pl.Datetime, "us", "UTC"]`` enforces UTC.
        assert _parse('Annotated[pl.Datetime, "us", "UTC"]') == ColumnSpec(
            Datetime(tz="UTC"), required=True
        )

    def test_annotated_form_single_arg_degrades_to_unknown(self):
        # Probed: pandera raises TypeError for the 1-arg form (it requires
        # all positional dtype arguments) — the schema is broken at runtime,
        # so polypolarism stays silent rather than guessing a dtype.
        assert _parse('Annotated[pl.Datetime, "us"]') == ColumnSpec(Unknown(), required=True)

    def test_distinct_time_zones_are_distinct_dtypes(self):
        assert _parse('pl.Datetime(time_zone="UTC")') != _parse(
            'pl.Datetime(time_zone="Asia/Tokyo")'
        )


class TestDatetimeDurationTimeUnit:
    """Issue #66: the time unit participates in dtype identity.

    Probed against pandera 0.31.1 / polars 1.41.2: the Annotated metadata
    are the dtype's positional arguments (ALL required — wrong arity is a
    pandera TypeError); a ``None`` literal takes the polars default.
    """

    def test_annotated_datetime_carries_unit(self):
        assert _parse('Annotated[pl.Datetime, "ns", None]') == ColumnSpec(
            Datetime(unit="ns"), required=True
        )
        assert _parse('Annotated[pl.Datetime, None, "UTC"]') == ColumnSpec(
            Datetime(tz="UTC"), required=True
        )

    def test_annotated_datetime_non_literal_unit_degrades_to_unknown(self):
        assert _parse('Annotated[pl.Datetime, UNIT, "UTC"]') == ColumnSpec(Unknown(), required=True)

    def test_annotated_duration_carries_unit(self):
        assert _parse('Annotated[pl.Duration, "ms"]') == ColumnSpec(
            Duration(unit="ms"), required=True
        )

    def test_annotated_duration_wrong_arity_degrades_to_unknown(self):
        assert _parse('Annotated[pl.Duration, "ms", "extra"]') == ColumnSpec(
            Unknown(), required=True
        )

    def test_duration_call_form_carries_unit(self):
        assert _parse('pl.Duration("ms")') == ColumnSpec(Duration(unit="ms"), required=True)
        assert _parse('pl.Duration(time_unit="ns")') == ColumnSpec(
            Duration(unit="ns"), required=True
        )

    def test_bare_duration_is_polars_default(self):
        assert _parse("pl.Duration") == ColumnSpec(Duration(unit="us"), required=True)

    def test_non_literal_duration_unit_degrades_to_unknown(self):
        assert _parse("pl.Duration(UNIT)") == ColumnSpec(Unknown(), required=True)


class TestAnnotatedDecimal:
    """Issue #65: ``Annotated[pl.Decimal, p, s]`` was parsed as the bare
    ``Decimal(38, 0)`` default, rejecting exactly-matching code."""

    def test_annotated_decimal_carries_precision_and_scale(self):
        assert _parse("Annotated[pl.Decimal, 12, 4]") == ColumnSpec(Decimal(12, 4), required=True)

    def test_none_literal_takes_polars_default(self):
        # Probed: ``Annotated[pl.Decimal, None, 4]`` -> Decimal(38, 4).
        assert _parse("Annotated[pl.Decimal, None, 4]") == ColumnSpec(Decimal(38, 4), required=True)

    def test_wrong_arity_degrades_to_unknown(self):
        # Probed: pandera raises TypeError for the 1-arg form (requires
        # all positional dtype arguments).
        assert _parse("Annotated[pl.Decimal, 12]") == ColumnSpec(Unknown(), required=True)

    def test_non_literal_args_degrade_to_unknown(self):
        assert _parse("Annotated[pl.Decimal, P, S]") == ColumnSpec(Unknown(), required=True)


class TestEnumCategories:
    """Issue #67: Enum categories are an ordered part of dtype identity."""

    def test_call_form_carries_ordered_categories(self):
        assert _parse('pl.Enum(["a", "b"])') == ColumnSpec(
            Enum(categories=("a", "b")), required=True
        )

    def test_category_order_is_identity(self):
        assert _parse('pl.Enum(["a", "b"])') != _parse('pl.Enum(["b", "a"])')

    def test_annotated_form_carries_categories(self):
        # Probed: pandera builds Enum(categories=['a', 'b']) for this form.
        assert _parse('Annotated[pl.Enum, ["a", "b"]]') == ColumnSpec(
            Enum(categories=("a", "b")), required=True
        )

    def test_bare_and_non_literal_forms_keep_unknown_categories(self):
        # Still provably SOME Enum — the categories=None wildcard, not
        # Unknown (cross-class mismatches stay detectable).
        assert _parse("pl.Enum") == ColumnSpec(Enum(), required=True)
        assert _parse("pl.Enum(CATS)") == ColumnSpec(Enum(), required=True)
        assert _parse("Annotated[pl.Enum, CATS]") == ColumnSpec(Enum(), required=True)


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

    def test_annotated_array_is_array(self):
        # Issue #53 / backlog C-7: Array is a real dtype and the width is
        # tracked.
        assert _parse("Annotated[pl.Array, pl.Int64(), 3]") == ColumnSpec(
            Array(Int64(), 3), required=True
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


class TestContainerCallForms:
    """Issue #10: ``pl.List(...)`` / ``pl.Array(...)`` / ``pl.Struct(...)``
    call-form annotations register as columns with element types."""

    def test_pl_list_of_int64(self):
        assert _parse("pl.List(pl.Int64)") == ColumnSpec(List(Int64()), required=True)

    def test_pl_list_of_int64_call(self):
        assert _parse("pl.List(pl.Int64())") == ColumnSpec(List(Int64()), required=True)

    def test_pl_list_of_bare_struct_is_list_unknown(self):
        assert _parse("pl.List(pl.Struct)") == ColumnSpec(List(Unknown()), required=True)

    def test_pl_list_nested(self):
        assert _parse("pl.List(pl.List(pl.Int64))") == ColumnSpec(
            List(List(Int64())), required=True
        )

    def test_pl_array_with_width(self):
        """Issue #53 / backlog C-7: ``pl.Array(...)`` tracks its width."""
        assert _parse("pl.Array(pl.Int64, 4)") == ColumnSpec(Array(Int64(), 4), required=True)

    def test_pl_array_shape_keyword(self):
        assert _parse("pl.Array(pl.Int64, shape=4)") == ColumnSpec(Array(Int64(), 4), required=True)

    def test_pl_array_shape_one_tuple(self):
        assert _parse("pl.Array(pl.Int64, (4,))") == ColumnSpec(Array(Int64(), 4), required=True)

    def test_pl_array_multidim_shape_width_unknown(self):
        # Multi-dimensional shapes are out of the 1-D width model.
        assert _parse("pl.Array(pl.Int64, (2, 3))") == ColumnSpec(Array(Int64()), required=True)

    def test_pl_array_non_literal_width_unknown(self):
        assert _parse("pl.Array(pl.Int64, n)") == ColumnSpec(Array(Int64()), required=True)

    def test_pl_array_nested_in_list(self):
        assert _parse("pl.List(pl.Array(pl.Int64, 4))") == ColumnSpec(
            List(Array(Int64(), 4)), required=True
        )

    def test_pl_array_unparseable_element_is_unknown(self):
        assert _parse("pl.Array(some_variable, 4)") == ColumnSpec(
            Array(Unknown(), 4), required=True
        )

    def test_pl_array_nullable_field(self):
        assert _parse("pl.Array(pl.Int64, 4)", "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(Array(Int64(), 4)), required=True
        )

    def test_pl_struct_dict_literal(self):
        assert _parse('pl.Struct({"a": pl.Utf8, "b": pl.Float64()})') == ColumnSpec(
            Struct({"a": Utf8(), "b": Float64()}), required=True
        )

    def test_pl_struct_dict_unparseable_value_becomes_unknown(self):
        assert _parse('pl.Struct({"a": some_variable})') == ColumnSpec(
            Struct({"a": Unknown()}), required=True
        )

    def test_pl_struct_non_dict_arg_is_unknown(self):
        assert _parse('pl.Struct([pl.Field("a", pl.Utf8)])') == ColumnSpec(Unknown(), required=True)

    def test_pl_list_with_field_value(self):
        assert _parse("pl.List(pl.Struct)", "pa.Field()") == ColumnSpec(
            List(Unknown()), required=True
        )

    def test_pl_list_nullable_field(self):
        assert _parse("pl.List(pl.Int64)", "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(List(Int64())), required=True
        )


class TestContainerBareForms:
    """Bare ``pl.List`` / ``pl.Array`` / ``pl.Struct`` attribute annotations."""

    def test_pl_list_bare(self):
        assert _parse("pl.List") == ColumnSpec(List(Unknown()), required=True)

    def test_pl_array_bare(self):
        assert _parse("pl.Array") == ColumnSpec(Array(Unknown()), required=True)

    def test_pl_struct_bare_is_unknown(self):
        """A struct whose fields we don't know carries no usable shape —
        Unknown keeps everything downstream lenient (not Struct({}), which
        would unnest to zero columns)."""
        assert _parse("pl.Struct") == ColumnSpec(Unknown(), required=True)
