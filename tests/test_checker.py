"""Tests for checker."""

import textwrap

from polypolarism.analyzer import FunctionAnalysis
from polypolarism.checker import (
    ExtraColumn,
    MissingColumn,
    TypeDifference,
    _is_coercible_difference,
    _subtype_verdict,
    check_function,
    check_source,
)
from polypolarism.types import (
    Array,
    Datetime,
    Duration,
    Enum,
    Float64,
    FrameType,
    Int64,
    Nullable,
    RowVar,
    UInt32,
    Unknown,
    Utf8,
)
from polypolarism.types import (
    List as ListType,
)


class TestCheckFunctionBasic:
    """Test basic function checking."""

    def test_matching_types_pass(self):
        """Function with matching declared and inferred types passes."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"id": Int64()})},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=FrameType({"id": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is True
        assert len(result.errors) == 0

    def test_analysis_errors_propagate(self):
        """Analysis errors are included in check result."""
        analysis = FunctionAnalysis(
            name="bad_func",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"id": Int64()})},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=None,
            errors=["Column 'missing' not found in DataFrame"],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any("missing" in str(e) for e in result.errors)


class TestCheckMissingColumn:
    """Test detection of missing columns."""

    def test_detects_missing_column(self):
        """Detect when declared column is missing from inferred type."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"id": Int64()})},
            declared_return_type=FrameType({"id": Int64(), "name": Utf8()}),
            inferred_return_type=FrameType({"id": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any(isinstance(e, MissingColumn) for e in result.errors)
        missing_errors = [e for e in result.errors if isinstance(e, MissingColumn)]
        assert any(e.column == "name" for e in missing_errors)

    def test_missing_column_error_message(self):
        """Missing column error has helpful message with the family code."""
        error = MissingColumn("name", Utf8())

        message = str(error)

        assert message.startswith("[pple-return-type] ")
        assert "name" in message
        assert "Utf8" in message


class TestCheckExtraColumn:
    """Test detection of extra columns under strict mode."""

    def test_detects_extra_column_when_strict(self):
        """Strict declared type: inferred extras are reported."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"id": Int64(), "name": Utf8()})},
            declared_return_type=FrameType({"id": Int64()}, strict=True),
            inferred_return_type=FrameType({"id": Int64(), "name": Utf8()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any(isinstance(e, ExtraColumn) for e in result.errors)
        extra_errors = [e for e in result.errors if isinstance(e, ExtraColumn)]
        assert any(e.column == "name" for e in extra_errors)

    def test_no_error_when_not_strict(self):
        """Non-strict (default) declared type: extras are tolerated (structural subtyping)."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"id": Int64(), "name": Utf8()})},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=FrameType({"id": Int64(), "name": Utf8()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is True

    def test_extra_column_error_message(self):
        """Extra column error has helpful message with the family code."""
        error = ExtraColumn("extra", Float64())

        message = str(error)

        assert message.startswith("[pple-return-type] ")
        assert "extra" in message


class TestCheckTypeDifference:
    """Test detection of type differences."""

    def test_detects_type_mismatch(self):
        """Detect when column has different type than declared."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"value": Int64()})},
            declared_return_type=FrameType({"value": Float64()}),
            inferred_return_type=FrameType({"value": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)
        type_errors = [e for e in result.errors if isinstance(e, TypeDifference)]
        assert any(
            e.column == "value" and e.declared == Float64() and e.inferred == Int64()
            for e in type_errors
        )

    def test_type_difference_error_message(self):
        """Type difference error has helpful message with the family code."""
        error = TypeDifference("value", declared=Float64(), inferred=Int64())

        message = str(error)

        assert message.startswith("[pple-return-type] ")
        assert "value" in message
        assert "Float64" in message
        assert "Int64" in message


class TestCheckNullability:
    """Test nullability checking."""

    def test_nullable_inferred_matches_nullable_declared(self):
        """Nullable inferred type matches nullable declared type."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"value": Nullable(Int64())})},
            declared_return_type=FrameType({"value": Nullable(Int64())}),
            inferred_return_type=FrameType({"value": Nullable(Int64())}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is True

    def test_inferred_nullable_declared_non_nullable_fails(self):
        """Inferred Nullable when declared non-nullable fails."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"value": Int64()})},
            declared_return_type=FrameType({"value": Int64()}),
            inferred_return_type=FrameType({"value": Nullable(Int64())}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)

    def test_inferred_non_nullable_declared_nullable_passes(self):
        """Inferred non-nullable is compatible with declared nullable."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"value": Int64()})},
            declared_return_type=FrameType({"value": Nullable(Int64())}),
            inferred_return_type=FrameType({"value": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        # Non-nullable is a subtype of nullable
        assert result.passed is True


class TestUnknownCompatibility:
    """Unknown dtype is compatible with every dtype in both directions."""

    def _analysis(self, declared: FrameType, inferred: FrameType) -> FunctionAnalysis:
        return FunctionAnalysis(
            name="f",
            lineno=1,
            end_lineno=1,
            input_types={},
            declared_return_type=declared,
            inferred_return_type=inferred,
            errors=[],
        )

    def test_inferred_unknown_passes_declared_int64(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Int64()}),
                inferred=FrameType({"a": Unknown()}),
            )
        )
        assert result.passed is True

    def test_declared_unknown_passes_inferred_utf8(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Unknown()}),
                inferred=FrameType({"a": Utf8()}),
            )
        )
        assert result.passed is True

    def test_inferred_nullable_unknown_passes_non_nullable_declared(self):
        """Uncertainty must not error, even against a non-nullable slot."""
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Int64()}),
                inferred=FrameType({"a": Nullable(Unknown())}),
            )
        )
        assert result.passed is True

    def test_list_of_unknown_passes_declared_list_of_int(self):
        """Unknown compatibility recurses into containers: an un-inferable
        ``list.eval`` body yields ``List[Unknown]``, which must satisfy a
        declared ``List[Int64]``."""
        result = check_function(
            self._analysis(
                declared=FrameType({"a": ListType(Int64())}),
                inferred=FrameType({"a": ListType(Unknown())}),
            )
        )
        assert result.passed is True

    def test_declared_list_of_unknown_passes_inferred_list_of_utf8(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": ListType(Unknown())}),
                inferred=FrameType({"a": ListType(Utf8())}),
            )
        )
        assert result.passed is True

    def test_list_inner_mismatch_still_fails(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": ListType(Utf8())}),
                inferred=FrameType({"a": ListType(Int64())}),
            )
        )
        assert result.passed is False

    def test_nullable_list_of_unknown_vs_non_nullable_declared_fails(self):
        """The Unknown leniency is about the dtype, not the column's own
        nullability — a Nullable list cannot fill a non-nullable slot."""
        result = check_function(
            self._analysis(
                declared=FrameType({"a": ListType(Int64())}),
                inferred=FrameType({"a": Nullable(ListType(Unknown()))}),
            )
        )
        assert result.passed is False


class TestArrayContainerChecking:
    """Issue #53: Array recurses like List, and Array vs List is NOT
    substitutable (probed: pandera validation rejects a List column where
    ``pl.Array(...)`` is declared and vice versa — without coerce)."""

    def _analysis(self, declared: FrameType, inferred: FrameType) -> FunctionAnalysis:
        return FunctionAnalysis(
            name="f",
            lineno=1,
            end_lineno=1,
            input_types={},
            declared_return_type=declared,
            inferred_return_type=inferred,
            errors=[],
        )

    def test_array_of_unknown_passes_declared_array_of_int(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Array(Int64())}),
                inferred=FrameType({"a": Array(Unknown())}),
            )
        )
        assert result.passed is True

    def test_declared_array_of_unknown_passes_inferred_array_of_utf8(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Array(Unknown())}),
                inferred=FrameType({"a": Array(Utf8())}),
            )
        )
        assert result.passed is True

    def test_array_inner_mismatch_fails(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Array(Utf8())}),
                inferred=FrameType({"a": Array(Int64())}),
            )
        )
        assert result.passed is False

    def test_inferred_list_fails_declared_array(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Array(Int64())}),
                inferred=FrameType({"a": ListType(Int64())}),
            )
        )
        assert result.passed is False

    def test_inferred_array_fails_declared_list(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": ListType(Int64())}),
                inferred=FrameType({"a": Array(Int64())}),
            )
        )
        assert result.passed is False

    def test_nullable_array_cannot_fill_non_nullable_slot(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Array(Int64())}),
                inferred=FrameType({"a": Nullable(Array(Int64()))}),
            )
        )
        assert result.passed is False

    def test_array_can_fill_nullable_array_slot(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Nullable(Array(Int64()))}),
                inferred=FrameType({"a": Array(Int64())}),
            )
        )
        assert result.passed is True


class TestArrayCoercibleDifference:
    """Probed (pandera + polars 1.41.2): under ``Config.coerce`` pandera
    casts a List column into a declared ``pl.Array(...)`` (succeeds when
    widths line up — value-dependent) and an Array column into a declared
    ``pl.List(...)`` (always valid). Every probed polars cast between
    list-like containers is structurally permissive, so any difference
    involving an Array container is treated as coercible — flagging would
    manufacture false positives."""

    def test_list_vs_declared_array_is_coercible(self):
        assert _is_coercible_difference(ListType(Int64()), Array(Int64())) is True

    def test_array_vs_declared_list_is_coercible(self):
        assert _is_coercible_difference(Array(Int64()), ListType(Int64())) is True

    def test_array_vs_array_inner_difference_is_coercible(self):
        assert _is_coercible_difference(Array(Int64()), Array(Utf8())) is True

    def test_list_vs_list_recurses_on_elements(self):
        # Issue #58: coerce casts list elements too (probed: List(Int64) ->
        # declared List(Utf8) / List(Float64) pass under pandera coerce).
        assert _is_coercible_difference(ListType(Int64()), ListType(Utf8())) is True
        assert _is_coercible_difference(ListType(Int64()), ListType(Float64())) is True
        # ... but value-dependent element casts stay errors,
        assert _is_coercible_difference(ListType(Utf8()), ListType(Int64())) is False
        # ... and casting does not remove element nulls.
        assert _is_coercible_difference(ListType(Nullable(Int64())), ListType(Utf8())) is False

    def test_array_vs_scalar_is_not_coercible(self):
        assert _is_coercible_difference(Array(Int64()), Int64()) is False
        assert _is_coercible_difference(Int64(), Array(Int64())) is False

    def test_nullable_array_vs_non_nullable_list_is_not_coercible(self):
        # Coercion does not remove nulls.
        assert _is_coercible_difference(Nullable(Array(Int64())), ListType(Int64())) is False


class TestOpenFrameChecking:
    """A declared column missing from an open inferred frame is not an error."""

    def _analysis(self, declared: FrameType, inferred: FrameType) -> FunctionAnalysis:
        return FunctionAnalysis(
            name="f",
            lineno=1,
            end_lineno=1,
            input_types={},
            declared_return_type=declared,
            inferred_return_type=inferred,
            errors=[],
        )

    def test_missing_required_column_passes_on_open_frame(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"id": Int64(), "qty": Int64()}),
                inferred=FrameType({"id": Int64()}, rest=RowVar("unnest")),
            )
        )
        assert result.passed is True

    def test_missing_required_column_fails_on_closed_frame(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"id": Int64(), "qty": Int64()}),
                inferred=FrameType({"id": Int64()}),
            )
        )
        assert result.passed is False
        assert any(isinstance(e, MissingColumn) for e in result.errors)

    def test_present_column_still_type_checked_on_open_frame(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"id": Int64()}),
                inferred=FrameType({"id": Utf8()}, rest=RowVar("unnest")),
            )
        )
        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)


class TestLeniencyTracing:
    """Leniency-mediated passes are recorded in ``CheckResult.leniency``.

    A pass that relies on a leniency rule (Unknown compatibility, open-frame
    missing-column skip, coerce-tolerated dtype difference) must be visible
    so it can show up in the golden files instead of silently masking a bug
    (the issue #47 failure mode). Leniency notes never affect ``passed``.
    """

    def _analysis(self, declared: FrameType, inferred: FrameType) -> FunctionAnalysis:
        return FunctionAnalysis(
            name="f",
            lineno=1,
            end_lineno=1,
            input_types={},
            declared_return_type=declared,
            inferred_return_type=inferred,
            errors=[],
        )

    def test_precise_pass_has_empty_leniency(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Int64()}),
                inferred=FrameType({"a": Int64()}),
            )
        )
        assert result.passed is True
        assert result.leniency == []

    def test_nullable_widening_is_not_leniency(self):
        """T <: Nullable[T] is a sound subtyping rule, not leniency."""
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Nullable(Int64())}),
                inferred=FrameType({"a": Int64()}),
            )
        )
        assert result.passed is True
        assert result.leniency == []

    def test_optional_declared_column_absent_is_not_leniency(self):
        """An Optional[T] column may be absent by declaration — not leniency."""
        from polypolarism.types import ColumnSpec

        result = check_function(
            self._analysis(
                declared=FrameType({"a": ColumnSpec(dtype=Int64(), required=False)}),
                inferred=FrameType({}),
            )
        )
        assert result.passed is True
        assert result.leniency == []

    def test_inferred_unknown_records_leniency(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Int64()}),
                inferred=FrameType({"a": Unknown()}),
            )
        )
        assert result.passed is True
        assert result.leniency == ["column 'a': passed via Unknown"]

    def test_declared_unknown_records_leniency(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Unknown()}),
                inferred=FrameType({"a": Utf8()}),
            )
        )
        assert result.passed is True
        assert result.leniency == ["column 'a': passed via Unknown"]

    def test_container_unknown_recursion_records_leniency(self):
        """List[Unknown] satisfying List[Int64] is an Unknown-mediated pass."""
        result = check_function(
            self._analysis(
                declared=FrameType({"a": ListType(Int64())}),
                inferred=FrameType({"a": ListType(Unknown())}),
            )
        )
        assert result.passed is True
        assert result.leniency == ["column 'a': passed via Unknown"]

    def test_open_frame_missing_column_records_leniency(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"id": Int64(), "qty": Int64()}),
                inferred=FrameType({"id": Int64()}, rest=RowVar("unnest")),
            )
        )
        assert result.passed is True
        assert result.leniency == ["column 'qty': not provably absent (open frame)"]

    def test_coerce_tolerated_difference_records_leniency(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"n": Int64()}, coerce=True),
                inferred=FrameType({"n": UInt32()}),
            )
        )
        assert result.passed is True
        assert result.leniency == ["column 'n': UInt32 -> Int64 via coerce"]

    def test_multiple_leniency_notes_accumulate(self):
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Int64(), "b": Utf8(), "c": Float64()}, coerce=True),
                inferred=FrameType(
                    {"a": Unknown(), "c": Int64()},
                    rest=RowVar("r"),
                ),
            )
        )
        assert result.passed is True
        assert result.leniency == [
            "column 'a': passed via Unknown",
            "column 'b': not provably absent (open frame)",
            "column 'c': Int64 -> Float64 via coerce",
        ]

    def test_leniency_does_not_affect_passed_on_failure(self):
        """Leniency notes coexist with real errors; passed reflects errors only."""
        result = check_function(
            self._analysis(
                declared=FrameType({"a": Int64(), "b": Utf8()}),
                inferred=FrameType({"a": Unknown(), "b": Int64()}),
            )
        )
        assert result.passed is False
        assert result.leniency == ["column 'a': passed via Unknown"]


class TestIsCoercibleDifference:
    """Unit tests for the coercion-compatibility helper.

    Direction-aware since issue #58: coercibility means "the INFERRED dtype
    is always castable to the DECLARED dtype" — probed against pandera
    ``coerce=True`` (polars 1.41.2 + pandera 0.31) with adversarial values.
    Value-DEPENDENT casts (Utf8 -> Int64 fails on "total", numeric ->
    Enum, ...) stay errors; the boundary pin is
    ``tests/fixtures/invalid/coerce_limits.py``.
    """

    def test_numeric_to_numeric_is_coercible(self):
        assert _is_coercible_difference(UInt32(), Int64()) is True

    def test_float_to_int_is_coercible(self):
        assert _is_coercible_difference(Float64(), Int64()) is True

    def test_formattable_to_string_is_coercible(self):
        # Issue #58 headline: anything probed always-castable -> String.
        from polypolarism.types import Boolean, Categorical, Date, Datetime, Decimal, Enum, Time

        assert _is_coercible_difference(UInt32(), Utf8()) is True
        assert _is_coercible_difference(Int64(), Utf8()) is True
        assert _is_coercible_difference(Float64(), Utf8()) is True
        assert _is_coercible_difference(Boolean(), Utf8()) is True
        assert _is_coercible_difference(Date(), Utf8()) is True
        assert _is_coercible_difference(Datetime(tz="UTC"), Utf8()) is True
        assert _is_coercible_difference(Time(), Utf8()) is True
        assert _is_coercible_difference(Categorical(), Utf8()) is True
        assert _is_coercible_difference(Enum(), Utf8()) is True
        assert _is_coercible_difference(Decimal(38, 2), Utf8()) is True

    def test_string_to_numeric_is_not_coercible(self):
        # Value-dependent ("total" fails) — the coerce_limits boundary pin.
        assert _is_coercible_difference(Utf8(), Int64()) is False
        assert _is_coercible_difference(Utf8(), Float64()) is False

    def test_duration_and_binary_to_string_are_not_coercible(self):
        # Probed: Duration -> String raises InvalidOperationError; Binary ->
        # String depends on the bytes being valid UTF-8 (value-dependent).
        from polypolarism.types import Binary, Duration

        assert _is_coercible_difference(Duration(), Utf8()) is False
        assert _is_coercible_difference(Binary(), Utf8()) is False

    def test_boolean_numeric_cells_are_coercible(self):
        # Probed: bool -> num is 0/1; num -> bool is "!= 0" (NaN included).
        from polypolarism.types import Boolean

        assert _is_coercible_difference(Boolean(), Int64()) is True
        assert _is_coercible_difference(Boolean(), Float64()) is True
        assert _is_coercible_difference(Int64(), Boolean()) is True
        assert _is_coercible_difference(Float64(), Boolean()) is True

    def test_temporal_always_cells_are_coercible(self):
        # Probed value-independent: date -> datetime (midnight), datetime ->
        # date (truncation), datetime -> time (extraction), time -> dur.
        from polypolarism.types import Date, Datetime, Duration, Time

        assert _is_coercible_difference(Date(), Datetime()) is True
        assert _is_coercible_difference(Date(), Datetime(tz="UTC")) is True
        assert _is_coercible_difference(Datetime(), Date()) is True
        assert _is_coercible_difference(Datetime(tz="UTC"), Time()) is True
        assert _is_coercible_difference(Time(), Duration()) is True
        # ... but the reverse Duration -> Time is a probed runtime error.
        assert _is_coercible_difference(Duration(), Time()) is False

    def test_categorical_always_cells_are_coercible(self):
        # Probed: any string is a valid category; Enum categories are strings.
        from polypolarism.types import Categorical, Enum

        assert _is_coercible_difference(Utf8(), Categorical()) is True
        assert _is_coercible_difference(Enum(), Categorical()) is True

    def test_enum_target_is_not_coercible(self):
        # Membership in the declared categories is value-dependent.
        from polypolarism.types import Categorical, Enum

        assert _is_coercible_difference(Utf8(), Enum()) is False
        assert _is_coercible_difference(Categorical(), Enum()) is False
        assert _is_coercible_difference(Int64(), Enum()) is False

    def test_nullable_inferred_to_non_nullable_declared_is_not_coercible(self):
        # Coercion casts values; it does not remove nulls.
        assert _is_coercible_difference(Nullable(UInt32()), Int64()) is False
        assert _is_coercible_difference(Nullable(UInt32()), Utf8()) is False

    def test_nullable_inferred_to_nullable_declared_is_coercible(self):
        assert _is_coercible_difference(Nullable(UInt32()), Nullable(Int64())) is True

    def test_non_nullable_inferred_to_nullable_declared_is_coercible(self):
        assert _is_coercible_difference(UInt32(), Nullable(Int64())) is True

    def test_datetime_tz_difference_is_coercible(self):
        # Probed (pandera + polars 1.41.2): ``Config.coerce = True`` casts
        # a tz-naive frame into a tz-aware schema and vice versa — flagging
        # the difference under coerce would be a false positive (issue #50).
        from polypolarism.types import Datetime

        assert _is_coercible_difference(Datetime(), Datetime(tz="UTC")) is True
        assert _is_coercible_difference(Datetime(tz="UTC"), Datetime()) is True
        assert _is_coercible_difference(Datetime(tz="UTC"), Datetime(tz="Asia/Tokyo")) is True

    def test_string_to_temporal_is_not_coercible(self):
        # Parsing strings into temporals is value-dependent.
        from polypolarism.types import Date, Datetime

        assert _is_coercible_difference(Utf8(), Datetime(tz="UTC")) is False
        assert _is_coercible_difference(Utf8(), Date()) is False


class TestCheckCoerce:
    """Config.coerce relaxes coercible dtype differences (issue #9)."""

    @staticmethod
    def _analysis(inferred_dtype, declared_dtype, coerce: bool) -> FunctionAnalysis:
        return FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={},
            declared_return_type=FrameType({"n": declared_dtype}, coerce=coerce),
            inferred_return_type=FrameType({"n": inferred_dtype}),
            errors=[],
        )

    def test_uint32_vs_int64_passes_with_coerce(self):
        result = check_function(self._analysis(UInt32(), Int64(), coerce=True))
        assert result.passed is True
        assert result.errors == []

    def test_uint32_vs_string_passes_with_coerce_and_records_leniency(self):
        # Issue #58: ``n: str`` declared for a ``pl.len()`` (UInt32) column
        # passes under coerce — and the pass is visible as a leniency note.
        result = check_function(self._analysis(UInt32(), Utf8(), coerce=True))
        assert result.passed is True
        assert result.leniency == ["column 'n': UInt32 -> Utf8 via coerce"]

    def test_uint32_vs_string_fails_without_coerce(self):
        result = check_function(self._analysis(UInt32(), Utf8(), coerce=False))
        assert result.passed is False

    def test_uint32_vs_int64_fails_without_coerce(self):
        result = check_function(self._analysis(UInt32(), Int64(), coerce=False))
        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)

    def test_nullable_mismatch_still_fails_with_coerce(self):
        # Coercion does not remove nulls — Nullable inferred vs required
        # non-nullable declared must still error.
        result = check_function(self._analysis(Nullable(UInt32()), Int64(), coerce=True))
        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)

    def test_nullable_both_sides_passes_with_coerce(self):
        result = check_function(self._analysis(Nullable(UInt32()), Nullable(Int64()), coerce=True))
        assert result.passed is True

    def test_non_numeric_mismatch_still_fails_with_coerce(self):
        result = check_function(self._analysis(Utf8(), Int64(), coerce=True))
        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)

    def test_coerce_does_not_excuse_missing_column(self):
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={},
            declared_return_type=FrameType({"n": Int64()}, coerce=True),
            inferred_return_type=FrameType({}),
            errors=[],
        )
        result = check_function(analysis)
        assert result.passed is False
        assert any(isinstance(e, MissingColumn) for e in result.errors)


class TestCheckSource:
    """Test source code checking."""

    def test_check_valid_source(self):
        """Check source with valid function."""
        source = textwrap.dedent("""
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class IdName(pa.DataFrameModel):
                id: int
                name: str

            def identity(
                data: DataFrame[IdName],
            ) -> DataFrame[IdName]:
                return data
        """)

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True

    def test_check_invalid_source_detects_mismatch(self):
        """Check source detects type mismatch."""
        source = textwrap.dedent("""
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class InSchema(pa.DataFrameModel):
                id: int

            class OutSchema(pa.DataFrameModel):
                id: int
                extra: str

            def wrong_return(
                data: DataFrame[InSchema],
            ) -> DataFrame[OutSchema]:
                return data
        """)

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, MissingColumn) for e in results[0].errors)

    def test_check_source_with_join(self):
        """Check source with join operation."""
        source = textwrap.dedent("""
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class L(pa.DataFrameModel):
                id: int
                name: str

            class R(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                id: int
                name: str
                value: pl.Float64

            def merge(
                left: DataFrame[L],
                right: DataFrame[R],
            ) -> DataFrame[Out]:
                return left.join(right, on="id", how="inner")
        """)

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True


class TestArrayCatEndToEnd:
    """End-to-end ``check_source`` runs for the issue #53 / #54 repros."""

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
            xs: pl.List(pl.Int64) = pa.Field()
            q: pl.Array(pl.Int64, 3) = pa.Field()
            c: pl.Categorical

        class Out(pa.DataFrameModel):
            r: int

            class Config:
                coerce = True
    """)

    def _check(self, body: str):
        source = self.HEADER + textwrap.dedent(f"""
            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(r={body})
        """)
        results = check_source(source)
        assert len(results) == 1
        return results[0]

    def test_arr_on_list_fails_ply012(self):
        result = self._check('pl.col("xs").arr.sum()')
        assert result.passed is False
        assert any("pple-wrong-namespace-dtype" in str(e) for e in result.errors)

    def test_list_on_array_fails_ply012(self):
        result = self._check('pl.col("q").list.sum()')
        assert result.passed is False
        assert any("pple-wrong-namespace-dtype" in str(e) for e in result.errors)

    def test_arr_sum_on_array_passes(self):
        # ``.arr.sum()`` on Array(Int64) -> Int64 satisfies the declared
        # ``int`` column (coerce on the Out schema is irrelevant here —
        # the dtypes match exactly).
        assert self._check('pl.col("q").arr.sum()').passed is True

    def test_list_sum_on_list_passes(self):
        assert self._check('pl.col("xs").list.sum()').passed is True

    def test_cat_on_int_fails_ply012(self):
        result = self._check('pl.col("a").cat.get_categories()')
        assert result.passed is False
        assert any(
            "pple-wrong-namespace-dtype" in str(e) and "SchemaError" in str(e)
            for e in result.errors
        )

    def test_cat_on_categorical_passes(self):
        source = self.HEADER + textwrap.dedent("""
            class CatsOut(pa.DataFrameModel):
                cats: str

            def f(df: DataFrame[In]) -> DataFrame[CatsOut]:
                return df.select(cats=pl.col("c").cat.get_categories())
        """)
        results = check_source(source)
        assert results[0].passed is True, results[0].errors

    def test_declared_array_inferred_list_fails_without_coerce(self):
        source = self.HEADER + textwrap.dedent("""
            class ArrOut(pa.DataFrameModel):
                q: pl.Array(pl.Int64, 3) = pa.Field()

            def f(df: DataFrame[In]) -> DataFrame[ArrOut]:
                return df.select(q=pl.col("xs"))
        """)
        results = check_source(source)
        assert results[0].passed is False
        assert any(isinstance(e, TypeDifference) for e in results[0].errors)

    def test_declared_array_inferred_list_passes_with_coerce(self):
        # Probed: pandera Config.coerce casts the List column into the
        # declared Array (succeeds whenever the widths line up).
        source = self.HEADER + textwrap.dedent("""
            class ArrOut(pa.DataFrameModel):
                q: pl.Array(pl.Int64, 3) = pa.Field()

                class Config:
                    coerce = True

            def f(df: DataFrame[In]) -> DataFrame[ArrOut]:
                return df.select(q=pl.col("xs"))
        """)
        results = check_source(source)
        assert results[0].passed is True, results[0].errors


class TestCoerceEndToEnd:
    """Issue #9 repro: pl.len() (UInt32) vs declared int under Config.coerce."""

    REPRO_TEMPLATE = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            g: str
            v: int
        {in_config}

        class Out(pa.DataFrameModel):
            g: str
            n: int
        {out_config}

        def agg(df: DataFrame[In]) -> DataFrame[Out]:
            return df.group_by("g").agg(n=pl.len())
    """

    COERCE_CONFIG = """
            class Config:
                coerce = True
    """

    def test_pl_len_vs_declared_int_passes_with_coerce(self):
        """The issue #9 repro validates fine at runtime — and now statically."""
        source = textwrap.dedent(
            self.REPRO_TEMPLATE.format(in_config=self.COERCE_CONFIG, out_config=self.COERCE_CONFIG)
        )

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].errors == []

    def test_pl_len_vs_declared_int_is_type_difference_without_coerce(self):
        """Regression: a present-but-mismatched column must be reported as a
        TypeDifference, not the misleading "Missing column"."""
        source = textwrap.dedent(self.REPRO_TEMPLATE.format(in_config="", out_config=""))

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(
            isinstance(e, TypeDifference)
            and e.column == "n"
            and e.inferred == UInt32()
            and e.declared == Int64()
            for e in results[0].errors
        )
        assert not any(isinstance(e, MissingColumn) for e in results[0].errors)

    def test_n_unique_vs_declared_int_passes_with_coerce(self):
        """Issue #9 sub-bug 2: an inferred UInt32 column under coerce."""
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class In(pa.DataFrameModel):
                g: str
                v: int

            class Out(pa.DataFrameModel):
                g: str
                n: int

                class Config:
                    coerce = True

            def agg(df: DataFrame[In]) -> DataFrame[Out]:
                return df.group_by("g").agg(n=pl.col("v").n_unique())
        """
        )

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True


JOIN_SUFFIX_SOURCE_TEMPLATE = """
    import polars as pl
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame

    class A(pa.DataFrameModel):
        g: int
        v: int

    class B(pa.DataFrameModel):
        g: int
        v: pl.Float64

    class Out(pa.DataFrameModel):
        g: int
        v: int
        {overlap_column}: pl.Float64

    def f(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[Out]:
        return a.join(b, on="g", suffix="_new")
"""


class TestCheckJoinSuffix:
    """#11: declared schema must follow the actual ``suffix=`` argument."""

    def test_declared_v_new_passes_with_custom_suffix(self):
        source = textwrap.dedent(JOIN_SUFFIX_SOURCE_TEMPLATE.format(overlap_column="v_new"))

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_declared_v_right_fails_with_custom_suffix(self):
        source = textwrap.dedent(JOIN_SUFFIX_SOURCE_TEMPLATE.format(overlap_column="v_right"))

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, MissingColumn) for e in results[0].errors)


class TestCheckResult:
    """Test CheckResult data class."""

    def test_check_result_function_name(self):
        """CheckResult includes function name."""
        analysis = FunctionAnalysis(
            name="my_function",
            lineno=1,
            end_lineno=1,
            input_types={},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=FrameType({"id": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.function_name == "my_function"

    def test_check_result_repr(self):
        """CheckResult has readable repr."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=FrameType({"id": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert "process" in repr(result)


class TestNoReturnTypeInferred:
    """Test cases where return type cannot be inferred."""

    def test_no_inferred_type_fails(self):
        """If no return type is inferred, check fails."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            end_lineno=1,
            input_types={"data": FrameType({"id": Int64()})},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=None,  # Could not infer
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any("infer" in str(e).lower() for e in result.errors)

    def test_inference_failure_message_carries_family_code(self):
        """InferenceFailure renders with the [pple-return-type] family code (issue #70)."""
        from polypolarism.checker import InferenceFailure

        message = str(InferenceFailure("Could not infer return type"))

        assert message == "[pple-return-type] Could not infer return type"


class TestIssue14TrueDivisionEndToEnd:
    """Issue #14 repro: int / int is Float64, int // int stays Int64.

    Schemas here have no ``Config``, so ``coerce`` is False — dtype
    differences are real errors.
    """

    SOURCE_TEMPLATE = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
            b: int

        class Out(pa.DataFrameModel):
            r: {declared}

        def f(df: DataFrame[In]) -> DataFrame[Out]:
            return df.select(r=pl.col("a") {op} pl.col("b"))
    """

    def _check(self, declared: str, op: str):
        source = textwrap.dedent(self.SOURCE_TEMPLATE.format(declared=declared, op=op))
        return check_source(source)[0]

    def test_truediv_declared_float_passes(self):
        """The killed false positive: a correct float declaration was rejected."""
        result = self._check("float", "/")
        assert result.passed is True, result.errors

    def test_floordiv_declared_int_passes(self):
        result = self._check("int", "//")
        assert result.passed is True, result.errors

    def test_truediv_declared_int_fails(self):
        """The killed false negative: a wrong int declaration was accepted."""
        result = self._check("int", "/")
        assert result.passed is False
        assert any(
            isinstance(e, TypeDifference)
            and e.column == "r"
            and e.declared == Int64()
            and e.inferred == Float64()
            for e in result.errors
        )


class TestIssue18NullabilityEndToEnd:
    """Issue #18 repro: nullability flows through expressions.

    ``x`` is declared ``pa.Field(nullable=True)``; ``a + x`` can hold
    nulls, so declaring the result non-nullable must fail (pandera
    rejects the nulls at runtime) and declaring it nullable must pass.
    """

    SOURCE_TEMPLATE = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
            x: int = pa.Field(nullable=True)

        class Out(pa.DataFrameModel):
            z: {declared}

        def f(df: DataFrame[In]) -> DataFrame[Out]:
            return df.select(z=pl.col("a") + pl.col("x"))
    """

    def _check(self, declared: str):
        source = textwrap.dedent(self.SOURCE_TEMPLATE.format(declared=declared))
        return check_source(source)[0]

    def test_sum_with_nullable_operand_declared_nonnull_fails(self):
        """The killed false negative: nulls flow into a non-nullable column."""
        result = self._check("int")
        assert result.passed is False
        assert any(
            isinstance(e, TypeDifference)
            and e.column == "z"
            and e.declared == Int64()
            and e.inferred == Nullable(Int64())
            for e in result.errors
        )

    def test_sum_with_nullable_operand_declared_nullable_passes(self):
        result = self._check("int = pa.Field(nullable=True)")
        assert result.passed is True, result.errors


class TestIssue30IncompatibleArithmeticEndToEnd:
    """Issue #30 repro: ``String + Int64`` arithmetic is flagged with pple-incompatible-operands.

    polars raises InvalidOperationError at runtime for such pairs; the
    output column registers as Unknown so the pple-incompatible-operands is the only error.
    Combinations polars permits (concat, temporal arithmetic, numeric
    promotion) must keep passing.
    """

    HEADER = textwrap.dedent(
        """
        import polars as pl
        import pandera.polars as pa
        from datetime import date, timedelta
        from pandera.typing.polars import DataFrame
        """
    )

    def test_string_plus_int_fails_with_exactly_one_ply009(self):
        source = self.HEADER + textwrap.dedent(
            """
            class In(pa.DataFrameModel):
                s: str
                n: int

            class Out(pa.DataFrameModel):
                r: int

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(r=pl.col("s") + pl.col("n"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is False
        assert len(result.errors) == 1, result.errors
        assert "pple-incompatible-operands" in str(result.errors[0])

    def test_string_concat_declared_str_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class In(pa.DataFrameModel):
                a: str
                b: str

            class Out(pa.DataFrameModel):
                r: str

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(r=pl.col("a") + pl.col("b"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors

    def test_date_difference_declared_duration_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class In(pa.DataFrameModel):
                start: date
                end: date

            class Out(pa.DataFrameModel):
                span: timedelta

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(span=pl.col("end") - pl.col("start"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors

    def test_duration_scaled_by_int_declared_duration_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class In(pa.DataFrameModel):
                gap: timedelta

            class Out(pa.DataFrameModel):
                doubled: pl.Duration

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(doubled=pl.col("gap") * 2)
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors

    def test_int_plus_float_declared_float_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class In(pa.DataFrameModel):
                n: int
                x: float

            class Out(pa.DataFrameModel):
                r: float

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(r=pl.col("n") + pl.col("x"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors


class TestIssue19StrToIntegerEndToEnd:
    """Issue #19: ``str.to_integer()`` infers Int64, not Unknown.

    The Unknown fallback used to mask real dtype mismatches — declaring
    the parsed column ``str`` was silently accepted. With the precise
    inference it must now be a TypeDifference (no ``coerce`` here).
    """

    SOURCE_TEMPLATE = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            s: str

        class Out(pa.DataFrameModel):
            n: {declared}

        def f(df: DataFrame[In]) -> DataFrame[Out]:
            return df.select(n=pl.col("s").str.to_integer())
    """

    def _check(self, declared: str):
        source = textwrap.dedent(self.SOURCE_TEMPLATE.format(declared=declared))
        return check_source(source)[0]

    def test_to_integer_declared_int_passes(self):
        result = self._check("int")
        assert result.passed is True, result.errors

    def test_to_integer_declared_str_fails(self):
        """The killed false negative: Unknown used to accept any declaration."""
        result = self._check("str")
        assert result.passed is False
        assert any(
            isinstance(e, TypeDifference)
            and e.column == "n"
            and e.declared == Utf8()
            and e.inferred == Int64()
            for e in result.errors
        )


class TestIssue23AggExprEndToEnd:
    """Issue #23 repros: Expr.len() and .filter(...).<agg>() in agg context.

    All schemas use ``strict = True`` + ``coerce = True`` like the issue's
    reproduction — the output columns must be inferred *precisely* (UInt32 /
    Int64), not Unknown, and the functions must pass.
    """

    HEADER = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            g: str
            v: int

            class Config:
                strict = True
                coerce = True
    """

    def _check(self, body: str):
        # HEADER and the per-test bodies have different leading indents, so
        # they must be dedented separately before concatenation.
        source = textwrap.dedent(self.HEADER) + textwrap.dedent(body)
        return check_source(source)[0]

    def test_agg_sum_alias_passes(self):
        """Regression guard: the plain sum repro already worked."""
        result = self._check("""
            class OSum(pa.DataFrameModel):
                g: str
                s: int

                class Config:
                    strict = True
                    coerce = True

            def agg_sum(df: DataFrame[In]) -> DataFrame[OSum]:
                return df.group_by("g").agg(pl.col("v").sum().alias("s"))
        """)
        assert result.passed is True, result.errors

    def test_agg_expr_len_alias_passes(self):
        """``pl.col("v").len()`` infers UInt32; coerce bridges to Int64."""
        result = self._check("""
            class OLen(pa.DataFrameModel):
                g: str
                n: int

                class Config:
                    strict = True
                    coerce = True

            def agg_len(df: DataFrame[In]) -> DataFrame[OLen]:
                return df.group_by("g").agg(pl.col("v").len().alias("n"))
        """)
        assert result.passed is True, result.errors

    def test_agg_filter_sum_alias_passes(self):
        """Conditional aggregation: ``filter(...).sum()`` infers Int64."""
        result = self._check("""
            class OFSum(pa.DataFrameModel):
                g: str
                fs: int

                class Config:
                    strict = True
                    coerce = True

            def agg_filter_sum(df: DataFrame[In]) -> DataFrame[OFSum]:
                return df.group_by("g").agg(
                    pl.col("v").filter(pl.col("v") > 0).sum().alias("fs")
                )
        """)
        assert result.passed is True, result.errors


class TestSemiAntiGatherEndToEnd:
    """Issue #15 repro: semi/anti joins and gather_every are schema-preserving."""

    ISSUE_SOURCE = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class L(pa.DataFrameModel):
            id: int
            v: int
            class Config:
                coerce = True

        class R(pa.DataFrameModel):
            id: int
            class Config:
                coerce = True

        def ok_semi(left: DataFrame[L], right: DataFrame[R]) -> DataFrame[L]:
            return left.join(right, on="id", how="semi")

        def ok_anti(left: DataFrame[L], right: DataFrame[R]) -> DataFrame[L]:
            return left.join(right, on="id", how="anti")

        def ok_gather_every(left: DataFrame[L]) -> DataFrame[L]:
            return left.gather_every(2)
    """)

    def test_issue_15_repro_functions_pass(self):
        results = check_source(self.ISSUE_SOURCE)

        assert len(results) == 3
        by_name = {r.function_name: r for r in results}
        for name in ("ok_semi", "ok_anti", "ok_gather_every"):
            assert by_name[name].passed is True, (name, by_name[name].errors)

    def test_semi_join_missing_key_fails(self):
        """Key validation still applies under how='semi'."""
        source = textwrap.dedent("""
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class L(pa.DataFrameModel):
                id: int

            class R(pa.DataFrameModel):
                id: int

            def bad(left: DataFrame[L], right: DataFrame[R]) -> DataFrame[L]:
                return left.join(right, on="missing", how="semi")
        """)

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-join-key" in str(e) for e in results[0].errors)


class TestJoinCoalesceCrossEndToEnd:
    """Issues #24/#26 repros: full-join key coalescing and cross joins."""

    ISSUE_24_SOURCE = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class L(pa.DataFrameModel):
            id: int
            x: int
            class Config:
                coerce = True

        class R(pa.DataFrameModel):
            id: int
            y: int
            class Config:
                coerce = True

        class FullOut(pa.DataFrameModel):
            id: int                                 # coalesced key — non-null
            x: int = pa.Field(nullable=True)        # from left  — nullable
            y: int = pa.Field(nullable=True)        # from right — nullable
            class Config:
                strict = True
                coerce = True

        def full_coalesce(l: DataFrame[L], r: DataFrame[R]) -> DataFrame[FullOut]:
            return l.join(r, on="id", how="full", coalesce=True)

        class InnerOut(pa.DataFrameModel):
            id: int
            x: int
            y: int
            class Config:
                strict = True
                coerce = True

        def inner_join(l: DataFrame[L], r: DataFrame[R]) -> DataFrame[InnerOut]:
            return l.join(r, on="id", how="inner")
    """)

    def test_issue_24_repro_functions_pass(self):
        results = check_source(self.ISSUE_24_SOURCE)

        assert len(results) == 2
        by_name = {r.function_name: r for r in results}
        for name in ("full_coalesce", "inner_join"):
            assert by_name[name].passed is True, (name, by_name[name].errors)

    def test_full_join_without_coalesce_keeps_both_keys(self):
        """Corrected default: full join keeps id (nullable) AND id_right."""
        source = textwrap.dedent("""
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class L(pa.DataFrameModel):
                id: int
                x: int
                class Config:
                    coerce = True

            class R(pa.DataFrameModel):
                id: int
                y: int
                class Config:
                    coerce = True

            class FullDefaultOut(pa.DataFrameModel):
                id: int = pa.Field(nullable=True)
                x: int = pa.Field(nullable=True)
                id_right: int = pa.Field(nullable=True)
                y: int = pa.Field(nullable=True)
                class Config:
                    strict = True
                    coerce = True

            def full_default(l: DataFrame[L], r: DataFrame[R]) -> DataFrame[FullDefaultOut]:
                return l.join(r, on="id", how="full")
        """)

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    ISSUE_26_SOURCE = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class L(pa.DataFrameModel):
            id: int
            x: int
            class Config:
                coerce = True

        class R(pa.DataFrameModel):
            rid: int
            y: int
            class Config:
                coerce = True

        class CrossOut(pa.DataFrameModel):
            id: int
            x: int
            rid: int
            y: int
            class Config:
                strict = True
                coerce = True

        def ok_cross_join(l: DataFrame[L], r: DataFrame[R]) -> DataFrame[CrossOut]:
            return l.join(r, how="cross")
    """)

    def test_issue_26_repro_passes(self):
        results = check_source(self.ISSUE_26_SOURCE)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_cross_join_with_collision_gets_suffix(self):
        """A shared column name lands as v_right in the cross-join output."""
        source = textwrap.dedent("""
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class L(pa.DataFrameModel):
                id: int
                v: int
                class Config:
                    coerce = True

            class R(pa.DataFrameModel):
                v: str
                class Config:
                    coerce = True

            class CrossOut(pa.DataFrameModel):
                id: int
                v: int
                v_right: str
                class Config:
                    strict = True
                    coerce = True

            def cross_collision(l: DataFrame[L], r: DataFrame[R]) -> DataFrame[CrossOut]:
                return l.join(r, how="cross")
        """)

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestIssue21SeqVariantsEndToEnd:
    """Issue #21 repro: with_columns_seq / select_seq infer like the
    non-seq forms (schema semantics identical; only evaluation order
    differs)."""

    ISSUE_SOURCE = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
            b: int
            class Config:
                coerce = True

        class Out(pa.DataFrameModel):
            a: int
            c: int
            class Config:
                strict = True
                coerce = True

        def ok_seq_strings(df: DataFrame[In]) -> DataFrame[Out]:
            return df.with_columns_seq(c=pl.col("a") + pl.col("b")).select_seq("a", "c")

        def ok_seq_exprs(df: DataFrame[In]) -> DataFrame[Out]:
            return df.with_columns_seq(c=pl.col("a") + pl.col("b")).select_seq(
                pl.col("a"), pl.col("c")
            )
    """)

    def test_issue_21_repro_functions_pass(self):
        results = check_source(self.ISSUE_SOURCE)

        assert len(results) == 2
        by_name = {r.function_name: r for r in results}
        for name in ("ok_seq_strings", "ok_seq_exprs"):
            assert by_name[name].passed is True, (name, by_name[name].errors)


class TestIssue20PlAllExcludeEndToEnd:
    """Issue #20 repro: pl.all() / pl.exclude(...) inside select(), with
    the already-working cs.* / with_columns forms as regression guards."""

    ISSUE_SOURCE = textwrap.dedent("""
        import polars as pl
        import polars.selectors as cs
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class TC(pa.DataFrameModel):
            a: int
            b: int
            name: str
            class Config:
                coerce = True

        class AB(pa.DataFrameModel):
            a: int
            b: int
            class Config:
                strict = True
                coerce = True

        def sel_all(df: DataFrame[TC]) -> DataFrame[TC]:
            return df.select(pl.all())

        def sel_exclude(df: DataFrame[TC]) -> DataFrame[AB]:
            return df.select(pl.exclude("name"))

        def sel_cs(df: DataFrame[TC]) -> DataFrame[AB]:
            return df.select(cs.numeric())

        def with_cols_all(df: DataFrame[TC]) -> DataFrame[TC]:
            return df.with_columns(pl.all())
    """)

    def test_issue_20_repro_functions_pass(self):
        results = check_source(self.ISSUE_SOURCE)

        assert len(results) == 4
        by_name = {r.function_name: r for r in results}
        for name in ("sel_all", "sel_exclude", "sel_cs", "with_cols_all"):
            assert by_name[name].passed is True, (name, by_name[name].errors)


class TestIssue22SelectConstantEndToEnd:
    """Issue #22 repro: select(KEY) with a module-level constant infers
    the same schema as the literal form, satisfying a strict Out."""

    ISSUE_SOURCE = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
            b: int
            class Config:
                coerce = True

        class OutA(pa.DataFrameModel):
            a: int
            class Config:
                strict = True
                coerce = True

        KEY = "a"

        def ok_select_var(df: DataFrame[In]) -> DataFrame[OutA]:
            return df.select(KEY)

        def ok_select_seq_var(df: DataFrame[In]) -> DataFrame[OutA]:
            return df.select_seq(KEY)
    """)

    def test_issue_22_repro_functions_pass(self):
        results = check_source(self.ISSUE_SOURCE)

        assert len(results) == 2
        by_name = {r.function_name: r for r in results}
        for name in ("ok_select_var", "ok_select_seq_var"):
            assert by_name[name].passed is True, (name, by_name[name].errors)


class TestImplicitListAggEndToEnd:
    """Issue #27 repro: ``agg(vs=pl.col("v"))`` must type as List(Int64)."""

    SOURCE = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            k: str
            v: int

        class Listed(pa.DataFrameModel):
            k: str
            vs: pl.List(pl.Int64) = pa.Field()

            class Config:
                strict = True

        @pa.check_types
        def agg_to_list(df: DataFrame[In]) -> DataFrame[Listed]:
            return df.group_by("k").agg(vs=pl.col("v"))
    """

    def test_agg_to_list_passes_strict_schema(self):
        results = check_source(textwrap.dedent(self.SOURCE))

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert results[0].errors == []

    def test_element_dtype_declared_fails(self):
        """Declaring the old (wrong) element dtype is a TypeDifference."""
        source = textwrap.dedent(self.SOURCE).replace(
            "vs: pl.List(pl.Int64) = pa.Field()", "vs: int"
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, TypeDifference) and e.column == "vs" for e in results[0].errors)


class TestFrameLiteralEndToEnd:
    """Issue #25 repros: functions returning frames built from scratch."""

    HEADER = textwrap.dedent(
        """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Empty(pa.DataFrameModel):
                pass
        """
    )

    def test_pure_literal_passes_strict_schema(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Lit(pa.DataFrameModel):
                a: int

                class Config:
                    strict = True

            @pa.check_types
            def pure_literal(df: DataFrame[Empty]) -> DataFrame[Lit]:
                return pl.DataFrame({"a": [1, 2, 3]})
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert results[0].errors == []

    def test_build_calendar_passes_strict_schema(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Cal(pa.DataFrameModel):
                d: pl.Date
                year: pl.Int32

                class Config:
                    strict = True

            @pa.check_types
            def build_calendar(df: DataFrame[Empty]) -> DataFrame[Cal]:
                cal = pl.DataFrame(
                    {"d": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 3), eager=True)}
                )
                return cal.with_columns(year=pl.col("d").dt.year())
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert results[0].errors == []

    def test_wrong_dtype_in_literal_fails(self):
        """A literal whose dtype mismatches the declared schema is caught."""
        source = self.HEADER + textwrap.dedent(
            """
            class Lit(pa.DataFrameModel):
                a: str

            @pa.check_types
            def pure_literal(df: DataFrame[Empty]) -> DataFrame[Lit]:
                return pl.DataFrame({"a": [1, 2, 3]})
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, TypeDifference) and e.column == "a" for e in results[0].errors)

    def test_lazy_literal_where_dataframe_declared_is_ply032(self):
        """A ``pl.LazyFrame({...})`` literal returned from a function declared
        ``-> DataFrame[...]`` trips the eager/lazy mismatch check."""
        source = self.HEADER + textwrap.dedent(
            """
            class Lit(pa.DataFrameModel):
                a: int

            @pa.check_types
            def lazy_literal(df: DataFrame[Empty]) -> DataFrame[Lit]:
                return pl.LazyFrame({"a": [1, 2, 3]})
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-eager-lazy-mismatch" in str(e) for e in results[0].errors)

    def test_lazy_literal_collected_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Lit(pa.DataFrameModel):
                a: int

            @pa.check_types
            def lazy_literal(df: DataFrame[Empty]) -> DataFrame[Lit]:
                return pl.LazyFrame({"a": [1, 2, 3]}).collect()
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert results[0].errors == []


class TestFilterPredicateEndToEnd:
    """Issue #28 repro: a non-boolean filter predicate must fail the check."""

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class AInt(pa.DataFrameModel):
            a: int

            class Config:
                coerce = True
    """)

    def test_nonbool_predicate_fails_with_ply008(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def bug_filter_nonbool(df: DataFrame[AInt]) -> DataFrame[AInt]:
                return df.filter(pl.col("a"))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-non-boolean-predicate" in str(e) for e in results[0].errors)

    def test_boolean_predicate_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class WithFlag(pa.DataFrameModel):
                a: int
                flag: bool

            @pa.check_types
            def keep_flagged(df: DataFrame[WithFlag]) -> DataFrame[WithFlag]:
                return df.filter(pl.col("flag"))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestSortKeyEndToEnd:
    """Issue #29 repro: sorting by a non-existent column must fail the check."""

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class AInt(pa.DataFrameModel):
            a: int

            class Config:
                coerce = True
    """)

    def test_missing_sort_key_fails_with_ply007(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def bug_sort_nonexistent(df: DataFrame[AInt]) -> DataFrame[AInt]:
                return df.sort("ghost")
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-column-not-found" in str(e) for e in results[0].errors)

    def test_existing_sort_key_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def sort_existing(df: DataFrame[AInt]) -> DataFrame[AInt]:
                return df.sort("a")
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestWhenConditionEndToEnd:
    """Issue #37 repro: a non-Boolean ``pl.when`` condition must fail the check.

    Probed (polars 1.41.2): ``pl.when(pl.col("a"))`` with ``a: Int64`` raises
    ``SchemaError: invalid series dtype: expected `Boolean`, got `i64```.
    """

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
            flag: bool
    """)

    def test_nonbool_when_condition_fails_with_ply008(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Out(pa.DataFrameModel):
                a: int
                flag: bool
                x: int

            @pa.check_types
            def bug_when_nonbool(df: DataFrame[In]) -> DataFrame[Out]:
                return df.with_columns(x=pl.when(pl.col("a")).then(1).otherwise(0))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(
            "pple-non-boolean-predicate" in str(e) and "when" in str(e) for e in results[0].errors
        )

    def test_boolean_when_condition_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Out(pa.DataFrameModel):
                a: int
                flag: bool
                x: int

            @pa.check_types
            def ok_when(df: DataFrame[In]) -> DataFrame[Out]:
                return df.with_columns(x=pl.when(pl.col("flag")).then(1).otherwise(0))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestWhenSupertypeEndToEnd:
    """Issue #40 repro: mixed-dtype when/then/otherwise branches infer the
    polars supertype, so wrong declarations fail and the right one passes.

    Probed (polars 1.41.2):
    ``pl.when(pl.col("a") > 0).then(pl.lit(1)).otherwise(pl.lit("x"))``
    -> ``Schema({'literal': String})``.
    """

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
    """)

    def _check(self, declared: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            class Out(pa.DataFrameModel):
                a: int
                x: {declared}

            @pa.check_types
            def mixed_branches(df: DataFrame[In]) -> DataFrame[Out]:
                return df.with_columns(
                    x=pl.when(pl.col("a") > 0).then(pl.lit(1)).otherwise(pl.lit("x"))
                )
        """
        )
        results = check_source(source)
        assert len(results) == 1
        return results[0]

    def test_str_declaration_passes(self):
        result = self._check("str")
        assert result.passed is True, result.errors

    def test_int_declaration_fails_with_type_difference(self):
        result = self._check("int")
        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)

    def test_bool_declaration_fails_with_type_difference(self):
        result = self._check("bool")
        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)

    def test_float_declaration_fails_with_type_difference(self):
        result = self._check("float")
        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)


class TestUnpivotSupertypeEndToEnd:
    """Issue #41 repro: mixed-dtype unpivot value columns supertype instead
    of raising pple-unpivot.

    Probed (polars 1.41.2): ``df.unpivot(index="id", on=["a", "s"])`` with
    ``a: Int64``, ``s: String`` produces
    ``Schema({'id': Int64, 'variable': String, 'value': String})``.
    """

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class Wide(pa.DataFrameModel):
            id: int
            a: int
            s: str
    """)

    def test_mixed_value_columns_pass_with_str_value(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Long(pa.DataFrameModel):
                id: int
                variable: str
                value: str

            @pa.check_types
            def melt(df: DataFrame[Wide]) -> DataFrame[Long]:
                return df.unpivot(index="id", on=["a", "s"])
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_mixed_value_columns_fail_int_value_declaration(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Long(pa.DataFrameModel):
                id: int
                variable: str
                value: int

            @pa.check_types
            def melt(df: DataFrame[Wide]) -> DataFrame[Long]:
                return df.unpivot(index="id", on=["a", "s"])
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, TypeDifference) for e in results[0].errors)


class TestShiftFillValueEndToEnd:
    """Issue #43 repro: ``shift(1, fill_value=0)`` is non-null at runtime, so
    a non-nullable declaration must pass; bare ``shift(1)`` stays Nullable.

    Probed (polars 1.41.2): ``[1, 2, 3].shift(1, fill_value=0)`` ->
    ``[0, 1, 2]`` with null_count 0 and dtype Int64.
    """

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
    """)

    def test_fill_value_passes_non_nullable_declaration(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def shifted(df: DataFrame[In]) -> DataFrame[In]:
                return df.with_columns(pl.col("a").shift(1, fill_value=0))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_no_fill_value_still_fails_non_nullable_declaration(self):
        # Regression guard: without a fill the head slot is null, so the
        # non-nullable declaration must keep failing.
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def shifted(df: DataFrame[In]) -> DataFrame[In]:
                return df.with_columns(pl.col("a").shift(1))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, TypeDifference) for e in results[0].errors)

    def test_cross_dtype_fill_supertype_checks(self):
        # shift(1, fill_value="x") on Int64 -> String (probed).
        source = self.HEADER + textwrap.dedent(
            """
            class Out(pa.DataFrameModel):
                a: str

            @pa.check_types
            def shifted(df: DataFrame[In]) -> DataFrame[Out]:
                return df.with_columns(pl.col("a").shift(1, fill_value="x"))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestOverMappingStrategyEndToEnd:
    """Issue #45 repro: ``over(..., mapping_strategy="join")`` is List.

    Probed (polars 1.41.2): a length-preserving expression under "join"
    gathers each partition's values into a List per row; an aggregation
    broadcasts its scalar unchanged.
    """

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class AG(pa.DataFrameModel):
            a: int
            g: str

            class Config:
                coerce = True
    """)

    def test_join_strategy_passes_list_declaration(self):
        source = self.HEADER + textwrap.dedent(
            """
            class OverJoinOut(pa.DataFrameModel):
                o: pl.List(pl.Int64)

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def ok_over_join(df: DataFrame[AG]) -> DataFrame[OverJoinOut]:
                return df.select(o=pl.col("a").over("g", mapping_strategy="join"))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_default_strategy_still_fails_list_declaration(self):
        # Regression guard: the default group_to_rows stays scalar, so the
        # List declaration must keep failing.
        source = self.HEADER + textwrap.dedent(
            """
            class OverJoinOut(pa.DataFrameModel):
                o: pl.List(pl.Int64)

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def over_default(df: DataFrame[AG]) -> DataFrame[OverJoinOut]:
                return df.select(o=pl.col("a").over("g"))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, TypeDifference) for e in results[0].errors)

    def test_join_aggregation_passes_scalar_declaration(self):
        source = self.HEADER + textwrap.dedent(
            """
            class ScalarOut(pa.DataFrameModel):
                o: int

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def over_join_sum(df: DataFrame[AG]) -> DataFrame[ScalarOut]:
                return df.select(o=pl.col("a").sum().over("g", mapping_strategy="join"))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestDiffTemporalEndToEnd:
    """Issue #46 repro: ``diff()`` on a Date column yields Duration.

    Probed (polars 1.41.2): ``pl.col(date).diff()`` -> Duration with a null
    head slot, so the declared column must be ``pl.Duration`` +
    ``pa.Field(nullable=True)``. polypolarism's parameterless ``Duration()``
    matches polars' time-unit-parametrised Duration.
    """

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class Dates(pa.DataFrameModel):
            d: pl.Date

            class Config:
                coerce = True
    """)

    def test_date_diff_passes_nullable_duration_declaration(self):
        source = self.HEADER + textwrap.dedent(
            """
            class DurOut(pa.DataFrameModel):
                df: pl.Duration = pa.Field(nullable=True)

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def ok_date_diff(df: DataFrame[Dates]) -> DataFrame[DurOut]:
                return df.select(df=pl.col("d").diff())
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_date_diff_mismatch_now_names_duration(self):
        # A non-nullable declaration still fails — but on nullability only:
        # the inferred side must read Duration?, not Date?.
        source = self.HEADER + textwrap.dedent(
            """
            class DurOut(pa.DataFrameModel):
                df: pl.Duration

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def date_diff(df: DataFrame[Dates]) -> DataFrame[DurOut]:
                return df.select(df=pl.col("d").diff())
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        diffs = [e for e in results[0].errors if isinstance(e, TypeDifference)]
        assert len(diffs) == 1
        assert "Duration[us]?" in str(diffs[0])
        assert "Date" not in str(diffs[0])


class TestUniqueSubsetEndToEnd:
    """Issue #35 repro: ``unique(subset=[...])`` with a ghost column must fail."""

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class ASB(pa.DataFrameModel):
            a: int
            s: str
            b: int

            class Config:
                coerce = True
    """)

    def test_missing_subset_column_fails_with_ply014(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def bug_unique_subset_ghost(df: DataFrame[ASB]) -> DataFrame[ASB]:
                return df.unique(subset=["ghost"])
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-column-not-found" in str(e) for e in results[0].errors)

    def test_existing_subset_column_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def unique_by_a(df: DataFrame[ASB]) -> DataFrame[ASB]:
                return df.unique(subset=["a"])
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_bare_unique_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def unique_all(df: DataFrame[ASB]) -> DataFrame[ASB]:
                return df.unique()
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestDecimalCastEndToEnd:
    """Issue #38 repro: a correctly-declared Decimal(10, 2) output must pass."""

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class XIn(pa.DataFrameModel):
            x: int

            class Config:
                coerce = True
    """)

    def test_decimal_cast_matches_declared_precision_scale(self):
        source = self.HEADER + textwrap.dedent(
            """
            class DecOut(pa.DataFrameModel):
                d: pl.Decimal(10, 2)

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def ok_decimal_cast(df: DataFrame[XIn]) -> DataFrame[DecOut]:
                return df.select(d=pl.col("x").cast(pl.Decimal(10, 2)))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert results[0].errors == []

    def test_decimal_cast_with_wrong_scale_still_fails(self):
        source = self.HEADER + textwrap.dedent(
            """
            class DecOut(pa.DataFrameModel):
                d: pl.Decimal(10, 2)

                class Config:
                    strict = True

            @pa.check_types
            def bad_decimal_cast(df: DataFrame[XIn]) -> DataFrame[DecOut]:
                return df.select(d=pl.col("x").cast(pl.Decimal(10, 4)))
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, TypeDifference) and e.column == "d" for e in results[0].errors)


class TestFrameLiteralVariableValuesEndToEnd:
    """Issue #39 repro: a frame-literal column whose values come from a
    constant binding must type like the literal-list case and join cleanly."""

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        NAMES = ["x", "y", "z"]

        class Ev(pa.DataFrameModel):
            name: str
            v: int

            class Config:
                coerce = True

        class Out(pa.DataFrameModel):
            step: int
            name: str
            v: int = pa.Field(nullable=True)

            class Config:
                strict = True
                coerce = True
    """)

    def test_via_variable_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def via_variable(ev: DataFrame[Ev]) -> DataFrame[Out]:
                sk = pl.DataFrame({"step": [1, 2, 3], "name": NAMES})
                return sk.join(ev, on="name", how="left")
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert results[0].errors == []

    def test_via_literal_still_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def via_literal(ev: DataFrame[Ev]) -> DataFrame[Out]:
                sk = pl.DataFrame({"step": [1, 2, 3], "name": ["x", "y", "z"]})
                return sk.join(ev, on="name", how="left")
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestUnknownJoinKeyEndToEnd:
    """Issue #39b: a genuinely-Unknown join key (values from an
    unresolvable variable) must not be reported as a dtype mismatch."""

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class Ev(pa.DataFrameModel):
            name: str
            v: int

            class Config:
                coerce = True

        class Out(pa.DataFrameModel):
            step: int
            name: str
            v: int = pa.Field(nullable=True)

            class Config:
                strict = True
                coerce = True
    """)

    def test_unknown_key_join_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def via_dynamic_values(ev: DataFrame[Ev]) -> DataFrame[Out]:
                names = load_names()
                sk = pl.DataFrame({"step": [1, 2, 3], "name": names})
                return sk.join(ev, on="name", how="left")
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert not any("pple-join-key" in str(e) for e in results[0].errors)

    def test_genuine_key_dtype_mismatch_still_fails(self):
        source = self.HEADER + textwrap.dedent(
            """
            @pa.check_types
            def bad_key_join(ev: DataFrame[Ev]) -> DataFrame[Out]:
                sk = pl.DataFrame({"step": [1, 2, 3], "name": [10, 20, 30]})
                return sk.join(ev, on="name", how="left")
        """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-join-key" in str(e) for e in results[0].errors)


class TestIssue48StructRenameFieldsEndToEnd:
    """Issue #48 repro: ``struct.rename_fields`` + ``unnest`` must satisfy a
    strict declared schema with the new field names."""

    SOURCE = textwrap.dedent(
        """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class SX(pa.DataFrameModel):
            s: pl.Struct({"x": pl.Int64, "y": pl.Int64})

            class Config:
                coerce = True

        class PQ(pa.DataFrameModel):
            p: int
            q: int

            class Config:
                strict = True
                coerce = True

        @pa.check_types
        def ok_struct_rename_fields(df: DataFrame[SX]) -> DataFrame[PQ]:
            return df.select(pl.col("s").struct.rename_fields(["p", "q"])).unnest("s")
    """
    )

    def test_repro_passes(self):
        results = check_source(self.SOURCE)
        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestIssue49CumSumOnStringEndToEnd:
    """Issue #49 repro: ``cum_sum`` on a String column raises
    InvalidOperationError at runtime — must be flagged (pple-non-numeric-operand)."""

    SOURCE = textwrap.dedent(
        """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class S(pa.DataFrameModel):
            s: str

            class Config:
                coerce = True

        class Out(pa.DataFrameModel):
            s: str

            class Config:
                coerce = True

        @pa.check_types
        def bug_cum_sum_on_string(df: DataFrame[S]) -> DataFrame[Out]:
            return df.select(pl.col("s").cum_sum())
    """
    )

    def test_repro_fails_with_ply016(self):
        results = check_source(self.SOURCE)
        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-non-numeric-operand" in str(e) for e in results[0].errors), results[
            0
        ].errors


class TestIssue50TzMixingEndToEnd:
    """Issue #50 repros: tz-aware and tz-naive Datetime mixing is flagged.

    Probed (polars 1.41.2): ``pl.concat([naive, utc])`` and
    ``aware - naive`` both raise SchemaError; the same-tz equivalents
    succeed (``Datetime[UTC] - Datetime[UTC]`` -> Duration).
    """

    HEADER = textwrap.dedent(
        """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame
        """
    )

    def test_concat_naive_and_utc_fails_with_ply020(self):
        source = self.HEADER + textwrap.dedent(
            """
            class TsNaive(pa.DataFrameModel):
                t: pl.Datetime

            class TsUTC(pa.DataFrameModel):
                t: pl.Datetime(time_zone="UTC")

            def f(naive: DataFrame[TsNaive], utc: DataFrame[TsUTC]) -> DataFrame[TsNaive]:
                return pl.concat([naive, utc], how="vertical")
            """
        )
        result = check_source(source)[0]
        assert result.passed is False
        assert any("pple-concat-mismatch" in str(e) for e in result.errors), result.errors

    def test_concat_same_tz_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class TsUTC(pa.DataFrameModel):
                t: pl.Datetime(time_zone="UTC")

            def f(x: DataFrame[TsUTC], y: DataFrame[TsUTC]) -> DataFrame[TsUTC]:
                return pl.concat([x, y], how="vertical")
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors

    def test_aware_minus_naive_fails_with_ply009(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Mixed(pa.DataFrameModel):
                a: pl.Datetime
                b: pl.Datetime(time_zone="UTC")

            class Out(pa.DataFrameModel):
                span: pl.Duration

            def f(df: DataFrame[Mixed]) -> DataFrame[Out]:
                return df.select(span=pl.col("b") - pl.col("a"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is False
        assert len(result.errors) == 1, result.errors
        assert "pple-incompatible-operands" in str(result.errors[0])

    def test_same_tz_difference_declared_duration_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class TwoUTC(pa.DataFrameModel):
                a: pl.Datetime(time_zone="UTC")
                b: pl.Datetime(time_zone="UTC")

            class Out(pa.DataFrameModel):
                span: pl.Duration

            def f(df: DataFrame[TwoUTC]) -> DataFrame[Out]:
                return df.select(span=pl.col("b") - pl.col("a"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors

    def test_declared_tz_round_trips_through_identity(self):
        # The declared tz must survive inference: returning the input
        # unchanged satisfies the same tz-aware schema.
        source = self.HEADER + textwrap.dedent(
            """
            class TsUTC(pa.DataFrameModel):
                t: pl.Datetime(time_zone="UTC")

            def f(df: DataFrame[TsUTC]) -> DataFrame[TsUTC]:
                return df.sort("t")
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors

    def test_naive_inferred_vs_aware_declared_is_flagged(self):
        # Probed: pandera rejects a naive frame against an aware schema —
        # the dtype difference is real, not noise.
        source = self.HEADER + textwrap.dedent(
            """
            class TsNaive(pa.DataFrameModel):
                t: pl.Datetime

            class TsUTC(pa.DataFrameModel):
                t: pl.Datetime(time_zone="UTC")

            def f(df: DataFrame[TsNaive]) -> DataFrame[TsUTC]:
                return df.sort("t")
            """
        )
        result = check_source(source)[0]
        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors), result.errors


class TestIssue51BinNamespaceEndToEnd:
    """Issue #51 repro: ``.bin`` on a non-Binary column flags pple-wrong-namespace-dtype;
    ``.bin.encode("hex")`` on a declared ``pl.Binary`` column passes with
    the result declared ``str`` (probed: encode -> String)."""

    HEADER = textwrap.dedent(
        """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame
        """
    )

    def test_bin_on_int_fails_with_ply012(self):
        source = self.HEADER + textwrap.dedent(
            """
            class In(pa.DataFrameModel):
                a: int

            class Out(pa.DataFrameModel):
                hex: str

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(hex=pl.col("a").bin.encode("hex"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is False
        assert any("pple-wrong-namespace-dtype" in str(e) for e in result.errors), result.errors

    def test_bin_encode_on_binary_declared_str_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class In(pa.DataFrameModel):
                payload: pl.Binary

            class Out(pa.DataFrameModel):
                hex: str

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(hex=pl.col("payload").bin.encode("hex"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors

    def test_bin_roundtrip_decode_declared_bytes_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class In(pa.DataFrameModel):
                hex: str

            class Out(pa.DataFrameModel):
                payload: bytes

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.select(payload=pl.col("hex").cast(pl.Binary).bin.decode("hex"))
            """
        )
        result = check_source(source)[0]
        assert result.passed is True, result.errors


class TestIssue52DecimalArithmeticEndToEnd:
    """Issue #52 repro: Decimal arithmetic propagates precision growth.

    Probed (polars 1.41.2): ``Decimal(10,2) + Decimal(10,2)`` materializes
    as ``Decimal(38, 2)`` — declaring the grown precision must pass (the
    old left-operand fallback claimed a stale ``Decimal(10, 2)`` and
    flagged a false positive), and declaring the stale pre-growth
    precision must now FAIL (the killed false negative).
    """

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class DecIn(pa.DataFrameModel):
            d: pl.Decimal(10, 2)
            e: pl.Decimal(10, 2)

            class Config:
                coerce = True
    """)

    def test_decimal_sum_declared_with_grown_precision_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class DecSum(pa.DataFrameModel):
                s: pl.Decimal(38, 2)

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def add(df: DataFrame[DecIn]) -> DataFrame[DecSum]:
                return df.select(s=pl.col("d") + pl.col("e"))
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert results[0].errors == []

    def test_decimal_sum_declared_with_stale_precision_fails(self):
        """The killed false negative: claiming the input precision is wrong."""
        source = self.HEADER + textwrap.dedent(
            """
            class DecSum(pa.DataFrameModel):
                s: pl.Decimal(10, 2)

                class Config:
                    strict = True

            @pa.check_types
            def add(df: DataFrame[DecIn]) -> DataFrame[DecSum]:
                return df.select(s=pl.col("d") + pl.col("e"))
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, TypeDifference) and e.column == "s" for e in results[0].errors)

    def test_decimal_int_product_declared_with_grown_precision_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Qty(pa.DataFrameModel):
                d: pl.Decimal(10, 2)
                n: int

                class Config:
                    coerce = True

            class Total(pa.DataFrameModel):
                t: pl.Decimal(38, 2)

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def total(df: DataFrame[Qty]) -> DataFrame[Total]:
                return df.select(t=pl.col("d") * pl.col("n"))
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_decimal_mixed_scale_sum_takes_max_scale(self):
        source = self.HEADER + textwrap.dedent(
            """
            class FineIn(pa.DataFrameModel):
                a: pl.Decimal(10, 2)
                b: pl.Decimal(12, 4)

                class Config:
                    coerce = True

            class FineOut(pa.DataFrameModel):
                s: pl.Decimal(38, 4)

                class Config:
                    strict = True
                    coerce = True

            @pa.check_types
            def add(df: DataFrame[FineIn]) -> DataFrame[FineOut]:
                return df.select(s=pl.col("a") + pl.col("b"))
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors

    def test_decimal_floordiv_flags_ply009(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Out(pa.DataFrameModel):
                s: pl.Decimal(38, 2)

            @pa.check_types
            def bad(df: DataFrame[DecIn]) -> DataFrame[Out]:
                return df.select(s=pl.col("d") // pl.col("e"))
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-incompatible-operands" in str(e) for e in results[0].errors), results[
            0
        ].errors


class TestIssue55ListSumOnStringsEndToEnd:
    """Issue #55 regression (30fc482): ``list.sum()`` on List(String) must
    fail again — probed-invalid container reductions flag pple-non-numeric-operand, so BOTH
    of the repro's wrong declarations (int and str, coerce=False) fail.
    """

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class Tags(pa.DataFrameModel):
            id: int
            tags: pl.List(pl.Utf8) = pa.Field()
    """)

    def test_declared_int_fails_with_ply016(self):
        source = self.HEADER + textwrap.dedent(
            """
            class WrongInt(pa.DataFrameModel):
                id: int
                total: int

            @pa.check_types
            def sum_tags(df: DataFrame[Tags]) -> DataFrame[WrongInt]:
                return df.select("id", total=pl.col("tags").list.sum())
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-non-numeric-operand" in str(e) for e in results[0].errors), results[
            0
        ].errors

    def test_declared_str_fails_with_ply016(self):
        source = self.HEADER + textwrap.dedent(
            """
            class WrongStr(pa.DataFrameModel):
                id: int
                total: str

            @pa.check_types
            def sum_tags(df: DataFrame[Tags]) -> DataFrame[WrongStr]:
                return df.select("id", total=pl.col("tags").list.sum())
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("pple-non-numeric-operand" in str(e) for e in results[0].errors), results[
            0
        ].errors

    def test_list_sum_on_ints_still_int64(self):
        # Regression guard: the numeric core is untouched by the pple-non-numeric-operand path.
        source = self.HEADER + textwrap.dedent(
            """
            class Nums(pa.DataFrameModel):
                id: int
                xs: pl.List(pl.Int64) = pa.Field()

            class Out(pa.DataFrameModel):
                id: int
                total: int

            @pa.check_types
            def sum_nums(df: DataFrame[Nums]) -> DataFrame[Out]:
                return df.select("id", total=pl.col("xs").list.sum())
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors


class TestIssue56NameNamespaceEndToEnd:
    """Issue #56 repro: ``.name.*`` chains previously failed with
    'Could not infer return type'."""

    HEADER = textwrap.dedent("""
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class In(pa.DataFrameModel):
            a: int
            b: str

            class Config:
                strict = True
    """)

    def test_pl_all_prefix_against_strict_schema_passes(self):
        source = self.HEADER + textwrap.dedent(
            """
            class PreO(pa.DataFrameModel):
                pre_a: int
                pre_b: str

                class Config:
                    strict = True

            @pa.check_types
            def add_prefix(df: DataFrame[In]) -> DataFrame[PreO]:
                return df.select(pl.all().name.prefix("pre_"))
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert results[0].errors == []

    def test_prefix_wrong_declared_name_fails(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Wrong(pa.DataFrameModel):
                a: int
                pre_b: str

            @pa.check_types
            def add_prefix(df: DataFrame[In]) -> DataFrame[Wrong]:
                return df.select(pl.all().name.prefix("pre_"))
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("a" in str(e) for e in results[0].errors), results[0].errors

    def test_name_map_degrades_gracefully(self):
        # Unknowable output names: the frame opens instead of failing with
        # 'Could not infer return type'; pplw-untyped-callable points at the callable.
        source = self.HEADER + textwrap.dedent(
            """
            class Mapped(pa.DataFrameModel):
                A: int
                B: str

            @pa.check_types
            def upper_names(df: DataFrame[In]) -> DataFrame[Mapped]:
                return df.select(pl.all().name.map(lambda c: c.upper()))
            """
        )
        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True, results[0].errors
        assert any("pplw-untyped-callable" in str(w) for w in results[0].warnings), results[
            0
        ].warnings


class TestArrayWidthVerdict:
    """Array width participates in the subtype verdict (backlog C-7).

    Probed (polars 1.41.2): pandera rejects a width mismatch (coerce
    cannot repair it — the underlying cast raises "cannot cast Array to a
    different width"), so two known-but-different widths fail; an unknown
    width on either side passes leniently (ADR-0003 visibility).
    """

    def test_same_width_passes_precisely(self):
        verdict = _subtype_verdict(Array(Int64(), 3), Array(Int64(), 3))
        assert verdict.ok and verdict.reason is None

    def test_width_mismatch_fails(self):
        assert not _subtype_verdict(Array(Int64(), 3), Array(Int64(), 5)).ok
        assert not _subtype_verdict(Array(Int64(), 5), Array(Int64(), 3)).ok

    def test_unknown_inferred_width_passes_with_leniency_note(self):
        verdict = _subtype_verdict(Array(Int64()), Array(Int64(), 3))
        assert verdict.ok
        assert verdict.reason is not None and "width" in verdict.reason

    def test_unknown_declared_width_passes_with_leniency_note(self):
        verdict = _subtype_verdict(Array(Int64(), 3), Array(Int64()))
        assert verdict.ok
        assert verdict.reason is not None and "width" in verdict.reason

    def test_inner_mismatch_still_fails_regardless_of_width(self):
        assert not _subtype_verdict(Array(Int64(), 3), Array(Utf8(), 3)).ok

    def test_coerce_cannot_fix_width_mismatch(self):
        assert _is_coercible_difference(Array(Int64(), 5), Array(Int64(), 3)) is False

    def test_coerce_still_tolerates_container_kind_difference(self):
        # issue #53 policy unchanged: List <-> Array kind differences stay
        # coerce-tolerated (the List->Array cast is value-dependent).
        assert _is_coercible_difference(ListType(Int64()), Array(Int64(), 3)) is True
        assert _is_coercible_difference(Array(Int64(), 3), ListType(Int64())) is True


class TestEnumCategoriesVerdict:
    """Enum categories participate in the subtype verdict (issue #67).

    Probed (polars 1.41.2): polars treats Enums with different category
    sequences — different sets AND reorderings — as distinct dtypes, and
    pandera validation rejects the mismatch; coerce cannot be relied on
    (the Enum -> Enum cast is value-dependent on category membership).
    Statically unknown categories on either side pass leniently
    (ADR-0003 visibility), mirroring the unknown Array width.
    """

    def test_same_categories_pass_precisely(self):
        verdict = _subtype_verdict(Enum(("a", "b")), Enum(("a", "b")))
        assert verdict.ok and verdict.reason is None

    def test_category_set_mismatch_fails(self):
        assert not _subtype_verdict(Enum(("a", "c")), Enum(("a", "b"))).ok

    def test_category_order_mismatch_fails(self):
        assert not _subtype_verdict(Enum(("b", "a")), Enum(("a", "b"))).ok

    def test_unknown_categories_pass_with_leniency_note(self):
        for inferred, declared in (
            (Enum(), Enum(("a", "b"))),
            (Enum(("a", "b")), Enum()),
        ):
            verdict = _subtype_verdict(inferred, declared)
            assert verdict.ok
            assert verdict.reason is not None and "categories" in verdict.reason

    def test_both_unknown_pass_silently(self):
        # Mirrors Array(None) == Array(None): structural equality, no note.
        verdict = _subtype_verdict(Enum(), Enum())
        assert verdict.ok and verdict.reason is None

    def test_nullable_widening_still_applies(self):
        assert _subtype_verdict(Enum(("a", "b")), Nullable(Enum(("a", "b")))).ok
        assert not _subtype_verdict(Nullable(Enum(("a", "b"))), Enum(("a", "b"))).ok

    def test_coerce_cannot_fix_category_mismatch(self):
        # Enum targets are value-dependent (issue #58 policy): coerce may
        # fail on out-of-category values, so the difference stays an error.
        assert _is_coercible_difference(Enum(("a", "c")), Enum(("a", "b"))) is False


class TestTimeUnitVerdict:
    """Datetime/Duration time units participate in dtype identity (issue #66)."""

    def test_unit_mismatch_fails(self):
        assert not _subtype_verdict(Datetime(unit="us"), Datetime(unit="ns")).ok
        assert not _subtype_verdict(Duration(unit="us"), Duration(unit="ms")).ok

    def test_same_unit_passes(self):
        assert _subtype_verdict(Datetime(unit="ns"), Datetime(unit="ns")).ok

    def test_coerce_repairs_coarsening_only(self):
        # us -> ms divides (value-independent, probed); us -> ns multiplies
        # and overflows for extreme values (probed InvalidOperationError).
        assert _is_coercible_difference(Datetime(unit="us"), Datetime(unit="ms")) is True
        assert _is_coercible_difference(Datetime(unit="us"), Datetime(unit="ns")) is False
        assert _is_coercible_difference(Duration(unit="us"), Duration(unit="ms")) is True
        assert _is_coercible_difference(Duration(unit="us"), Duration(unit="ns")) is False
