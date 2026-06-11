"""Tests for Expr type inference."""

import pytest

from polypolarism.expr_infer import (
    ColumnNotFoundError,
    TypePromotionError,
    TypeUnificationError,
    infer_cast,
    infer_col,
    infer_lit,
    infer_shift_fill,
    promote_types,
    supertype,
    unify_types,
)
from polypolarism.types import (
    Array,
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float16,
    Float32,
    Float64,
    FrameType,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    List,
    Null,
    Nullable,
    RowVar,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
    Utf8,
)


class TestInferCol:
    """Tests for pl.col() type inference."""

    def test_col_returns_column_type_from_frame(self) -> None:
        """pl.col("x") should return the dtype of column x from FrameType."""
        frame = FrameType(columns={"x": Int64(), "y": Utf8()})
        result = infer_col("x", frame)
        assert result == Int64()

    def test_col_returns_different_column_type(self) -> None:
        """pl.col("y") should return Utf8 when y is Utf8."""
        frame = FrameType(columns={"x": Int64(), "y": Utf8()})
        result = infer_col("y", frame)
        assert result == Utf8()

    def test_col_raises_error_for_nonexistent_column(self) -> None:
        """pl.col("z") should raise error when z doesn't exist."""
        frame = FrameType(columns={"x": Int64()})
        with pytest.raises(ColumnNotFoundError) as exc_info:
            infer_col("z", frame)
        assert "z" in str(exc_info.value)

    def test_col_returns_nullable_type(self) -> None:
        """pl.col("x") should return Nullable type when column is nullable."""
        frame = FrameType(columns={"x": Nullable(Int64())})
        result = infer_col("x", frame)
        assert result == Nullable(Int64())

    def test_col_error_message_includes_available_columns(self) -> None:
        """Error message should include available columns for debugging."""
        frame = FrameType(columns={"a": Int64(), "b": Utf8()})
        with pytest.raises(ColumnNotFoundError) as exc_info:
            infer_col("c", frame)
        error_msg = str(exc_info.value)
        assert "a" in error_msg or "b" in error_msg

    def test_col_missing_on_open_frame_returns_unknown(self) -> None:
        """An open frame may hold extra unknown columns — don't raise."""
        frame = FrameType(columns={"a": Int64()}, rest=RowVar("r"))
        result = infer_col("z", frame)
        assert result == Unknown()

    def test_col_present_on_open_frame_returns_its_dtype(self) -> None:
        frame = FrameType(columns={"a": Int64()}, rest=RowVar("r"))
        result = infer_col("a", frame)
        assert result == Int64()


class TestInferLit:
    """Tests for pl.lit() type inference."""

    def test_lit_int_returns_int64(self) -> None:
        """pl.lit(42) should infer Int64."""
        result = infer_lit(42)
        assert result == Int64()

    def test_lit_float_returns_float64(self) -> None:
        """pl.lit(3.14) should infer Float64."""
        result = infer_lit(3.14)
        assert result == Float64()

    def test_lit_str_returns_utf8(self) -> None:
        """pl.lit("hello") should infer Utf8."""
        result = infer_lit("hello")
        assert result == Utf8()

    def test_lit_bytes_returns_binary(self) -> None:
        """Probed (polars 1.41.2): ``pl.lit(b"x")`` is a Binary literal."""
        from polypolarism.types import Binary

        result = infer_lit(b"x")
        assert result == Binary()

    def test_lit_bool_returns_boolean(self) -> None:
        """pl.lit(True) should infer Boolean."""
        result = infer_lit(True)
        assert result == Boolean()

    def test_lit_none_returns_nullable(self) -> None:
        """pl.lit(None) should infer Nullable type (null literal)."""
        # None is a special case - Polars infers it as Null type
        # We represent this as Nullable with a placeholder inner type
        result = infer_lit(None)
        assert result == Null()


class TestPromoteTypes:
    """Tests for arithmetic operators type promotion."""

    def test_int64_plus_int64_returns_int64(self) -> None:
        """Int64 + Int64 should return Int64."""
        result = promote_types(Int64(), Int64())
        assert result == Int64()

    def test_int64_plus_float64_returns_float64(self) -> None:
        """Int64 + Float64 should promote to Float64."""
        result = promote_types(Int64(), Float64())
        assert result == Float64()

    def test_float64_plus_int64_returns_float64(self) -> None:
        """Float64 + Int64 should promote to Float64 (order doesn't matter)."""
        result = promote_types(Float64(), Int64())
        assert result == Float64()

    def test_int32_plus_int64_returns_int64(self) -> None:
        """Int32 + Int64 should promote to Int64 (larger type wins)."""
        result = promote_types(Int32(), Int64())
        assert result == Int64()

    def test_float32_plus_float64_returns_float64(self) -> None:
        """Float32 + Float64 should promote to Float64."""
        result = promote_types(Float32(), Float64())
        assert result == Float64()

    def test_int32_plus_float32_returns_float64(self) -> None:
        """Int32 + Float32 should promote to Float64 (Polars behavior)."""
        result = promote_types(Int32(), Float32())
        assert result == Float64()

    def test_nullable_int64_plus_int64_returns_nullable_int64(self) -> None:
        """Nullable[Int64] + Int64 should return Nullable[Int64]."""
        result = promote_types(Nullable(Int64()), Int64())
        assert result == Nullable(Int64())

    def test_int64_plus_nullable_int64_returns_nullable_int64(self) -> None:
        """Int64 + Nullable[Int64] should return Nullable[Int64]."""
        result = promote_types(Int64(), Nullable(Int64()))
        assert result == Nullable(Int64())

    def test_nullable_int64_plus_float64_returns_nullable_float64(self) -> None:
        """Nullable[Int64] + Float64 should return Nullable[Float64]."""
        result = promote_types(Nullable(Int64()), Float64())
        assert result == Nullable(Float64())

    def test_null_plus_int64_returns_nullable_int64(self) -> None:
        """Null + Int64 should return Nullable[Int64]."""
        result = promote_types(Null(), Int64())
        assert result == Nullable(Int64())

    def test_int64_plus_utf8_raises_error(self) -> None:
        """Int64 + Utf8 should raise TypePromotionError."""
        with pytest.raises(TypePromotionError) as exc_info:
            promote_types(Int64(), Utf8())
        assert "Int64" in str(exc_info.value)
        assert "Utf8" in str(exc_info.value)

    def test_boolean_plus_int64_raises_error(self) -> None:
        """Boolean + Int64 should raise TypePromotionError."""
        with pytest.raises(TypePromotionError):
            promote_types(Boolean(), Int64())

    def test_unknown_plus_int64_returns_unknown(self) -> None:
        """Unknown absorbs any other type on the left (like unify_types)."""
        result = promote_types(Unknown(), Int64())
        assert result == Unknown()

    def test_int64_plus_unknown_returns_unknown(self) -> None:
        """Unknown absorbs any other type on the right."""
        result = promote_types(Int64(), Unknown())
        assert result == Unknown()

    def test_unknown_plus_utf8_returns_unknown_instead_of_raising(self) -> None:
        """Unknown vs non-numeric must not raise — uncertainty propagates."""
        result = promote_types(Unknown(), Utf8())
        assert result == Unknown()

    def test_nullable_unknown_plus_float64_returns_unknown(self) -> None:
        """Nullable-wrapped Unknown still promotes to Unknown."""
        result = promote_types(Nullable(Unknown()), Float64())
        assert result == Unknown()


class TestInferCast:
    """Tests for cast() type inference."""

    def test_cast_int64_to_float64(self) -> None:
        """cast(Int64, Float64) should return Float64."""
        result = infer_cast(Int64(), Float64())
        assert result == Float64()

    def test_cast_float64_to_int64(self) -> None:
        """cast(Float64, Int64) should return Int64."""
        result = infer_cast(Float64(), Int64())
        assert result == Int64()

    def test_cast_int64_to_utf8(self) -> None:
        """cast(Int64, Utf8) should return Utf8."""
        result = infer_cast(Int64(), Utf8())
        assert result == Utf8()

    def test_cast_nullable_int64_to_float64(self) -> None:
        """cast(Nullable[Int64], Float64) should return Nullable[Float64].

        Nullability is preserved through cast.
        """
        result = infer_cast(Nullable(Int64()), Float64())
        assert result == Nullable(Float64())

    def test_cast_int64_to_nullable_float64(self) -> None:
        """cast(Int64, Nullable[Float64]) should return Nullable[Float64].

        If target type is nullable, result is nullable.
        """
        result = infer_cast(Int64(), Nullable(Float64()))
        assert result == Nullable(Float64())

    def test_cast_nullable_to_non_nullable_preserves_nullability(self) -> None:
        """cast(Nullable[Int64], Float64) should still be Nullable[Float64].

        Cast cannot remove nullability - data may still contain nulls.
        """
        result = infer_cast(Nullable(Int64()), Float64())
        assert result == Nullable(Float64())


class TestUnifyTypes:
    """Tests for type unification (used by when/then/otherwise)."""

    def test_unify_same_types_returns_same_type(self) -> None:
        """Unifying Int64 with Int64 should return Int64."""
        result = unify_types(Int64(), Int64())
        assert result == Int64()

    def test_unify_int64_and_float64_returns_float64(self) -> None:
        """Unifying Int64 with Float64 should return Float64."""
        result = unify_types(Int64(), Float64())
        assert result == Float64()

    def test_unify_nullable_and_non_nullable_returns_nullable(self) -> None:
        """Unifying Nullable[Int64] with Int64 should return Nullable[Int64]."""
        result = unify_types(Nullable(Int64()), Int64())
        assert result == Nullable(Int64())

    def test_unify_null_and_int64_returns_nullable_int64(self) -> None:
        """Unifying Null with Int64 should return Nullable[Int64]."""
        result = unify_types(Null(), Int64())
        assert result == Nullable(Int64())

    def test_unify_int64_and_utf8_raises_error(self) -> None:
        """Unifying Int64 with Utf8 should raise TypeUnificationError."""
        with pytest.raises(TypeUnificationError):
            unify_types(Int64(), Utf8())

    def test_unify_tz_mismatched_datetimes_raises_error(self) -> None:
        """Issue #50 regression: ``pl.concat`` of tz-aware and tz-naive
        Datetime columns is a probed SchemaError — unification must fail
        for any tz-mismatched pair (naive/aware and aware/aware alike)."""
        with pytest.raises(TypeUnificationError):
            unify_types(Datetime(), Datetime(tz="UTC"))
        with pytest.raises(TypeUnificationError):
            unify_types(Datetime(tz="UTC"), Datetime(tz="Asia/Tokyo"))

    def test_unify_same_tz_datetimes_returns_same_dtype(self) -> None:
        assert unify_types(Datetime(tz="UTC"), Datetime(tz="UTC")) == Datetime(tz="UTC")
        assert unify_types(Datetime(), Datetime()) == Datetime()

    def test_unify_utf8_and_utf8_returns_utf8(self) -> None:
        """Unifying Utf8 with Utf8 should return Utf8."""
        result = unify_types(Utf8(), Utf8())
        assert result == Utf8()

    def test_unify_boolean_and_boolean_returns_boolean(self) -> None:
        """Unifying Boolean with Boolean should return Boolean."""
        result = unify_types(Boolean(), Boolean())
        assert result == Boolean()

    def test_unify_unknown_and_int64_returns_unknown(self) -> None:
        """Unknown absorbs any other type on the left."""
        result = unify_types(Unknown(), Int64())
        assert result == Unknown()

    def test_unify_int64_and_unknown_returns_unknown(self) -> None:
        """Unknown absorbs any other type on the right."""
        result = unify_types(Int64(), Unknown())
        assert result == Unknown()

    def test_unify_nullable_unknown_and_utf8_returns_unknown(self) -> None:
        """Nullable-wrapped Unknown still unifies to Unknown."""
        result = unify_types(Nullable(Unknown()), Utf8())
        assert result == Unknown()


# ``TestInferWhenThenOtherwise`` was deliberately removed together with
# ``infer_when_then_otherwise`` (issues #37/#40): the function was never
# wired into the analyzer, and its nullable-condition rule contradicts
# probed polars 1.41.2 behaviour — a null condition row takes the
# *otherwise* branch (``pl.when(pl.Series([True, None, False]))
# .then(10).otherwise(20)`` -> ``[10, 20, 20]``, null_count 0). The probed
# semantics live in ``analyzer._analyze_when_chain`` + ``supertype`` below.


class TestSupertype:
    """Probed polars common-supertype relation (issues #37/#40/#41/#43).

    Ground truth: polars 1.41.2, full dtype-pair matrix driven through
    ``pl.when(c).then(pl.col("l")).otherwise(pl.col("r"))`` and cross-checked
    via ``df.unpivot`` and ``Expr.shift(fill_value=pl.col(...))`` — all three
    ops agree on every probed cell. Representative probe output::

        Int64 + String  -> String          Boolean + Int64  -> Int64
        Int32 + Float32 -> Float64         Date + Datetime  -> Datetime
        Date  + Int64   -> Int64           Duration + Int64 -> Int64
        Null  + Int64   -> Int64 (rows stay null)
        List(Int64) + Int64   -> SchemaError (no supertype)
        List(Int64) + List(String) -> List(String)
        Boolean + Date  -> SchemaError     String + Duration -> InvalidOperationError
    """

    # ---- probed precise cells ----------------------------------------------

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            # identical dtypes
            (Int64(), Int64(), Int64()),
            (Utf8(), Utf8(), Utf8()),
            # numeric lattice (probed widths)
            (Int32(), Int64(), Int64()),
            (Int8(), Int16(), Int16()),
            (UInt8(), UInt16(), UInt16()),
            (Int8(), UInt8(), Int16()),
            (Int8(), UInt16(), Int32()),
            (Int8(), UInt32(), Int64()),
            (Int8(), UInt64(), Float64()),
            (Int32(), UInt16(), Int32()),
            (Int64(), UInt64(), Float64()),
            (Int128(), UInt64(), Int128()),
            (Int64(), Int128(), Int128()),
            (Int8(), Float32(), Float32()),
            (Int16(), Float32(), Float32()),
            (UInt16(), Float32(), Float32()),
            (Int32(), Float32(), Float64()),
            (Int128(), Float32(), Float64()),
            (Int64(), Float64(), Float64()),
            (Float32(), Float64(), Float64()),
            # Boolean widens into any probed numeric
            (Boolean(), Int64(), Int64()),
            (Boolean(), Int8(), Int8()),
            (Boolean(), UInt32(), UInt32()),
            (Boolean(), Float64(), Float64()),
            # Utf8 absorbs most scalar dtypes
            (Int64(), Utf8(), Utf8()),
            (Float64(), Utf8(), Utf8()),
            (Boolean(), Utf8(), Utf8()),
            (Date(), Utf8(), Utf8()),
            (Time(), Utf8(), Utf8()),
            (Datetime(), Utf8(), Utf8()),
            (Decimal(10, 2), Utf8(), Utf8()),
            (Categorical(), Utf8(), Utf8()),
            (Enum(), Utf8(), Utf8()),
            # temporal pairs
            (Date(), Datetime(), Datetime()),
            (Datetime("UTC"), Date(), Datetime("UTC")),
            # temporal x numeric quirk cells (physical-repr promotion, probed)
            (Date(), Int32(), Int32()),
            (Date(), Int64(), Int64()),
            (Date(), UInt32(), Int64()),
            (Date(), UInt64(), Int64()),
            (Date(), Float32(), Float32()),
            (Date(), Float64(), Float64()),
            (Datetime(), Int32(), Int64()),
            (Datetime(), Int64(), Int64()),
            (Datetime("UTC"), Float32(), Float64()),
            (Duration(), Int32(), Int64()),
            (Duration(), Int64(), Int64()),
            (Duration(), Float64(), Float64()),
            (Time(), Int32(), Int64()),
            (Time(), Int64(), Int64()),
            (Time(), Float32(), Float64()),
            # Decimal x float
            (Decimal(10, 2), Float32(), Float64()),
            (Decimal(10, 2), Float64(), Float64()),
            # List recursion
            (List(Int64()), List(Utf8()), List(Utf8())),
            (List(Int32()), List(Float64()), List(Float64())),
            (List(List(Int64())), List(List(Utf8())), List(List(Utf8()))),
        ],
    )
    def test_probed_supertype_cells(
        self, left: DataType, right: DataType, expected: DataType
    ) -> None:
        assert supertype(left, right) == expected
        assert supertype(right, left) == expected  # the relation is symmetric

    # ---- probed no-supertype cells (polars raises at runtime) ---------------

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (List(Int64()), Int64()),
            (List(Int64()), Utf8()),
            (List(Int64()), Date()),
            (Utf8(), Duration()),
            (Boolean(), Date()),
            (Boolean(), Time()),
            (Boolean(), Datetime()),
            (Boolean(), Duration()),
            (Boolean(), Decimal(10, 2)),
            (Boolean(), Categorical()),
            (Boolean(), Enum()),
            (Date(), Time()),
            (Time(), Datetime()),
            (Duration(), Date()),
            (Duration(), Datetime()),
            (Duration(), Time()),
            (Datetime(), Datetime("UTC")),  # tz mismatch
            (Datetime("UTC"), Datetime("Asia/Tokyo")),
            (Date(), Int8()),  # probed: only the >=32-bit widths promote
            (Date(), Int16()),
            (Date(), UInt8()),
            (Date(), Int128()),
            (Time(), UInt32()),  # probed polars quirk: Datetime+UInt32 works, Time+UInt32 errors
            (Time(), UInt64()),
            (Datetime(), Int8()),
            (Duration(), UInt16()),
            (Decimal(10, 2), Date()),
            (Decimal(10, 2), Boolean()),
            (Decimal(10, 2), Categorical()),
            (Categorical(), Int64()),
            (Categorical(), Enum()),
            (Enum(), Float64()),
            (List(Date()), List(Time())),  # recursion propagates no-supertype
        ],
    )
    def test_probed_no_supertype_cells_return_none(self, left: DataType, right: DataType) -> None:
        assert supertype(left, right) is None
        assert supertype(right, left) is None

    # ---- Unknown / unprobed combinations stay silent -------------------------

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (Unknown(), Int64()),
            (Int64(), Unknown()),
            (Nullable(Unknown()), Utf8()),
            (Unknown(), Unknown()),
            # quirky-but-succeeding combos polypolarism deliberately does not
            # model precisely (when/then broadcasts scalars into Structs;
            # Decimal precision arithmetic is data-dependent):
            (Struct({"f": Int64()}), Int64()),
            (Struct({"f": Int64()}), Utf8()),
            (Struct({"f": Int64()}), Boolean()),
            (Decimal(10, 2), Int64()),
            (Decimal(10, 2), Decimal(20, 4)),
            # dtypes outside the probed numeric widths
            (Float16(), Int64()),
            (UInt128(), Int64()),
            # Array cells (issue #53): width-untracked, and the probed
            # behavior is operation-dependent — when/then(Array(i64,3),
            # List(i64)) -> List(Int64) while concat raises SchemaError;
            # when/then(Array(i64,3), Array(i64,2)) -> List(Int64) although
            # the dtypes are "equal" in our width-less model. Every Array
            # cell stays Unknown (silent), including Array vs Array.
            (Array(Int64()), Array(Int64())),
            (Array(Int64()), Array(Float64())),
            (Array(Int64()), List(Int64())),
            (Array(Int64()), Int64()),
            (Array(Int64()), Utf8()),
            (Array(Int64()), Boolean()),
        ],
    )
    def test_unprobed_combinations_return_unknown(self, left: DataType, right: DataType) -> None:
        assert supertype(left, right) == Unknown()
        assert supertype(right, left) == Unknown()

    # ---- Null / Nullable handling -------------------------------------------

    def test_null_plus_type_is_nullable(self) -> None:
        """Probed: Null + Int64 -> Int64 with the null rows preserved, which
        polypolarism models as Nullable[Int64] (consistent with unify_types)."""
        assert supertype(Null(), Int64()) == Nullable(Int64())
        assert supertype(Utf8(), Null()) == Nullable(Utf8())

    def test_null_plus_null_is_null(self) -> None:
        assert supertype(Null(), Null()) == Null()

    def test_null_plus_nullable_stays_nullable(self) -> None:
        assert supertype(Null(), Nullable(Int64())) == Nullable(Int64())

    def test_nullable_operand_makes_result_nullable(self) -> None:
        assert supertype(Nullable(Int64()), Utf8()) == Nullable(Utf8())
        assert supertype(Int32(), Nullable(Int64())) == Nullable(Int64())
        assert supertype(Nullable(Int64()), Nullable(Float64())) == Nullable(Float64())

    def test_nullable_no_supertype_still_none(self) -> None:
        assert supertype(Nullable(Boolean()), Date()) is None

    def test_nullable_unknown_absorbs(self) -> None:
        assert supertype(Nullable(Int64()), Unknown()) == Unknown()

    def test_identical_nullable_types(self) -> None:
        assert supertype(Nullable(Int64()), Nullable(Int64())) == Nullable(Int64())

    def test_equal_parametrized_dtypes(self) -> None:
        assert supertype(Datetime("UTC"), Datetime("UTC")) == Datetime("UTC")
        assert supertype(Decimal(10, 2), Decimal(10, 2)) == Decimal(10, 2)
        assert supertype(Struct({"f": Int64()}), Struct({"f": Int64()})) == Struct({"f": Int64()})


class TestInferShiftFill:
    """Probed dtype rules for ``Expr.shift(n, fill_value=...)`` (issue #43).

    Ground truth (polars 1.41.2): with a fill the shifted-in slots are
    plugged, so a non-nullable receiver stays non-nullable while original
    nulls keep flowing (``[1, None, 3].shift(1, fill_value=0)`` ->
    ``[0, 1, None]``). Expression fills follow the supertype matrix exactly
    (probed via ``fill_value=pl.col(...).first()``); *literal* fills are
    lenient — polars casts the literal INTO the receiver dtype whenever it
    can hold it::

        shift(Int64,   fill=0)      -> Int64      shift(Date,  fill=5)   -> Date
        shift(Float32, fill=5)      -> Float32    shift(Time,  fill=7)   -> Time
        shift(Int64,   fill="x")    -> String     shift(Date,  fill="x") -> String
        shift(Int32,   fill=5.5)    -> Float64    shift(Date,  fill=5.5) -> Date
        shift(Int64,   fill=True)   -> Int64      shift(Utf8,  fill=5)   -> String
        shift(Bool,    fill=5)      -> Int32      shift(Categorical, fill="z") -> Categorical
    """

    # ---- literal fills -------------------------------------------------------

    @pytest.mark.parametrize(
        ("receiver", "fill", "expected"),
        [
            # int literal coerces into the receiver
            (Int64(), Int64(), Int64()),
            (Int32(), Int64(), Int32()),
            (Float32(), Int64(), Float32()),
            (Date(), Int64(), Date()),
            (Datetime(), Int64(), Datetime()),
            (Time(), Int64(), Time()),
            (Duration(), Int64(), Duration()),
            (Utf8(), Int64(), Utf8()),
            # ... except a Boolean receiver, which widens (probed Int32 at
            # runtime; polypolarism models int literals as Int64 throughout)
            (Boolean(), Int64(), Int64()),
            # bool literal coerces into the receiver (probed: Int64 receiver
            # gives [1, ...], Utf8 receiver gives ['true', ...])
            (Boolean(), Boolean(), Boolean()),
            (Int64(), Boolean(), Int64()),
            (Utf8(), Boolean(), Utf8()),
            # float literal: ints/Decimal promote to Float64 (probed); float
            # and lenient receivers keep their dtype
            (Float32(), Float64(), Float32()),
            (Float64(), Float64(), Float64()),
            (Int32(), Float64(), Float64()),
            (Int64(), Float64(), Float64()),
            (Decimal(10, 2), Float64(), Float64()),
            (Date(), Float64(), Date()),
            (Utf8(), Float64(), Utf8()),
            # str literal: string-family receivers absorb it; everything else
            # follows the supertype matrix (Int64 -> String, Date -> String)
            (Utf8(), Utf8(), Utf8()),
            (Categorical(), Utf8(), Categorical()),
            (Enum(), Utf8(), Enum()),
            (Int64(), Utf8(), Utf8()),
            (Date(), Utf8(), Utf8()),
            # ... and where even the matrix has no supertype, keep the
            # receiver dtype (silent; polars errors at runtime)
            (Duration(), Utf8(), Duration()),
        ],
    )
    def test_literal_fill_dtypes(
        self, receiver: DataType, fill: DataType, expected: DataType
    ) -> None:
        assert infer_shift_fill(receiver, fill, fill_is_literal=True) == expected

    # ---- expression fills follow the supertype matrix ------------------------

    def test_expression_fill_follows_supertype(self) -> None:
        assert infer_shift_fill(Int64(), Utf8(), fill_is_literal=False) == Utf8()
        assert infer_shift_fill(Date(), Int64(), fill_is_literal=False) == Int64()
        assert infer_shift_fill(Boolean(), Int64(), fill_is_literal=False) == Int64()

    def test_expression_fill_without_supertype_keeps_receiver(self) -> None:
        assert infer_shift_fill(Int64(), List(Int64()), fill_is_literal=False) == Int64()

    def test_expression_fill_unknown_supertype_stays_unknown(self) -> None:
        # Struct mixes are deliberately Unknown in the supertype relation.
        assert infer_shift_fill(Struct({"f": Int64()}), Int64(), fill_is_literal=False) == Unknown()

    # ---- nullability ----------------------------------------------------------

    def test_nullable_receiver_stays_nullable(self) -> None:
        # Probed: [1, None, 3].shift(1, fill_value=0) -> [0, 1, None].
        assert infer_shift_fill(Nullable(Int64()), Int64(), fill_is_literal=True) == Nullable(
            Int64()
        )

    def test_non_nullable_receiver_stays_non_nullable(self) -> None:
        assert infer_shift_fill(Int64(), Int64(), fill_is_literal=True) == Int64()

    def test_nullable_expression_fill_makes_result_nullable(self) -> None:
        # A Nullable fill expression can plug null values into the
        # shifted-in slots.
        assert infer_shift_fill(Int64(), Nullable(Float64()), fill_is_literal=False) == Nullable(
            Float64()
        )

    def test_unknown_receiver_stays_unknown(self) -> None:
        assert infer_shift_fill(Unknown(), Int64(), fill_is_literal=True) == Unknown()
        assert infer_shift_fill(Nullable(Unknown()), Utf8(), fill_is_literal=False) == Unknown()


class TestShiftFillTzMismatch:
    """Issue #50: a tz-mismatched expression fill has no supertype; the
    probed runtime failure is out of scope for shift, so the receiver
    dtype is kept SILENTLY — pinned so the no-supertype path never starts
    fabricating a dtype or erroring."""

    def test_tz_mismatched_expression_fill_keeps_receiver(self) -> None:
        result = infer_shift_fill(Datetime("UTC"), Datetime(), fill_is_literal=False)
        assert result == Datetime("UTC")

    def test_same_tz_expression_fill_keeps_dtype(self) -> None:
        result = infer_shift_fill(Datetime("UTC"), Datetime("UTC"), fill_is_literal=False)
        assert result == Datetime("UTC")
