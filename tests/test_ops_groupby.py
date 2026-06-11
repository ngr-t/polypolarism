"""Tests for groupby operation type inference."""

import pytest

from polypolarism.ops.groupby import (
    AggExpr,
    AggFunction,
    GroupByTypeError,
    infer_agg_result_type,
    infer_groupby_result,
)
from polypolarism.types import (
    Date,
    Datetime,
    Duration,
    Float16,
    Float32,
    Float64,
    FrameType,
    Int8,
    Int16,
    Int64,
    Int128,
    List,
    Nullable,
    RowVar,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt128,
    Unknown,
    Utf8,
)


class TestAggFunctionSignatures:
    """Test aggregation function type signatures."""

    def test_sum_int64_returns_int64(self):
        """sum(Int64) -> Int64"""
        result = infer_agg_result_type(AggFunction.SUM, Int64())
        assert result == Int64()

    def test_sum_float64_returns_float64(self):
        """sum(Float64) -> Float64"""
        result = infer_agg_result_type(AggFunction.SUM, Float64())
        assert result == Float64()

    def test_mean_int64_returns_float64(self):
        """mean(Int64) -> Float64"""
        result = infer_agg_result_type(AggFunction.MEAN, Int64())
        assert result == Float64()

    def test_mean_float64_returns_float64(self):
        """mean(Float64) -> Float64"""
        result = infer_agg_result_type(AggFunction.MEAN, Float64())
        assert result == Float64()

    def test_count_returns_uint32(self):
        """count(*) -> UInt32"""
        result = infer_agg_result_type(AggFunction.COUNT, Int64())
        assert result == UInt32()

    def test_count_any_type_returns_uint32(self):
        """count works on any type"""
        assert infer_agg_result_type(AggFunction.COUNT, Utf8()) == UInt32()
        assert infer_agg_result_type(AggFunction.COUNT, Float64()) == UInt32()

    def test_n_unique_returns_uint32(self):
        """n_unique(T) -> UInt32"""
        result = infer_agg_result_type(AggFunction.N_UNIQUE, Utf8())
        assert result == UInt32()

    def test_list_returns_list_of_inner_type(self):
        """list(T) -> List[T]"""
        result = infer_agg_result_type(AggFunction.LIST, Int64())
        assert result == List(Int64())

    def test_list_utf8_returns_list_utf8(self):
        """list(Utf8) -> List[Utf8]"""
        result = infer_agg_result_type(AggFunction.LIST, Utf8())
        assert result == List(Utf8())

    def test_first_returns_same_type(self):
        """first(T) -> T"""
        result = infer_agg_result_type(AggFunction.FIRST, Int64())
        assert result == Int64()

    def test_first_utf8_returns_utf8(self):
        """first(Utf8) -> Utf8"""
        result = infer_agg_result_type(AggFunction.FIRST, Utf8())
        assert result == Utf8()

    def test_last_returns_same_type(self):
        """last(T) -> T"""
        result = infer_agg_result_type(AggFunction.LAST, Int64())
        assert result == Int64()

    def test_min_returns_same_type(self):
        """min(T) -> T"""
        result = infer_agg_result_type(AggFunction.MIN, Int64())
        assert result == Int64()

    def test_max_returns_same_type(self):
        """max(T) -> T"""
        result = infer_agg_result_type(AggFunction.MAX, Float64())
        assert result == Float64()


class TestAggFunctionTypeErrors:
    """Test that invalid aggregation function applications raise errors."""

    def test_sum_on_utf8_raises_error(self):
        """sum cannot be applied to Utf8."""
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_agg_result_type(AggFunction.SUM, Utf8())
        assert "sum" in str(exc_info.value).lower()
        assert "Utf8" in str(exc_info.value)

    def test_mean_on_utf8_raises_error(self):
        """mean cannot be applied to Utf8."""
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_agg_result_type(AggFunction.MEAN, Utf8())
        assert "mean" in str(exc_info.value).lower()


class TestNullableHandling:
    """Test nullable type handling in aggregations."""

    def test_sum_nullable_int64_returns_nullable_int64(self):
        """sum(Nullable[Int64]) -> Nullable[Int64]"""
        result = infer_agg_result_type(AggFunction.SUM, Nullable(Int64()))
        assert result == Nullable(Int64())

    def test_mean_nullable_int64_returns_nullable_float64(self):
        """mean(Nullable[Int64]) -> Nullable[Float64]"""
        result = infer_agg_result_type(AggFunction.MEAN, Nullable(Int64()))
        assert result == Nullable(Float64())

    def test_count_ignores_nullability(self):
        """count always returns UInt32, regardless of nullability."""
        result = infer_agg_result_type(AggFunction.COUNT, Nullable(Int64()))
        assert result == UInt32()

    def test_n_unique_ignores_nullability_for_return(self):
        """n_unique always returns UInt32."""
        result = infer_agg_result_type(AggFunction.N_UNIQUE, Nullable(Utf8()))
        assert result == UInt32()

    def test_list_preserves_nullability(self):
        """list(Nullable[T]) -> List[Nullable[T]]"""
        result = infer_agg_result_type(AggFunction.LIST, Nullable(Int64()))
        assert result == List(Nullable(Int64()))

    def test_first_preserves_nullability(self):
        """first(Nullable[T]) -> Nullable[T]"""
        result = infer_agg_result_type(AggFunction.FIRST, Nullable(Int64()))
        assert result == Nullable(Int64())


class TestStdVarNullability:
    """std/var are Nullable(Float64) even on non-nullable input (issue #60).

    Probed (polars 1.41.2): ``group_by(...).agg(std())`` with the default
    ``ddof=1`` yields null for every group of size 1, so the result column
    is honestly nullable regardless of the input's nullability.
    """

    def test_std_returns_nullable_float64(self):
        result = infer_agg_result_type(AggFunction.STD, Float64())
        assert result == Nullable(Float64())

    def test_var_returns_nullable_float64(self):
        result = infer_agg_result_type(AggFunction.VAR, Int64())
        assert result == Nullable(Float64())

    def test_std_nullable_input_stays_single_nullable(self):
        result = infer_agg_result_type(AggFunction.STD, Nullable(Float64()))
        assert result == Nullable(Float64())

    def test_median_stays_non_nullable(self):
        # Probed: median of a non-empty all-non-null group is never null.
        result = infer_agg_result_type(AggFunction.MEDIAN, Float64())
        assert result == Float64()

    def test_quantile_stays_non_nullable(self):
        # Probed: quantile of a non-empty all-non-null group is never null.
        result = infer_agg_result_type(AggFunction.QUANTILE, Int64())
        assert result == Float64()

    def test_std_on_utf8_still_raises(self):
        with pytest.raises(GroupByTypeError):
            infer_agg_result_type(AggFunction.STD, Utf8())


class TestFloat32WidthPreservation:
    """mean/std/var/median/quantile keep the Float32 width (backlog N-2).

    Probed (polars 1.41.2): ``group_by().agg()`` AND select-context
    ``mean``/``std``/``var``/``median``/``quantile`` on a Float32 column
    return **Float32**, not Float64. Every other accepted receiver
    (ints, Float64) still yields Float64.
    """

    def test_mean_float32_returns_float32(self):
        result = infer_agg_result_type(AggFunction.MEAN, Float32())
        assert result == Float32()

    def test_mean_nullable_float32_returns_nullable_float32(self):
        result = infer_agg_result_type(AggFunction.MEAN, Nullable(Float32()))
        assert result == Nullable(Float32())

    def test_std_float32_returns_nullable_float32(self):
        # std stays Nullable (ddof=1 singleton-group rule, issue #60) but
        # must keep the Float32 width.
        result = infer_agg_result_type(AggFunction.STD, Float32())
        assert result == Nullable(Float32())

    def test_var_nullable_float32_stays_single_nullable_float32(self):
        result = infer_agg_result_type(AggFunction.VAR, Nullable(Float32()))
        assert result == Nullable(Float32())

    def test_median_float32_returns_float32(self):
        result = infer_agg_result_type(AggFunction.MEDIAN, Float32())
        assert result == Float32()

    def test_quantile_float32_returns_float32(self):
        result = infer_agg_result_type(AggFunction.QUANTILE, Float32())
        assert result == Float32()


class TestSmallIntAndLandmarkReceivers:
    """Full numeric receiver matrix (backlog N-5), probed on polars 1.41.2.

    - ``sum``/``product`` upcast sub-32-bit integer receivers
      (Int8/Int16/UInt8/UInt16) to **Int64** — signed Int64 even for the
      unsigned receivers — identically in select and group_by().agg().
    - ``mean``/``std``/``var``/``median``/``quantile`` on every integer
      receiver (Int8 through UInt128) return Float64.
    - Float16 keeps its width through EVERY reduction in select context
      (like Float32), but mean/median/quantile on Float16 and product on
      UInt128 PANIC in rust inside grouped evaluation (group_by().agg(),
      over windows) — those cells must raise in context="agg".
    - Int128/UInt128 sum/product/min/max preserve the receiver width.
    """

    # -- sub-32-bit integer upcasts (identical in both contexts) ----------
    @pytest.mark.parametrize("dtype", [Int8(), Int16(), UInt8(), UInt16()])
    @pytest.mark.parametrize("func", [AggFunction.SUM, AggFunction.PRODUCT])
    @pytest.mark.parametrize("context", ["select", "agg"])
    def test_sum_product_small_int_upcasts_to_int64(self, func, dtype, context):
        result = infer_agg_result_type(func, dtype, context=context)
        assert result == Int64()

    def test_sum_nullable_small_int_keeps_nullability(self):
        result = infer_agg_result_type(AggFunction.SUM, Nullable(UInt8()))
        assert result == Nullable(Int64())

    @pytest.mark.parametrize("dtype", [Int8(), Int16(), UInt8(), UInt16()])
    def test_mean_small_int_returns_float64(self, dtype):
        assert infer_agg_result_type(AggFunction.MEAN, dtype) == Float64()
        assert infer_agg_result_type(AggFunction.MEDIAN, dtype) == Float64()
        assert infer_agg_result_type(AggFunction.STD, dtype) == Nullable(Float64())

    @pytest.mark.parametrize("dtype", [Int8(), UInt16()])
    def test_min_max_small_int_preserve_width(self, dtype):
        assert infer_agg_result_type(AggFunction.MIN, dtype) == dtype
        assert infer_agg_result_type(AggFunction.MAX, dtype) == dtype

    # -- 128-bit receivers -------------------------------------------------
    @pytest.mark.parametrize("dtype", [Int128(), UInt128()])
    def test_sum_128bit_preserves_width(self, dtype):
        assert infer_agg_result_type(AggFunction.SUM, dtype) == dtype

    @pytest.mark.parametrize("dtype", [Int128(), UInt128()])
    def test_float_reductions_128bit_return_float64(self, dtype):
        assert infer_agg_result_type(AggFunction.MEAN, dtype) == Float64()
        assert infer_agg_result_type(AggFunction.QUANTILE, dtype) == Float64()
        assert infer_agg_result_type(AggFunction.VAR, dtype) == Nullable(Float64())

    def test_product_int128_preserves_width_both_contexts(self):
        assert infer_agg_result_type(AggFunction.PRODUCT, Int128(), context="select") == Int128()
        assert infer_agg_result_type(AggFunction.PRODUCT, Int128(), context="agg") == Int128()

    def test_product_uint128_select_preserves_width(self):
        result = infer_agg_result_type(AggFunction.PRODUCT, UInt128(), context="select")
        assert result == UInt128()

    def test_product_uint128_agg_raises(self):
        # Probed (polars 1.41.2): group_by().agg(product) on UInt128 panics
        # in rust (SchemaMismatch "Expected list[i64], got u128").
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_agg_result_type(AggFunction.PRODUCT, UInt128(), context="agg")
        assert "product" in str(exc_info.value).lower()
        assert "panic" in str(exc_info.value).lower()

    # -- Float16 -----------------------------------------------------------
    @pytest.mark.parametrize(
        ("func", "expected"),
        [
            (AggFunction.SUM, Float16()),
            (AggFunction.MEAN, Float16()),
            (AggFunction.MEDIAN, Float16()),
            (AggFunction.QUANTILE, Float16()),
            (AggFunction.PRODUCT, Float16()),
            (AggFunction.MIN, Float16()),
            (AggFunction.MAX, Float16()),
            (AggFunction.STD, Nullable(Float16())),
            (AggFunction.VAR, Nullable(Float16())),
        ],
        ids=lambda p: str(p),
    )
    def test_float16_select_keeps_width(self, func, expected):
        # Probed (polars 1.41.2): every select-context reduction on Float16
        # keeps the half-precision width (like Float32).
        result = infer_agg_result_type(func, Float16(), context="select")
        assert result == expected

    @pytest.mark.parametrize("func", [AggFunction.MEAN, AggFunction.MEDIAN, AggFunction.QUANTILE])
    def test_float16_grouped_float_reductions_raise(self, func):
        # Probed (polars 1.41.2): mean/median/quantile on Float16 panic in
        # rust in grouped contexts ("not implemented for dtype Float16").
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_agg_result_type(func, Float16(), context="agg")
        assert "Float16" in str(exc_info.value)
        assert "panic" in str(exc_info.value).lower()

    def test_float16_grouped_panic_applies_to_nullable_receiver(self):
        with pytest.raises(GroupByTypeError):
            infer_agg_result_type(AggFunction.MEAN, Nullable(Float16()), context="agg")

    @pytest.mark.parametrize(
        ("func", "expected"),
        [
            (AggFunction.SUM, Float16()),
            (AggFunction.PRODUCT, Float16()),
            (AggFunction.STD, Nullable(Float16())),
            (AggFunction.VAR, Nullable(Float16())),
            (AggFunction.MIN, Float16()),
            (AggFunction.MAX, Float16()),
        ],
        ids=lambda p: str(p),
    )
    def test_float16_grouped_non_panicking_cells_keep_width(self, func, expected):
        # Probed (polars 1.41.2): sum/product/std/var/min/max on Float16 do
        # NOT panic in group_by().agg() and keep the receiver width.
        result = infer_agg_result_type(func, Float16(), context="agg")
        assert result == expected

    def test_default_context_is_agg(self):
        # The conservative default: an unspecified context must reject the
        # guaranteed-crash cells.
        with pytest.raises(GroupByTypeError):
            infer_agg_result_type(AggFunction.MEAN, Float16())

    def test_infer_groupby_result_rejects_float16_mean(self):
        input_frame = FrameType({"g": Utf8(), "v": Float16()})
        agg_exprs = [AggExpr(column="v", function=AggFunction.MEAN, alias="avg")]
        with pytest.raises(GroupByTypeError):
            infer_groupby_result(input_frame, ["g"], agg_exprs)

    def test_infer_groupby_result_accepts_small_int_sum(self):
        input_frame = FrameType({"g": Utf8(), "v": Int8()})
        agg_exprs = [AggExpr(column="v", function=AggFunction.SUM, alias="total")]
        result = infer_groupby_result(input_frame, ["g"], agg_exprs)
        assert result.columns["total"].dtype == Int64()


class TestTemporalReceivers:
    """Temporal receivers through the reduction matrix (issue #85).

    Probed (polars 1.41.2), identical in select and group_by().agg()
    contexts unless noted:

    - ``mean``/``median``/``quantile`` on Datetime[unit, tz?] /
      Duration[unit] / Time preserve the receiver dtype EXACTLY (time unit
      and tz flow through); on Date they return **Datetime[us]**.
      Singleton groups produce a value (not null), so only the input's
      nullability propagates.
    - ``sum`` and ``std`` on Duration[unit] preserve the unit; std is null
      on singleton groups (ddof=1) exactly like the numeric form, so it
      stays always-nullable.
    - ``var`` on Duration raises InvalidOperationError in BOTH contexts.
    - ``sum``/``std``/``var`` on Date/Datetime/Time raise
      InvalidOperationError as whole-frame reductions; in grouped contexts
      polars instead silently yields an unconditionally all-null column of
      the receiver dtype — never what the author meant, so both contexts
      are rejected statically.
    - ``min``/``max`` preserve the receiver dtype (regression pins).
    """

    MEAN_LIKE = (AggFunction.MEAN, AggFunction.MEDIAN, AggFunction.QUANTILE)
    PRESERVED_RECEIVERS = [
        Datetime(),
        Datetime(unit="ms"),
        Datetime(unit="ns", tz="UTC"),
        Datetime(unit="ms", tz="Asia/Tokyo"),
        Duration(unit="ms"),
        Duration(unit="ns"),
        Time(),
    ]

    # -- mean/median/quantile: preserve / transform -------------------------
    @pytest.mark.parametrize("func", MEAN_LIKE, ids=lambda f: f.name)
    @pytest.mark.parametrize("dtype", PRESERVED_RECEIVERS, ids=str)
    @pytest.mark.parametrize("context", ["select", "agg"])
    def test_mean_like_preserves_temporal_receiver(self, func, dtype, context):
        result = infer_agg_result_type(func, dtype, context=context)
        assert result == dtype

    @pytest.mark.parametrize("func", MEAN_LIKE, ids=lambda f: f.name)
    @pytest.mark.parametrize("context", ["select", "agg"])
    def test_mean_like_on_date_returns_datetime_us(self, func, context):
        # Probed: mean/median/quantile on Date return Datetime[us] (naive).
        result = infer_agg_result_type(func, Date(), context=context)
        assert result == Datetime(unit="us")

    def test_mean_nullable_datetime_propagates_nullability(self):
        dtype = Nullable(Datetime(unit="ms", tz="UTC"))
        assert infer_agg_result_type(AggFunction.MEAN, dtype) == dtype

    def test_median_temporal_stays_non_nullable(self):
        # Probed: median of a non-empty all-non-null group is never null.
        result = infer_agg_result_type(AggFunction.MEDIAN, Duration(unit="us"))
        assert result == Duration(unit="us")

    # -- sum/std on Duration ------------------------------------------------
    @pytest.mark.parametrize("context", ["select", "agg"])
    def test_sum_duration_preserves_unit(self, context):
        result = infer_agg_result_type(AggFunction.SUM, Duration(unit="ns"), context=context)
        assert result == Duration(unit="ns")

    @pytest.mark.parametrize("context", ["select", "agg"])
    def test_std_duration_preserves_unit_and_is_nullable(self, context):
        # std keeps the ddof=1 singleton-group rule (issue #60) on Duration
        # too: probed, a singleton group yields null.
        result = infer_agg_result_type(AggFunction.STD, Duration(unit="ms"), context=context)
        assert result == Nullable(Duration(unit="ms"))

    def test_std_nullable_duration_stays_single_nullable(self):
        result = infer_agg_result_type(AggFunction.STD, Nullable(Duration(unit="ms")))
        assert result == Nullable(Duration(unit="ms"))

    # -- genuinely-invalid cells keep raising (PLY011) ----------------------
    @pytest.mark.parametrize("context", ["select", "agg"])
    def test_var_duration_raises(self, context):
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_agg_result_type(AggFunction.VAR, Duration(unit="ms"), context=context)
        assert "var" in str(exc_info.value).lower()

    @pytest.mark.parametrize("func", [AggFunction.SUM, AggFunction.STD, AggFunction.VAR])
    @pytest.mark.parametrize("dtype", [Date(), Datetime(), Datetime(tz="UTC"), Time()], ids=str)
    def test_sum_std_var_on_non_duration_temporals_select_raises(self, func, dtype):
        # Issue #91: the contexts DIVERGE for these cells — the whole-frame
        # reduction raises InvalidOperationError (a PLY011 proof) ...
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_agg_result_type(func, dtype, context="select")
        assert str(dtype) in str(exc_info.value)

    @pytest.mark.parametrize("func", [AggFunction.SUM, AggFunction.STD, AggFunction.VAR])
    @pytest.mark.parametrize("dtype", [Date(), Datetime(), Datetime(tz="UTC"), Time()], ids=str)
    def test_sum_std_var_on_non_duration_temporals_grouped_all_null(self, func, dtype):
        # ... while the GROUPED context silently succeeds with an
        # unconditionally all-null column of the receiver dtype (probed
        # identical on polars 1.37.0 through 1.41.2). The analyzer layers
        # a PLW012 advisory on top.
        result = infer_agg_result_type(func, dtype, context="agg")
        assert result == Nullable(dtype)

    @pytest.mark.parametrize("dtype", [Date(), Datetime(), Duration(), Time()], ids=str)
    def test_product_temporal_raises(self, dtype):
        # Probed: product raises InvalidOperationError on every temporal
        # receiver in both contexts.
        with pytest.raises(GroupByTypeError):
            infer_agg_result_type(AggFunction.PRODUCT, dtype)

    # -- regression pins ----------------------------------------------------
    @pytest.mark.parametrize(
        "dtype", [Date(), Datetime(unit="ns", tz="UTC"), Duration(unit="ms"), Time()], ids=str
    )
    def test_min_max_preserve_temporal_receiver(self, dtype):
        assert infer_agg_result_type(AggFunction.MIN, dtype) == dtype
        assert infer_agg_result_type(AggFunction.MAX, dtype) == dtype

    def test_infer_groupby_result_accepts_datetime_mean(self):
        # The issue #85 report's shape: grouped mean on Datetime[us].
        input_frame = FrameType({"k": Utf8(), "t": Datetime(unit="us")})
        agg_exprs = [AggExpr(column="t", function=AggFunction.MEAN, alias="mean_t")]
        result = infer_groupby_result(input_frame, ["k"], agg_exprs)
        assert result.columns["mean_t"].dtype == Datetime(unit="us")


class TestUnknownAggregation:
    """Aggregating an Unknown-typed column never raises."""

    def test_count_unknown_returns_uint32(self):
        result = infer_agg_result_type(AggFunction.COUNT, Unknown())
        assert result == UInt32()

    def test_n_unique_unknown_returns_uint32(self):
        result = infer_agg_result_type(AggFunction.N_UNIQUE, Unknown())
        assert result == UInt32()

    def test_list_unknown_returns_list_unknown(self):
        result = infer_agg_result_type(AggFunction.LIST, Unknown())
        assert result == List(Unknown())

    def test_sum_unknown_returns_unknown(self):
        result = infer_agg_result_type(AggFunction.SUM, Unknown())
        assert result == Unknown()

    def test_mean_unknown_returns_unknown(self):
        result = infer_agg_result_type(AggFunction.MEAN, Unknown())
        assert result == Unknown()

    def test_nullable_unknown_returns_unknown(self):
        result = infer_agg_result_type(AggFunction.SUM, Nullable(Unknown()))
        assert result == Unknown()


class TestOpenFrameGroupBy:
    """Missing keys / agg columns on an open frame become Unknown, not errors."""

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_missing_key_on_open_frame_becomes_unknown(self):
        input_frame = FrameType({"v": Int64()}, rest=RowVar("r"))
        agg_exprs = [AggExpr(column="v", function=AggFunction.SUM, alias="total")]
        result = infer_groupby_result(input_frame, ["ym"], agg_exprs)
        assert result.columns["ym"].dtype == Unknown()
        assert result.columns["total"].dtype == Int64()

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_missing_agg_column_on_open_frame_becomes_unknown(self):
        input_frame = FrameType({"k": Utf8()}, rest=RowVar("r"))
        agg_exprs = [AggExpr(column="v", function=AggFunction.SUM, alias="total")]
        result = infer_groupby_result(input_frame, ["k"], agg_exprs)
        assert result.columns["total"].dtype == Unknown()

    def test_missing_key_on_closed_frame_still_raises(self):
        input_frame = FrameType({"v": Int64()})
        agg_exprs = [AggExpr(column="v", function=AggFunction.SUM, alias="total")]
        with pytest.raises(GroupByTypeError):
            infer_groupby_result(input_frame, ["ym"], agg_exprs)


class TestInferGroupByResult:
    """Test infer_groupby_result function."""

    def test_single_key_single_agg(self):
        """Basic group_by with one key and one aggregation."""
        input_frame = FrameType(
            {
                "country": Utf8(),
                "amount": Float64(),
            }
        )
        agg_exprs = [
            AggExpr(column="amount", function=AggFunction.SUM, alias="total_amount"),
        ]
        result = infer_groupby_result(input_frame, ["country"], agg_exprs)

        expected = FrameType(
            {
                "country": Utf8(),
                "total_amount": Float64(),
            }
        )
        assert result == expected

    def test_single_key_multiple_agg(self):
        """Group by with multiple aggregations."""
        input_frame = FrameType(
            {
                "product_id": Int64(),
                "quantity": Int64(),
                "price": Float64(),
            }
        )
        agg_exprs = [
            AggExpr(column="quantity", function=AggFunction.SUM, alias="total_qty"),
            AggExpr(column="price", function=AggFunction.MEAN, alias="avg_price"),
            AggExpr(column="price", function=AggFunction.COUNT, alias="order_count"),
        ]
        result = infer_groupby_result(input_frame, ["product_id"], agg_exprs)

        expected = FrameType(
            {
                "product_id": Int64(),
                "total_qty": Int64(),
                "avg_price": Float64(),
                "order_count": UInt32(),
            }
        )
        assert result == expected

    def test_multiple_keys(self):
        """Group by with multiple keys."""
        input_frame = FrameType(
            {
                "country": Utf8(),
                "year": Int64(),
                "sales": Float64(),
            }
        )
        agg_exprs = [
            AggExpr(column="sales", function=AggFunction.SUM, alias="total_sales"),
        ]
        result = infer_groupby_result(input_frame, ["country", "year"], agg_exprs)

        expected = FrameType(
            {
                "country": Utf8(),
                "year": Int64(),
                "total_sales": Float64(),
            }
        )
        assert result == expected

    def test_key_column_not_found_raises_error(self):
        """Raises error when group by key column doesn't exist."""
        input_frame = FrameType(
            {
                "product": Utf8(),
                "amount": Float64(),
            }
        )
        agg_exprs = [
            AggExpr(column="amount", function=AggFunction.SUM, alias="total"),
        ]
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_groupby_result(input_frame, ["category"], agg_exprs)
        assert "category" in str(exc_info.value)

    def test_agg_column_not_found_raises_error(self):
        """Raises error when aggregation column doesn't exist."""
        input_frame = FrameType(
            {
                "category": Utf8(),
                "amount": Float64(),
            }
        )
        agg_exprs = [
            AggExpr(column="price", function=AggFunction.SUM, alias="total"),
        ]
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_groupby_result(input_frame, ["category"], agg_exprs)
        assert "price" in str(exc_info.value)


class TestDefaultAliasNaming:
    """Test Polars default naming conventions when alias is not provided."""

    def test_sum_default_alias(self):
        """Default alias for sum is column name."""
        input_frame = FrameType(
            {
                "category": Utf8(),
                "amount": Float64(),
            }
        )
        agg_exprs = [
            AggExpr(column="amount", function=AggFunction.SUM, alias=None),
        ]
        result = infer_groupby_result(input_frame, ["category"], agg_exprs)

        # Polars default: column name is used as-is for most aggregations
        assert "amount" in result.columns
        assert result.columns["amount"].dtype == Float64()

    def test_count_default_alias(self):
        """Default alias for count is column name."""
        input_frame = FrameType(
            {
                "category": Utf8(),
                "value": Int64(),
            }
        )
        agg_exprs = [
            AggExpr(column="value", function=AggFunction.COUNT, alias=None),
        ]
        result = infer_groupby_result(input_frame, ["category"], agg_exprs)

        assert "value" in result.columns
        assert result.columns["value"].dtype == UInt32()


class TestAggExprDataClass:
    """Test AggExpr data class."""

    def test_agg_expr_with_alias(self):
        expr = AggExpr(column="amount", function=AggFunction.SUM, alias="total")
        assert expr.column == "amount"
        assert expr.function == AggFunction.SUM
        assert expr.alias == "total"

    def test_agg_expr_without_alias(self):
        expr = AggExpr(column="amount", function=AggFunction.SUM)
        assert expr.alias is None

    def test_agg_expr_output_name_with_alias(self):
        expr = AggExpr(column="amount", function=AggFunction.SUM, alias="total")
        assert expr.output_name == "total"

    def test_agg_expr_output_name_without_alias(self):
        expr = AggExpr(column="amount", function=AggFunction.SUM)
        assert expr.output_name == "amount"
