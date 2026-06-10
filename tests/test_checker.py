"""Tests for checker."""

import textwrap

from polypolarism.analyzer import FunctionAnalysis
from polypolarism.checker import (
    ExtraColumn,
    MissingColumn,
    TypeDifference,
    _is_coercible_difference,
    check_function,
    check_source,
)
from polypolarism.types import (
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
        """Missing column error has helpful message."""
        error = MissingColumn("name", Utf8())

        message = str(error)

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
        """Extra column error has helpful message."""
        error = ExtraColumn("extra", Float64())

        message = str(error)

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
        """Type difference error has helpful message."""
        error = TypeDifference("value", declared=Float64(), inferred=Int64())

        message = str(error)

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


class TestIsCoercibleDifference:
    """Unit tests for the coercion-compatibility helper."""

    def test_numeric_to_numeric_is_coercible(self):
        assert _is_coercible_difference(UInt32(), Int64()) is True

    def test_float_to_int_is_coercible(self):
        assert _is_coercible_difference(Float64(), Int64()) is True

    def test_non_numeric_is_not_coercible(self):
        assert _is_coercible_difference(Utf8(), Int64()) is False
        assert _is_coercible_difference(Int64(), Utf8()) is False

    def test_nullable_inferred_to_non_nullable_declared_is_not_coercible(self):
        # Coercion casts values; it does not remove nulls.
        assert _is_coercible_difference(Nullable(UInt32()), Int64()) is False

    def test_nullable_inferred_to_nullable_declared_is_coercible(self):
        assert _is_coercible_difference(Nullable(UInt32()), Nullable(Int64())) is True

    def test_non_nullable_inferred_to_nullable_declared_is_coercible(self):
        assert _is_coercible_difference(UInt32(), Nullable(Int64())) is True


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
    """Issue #30 repro: ``String + Int64`` arithmetic is flagged with PLY009.

    polars raises InvalidOperationError at runtime for such pairs; the
    output column registers as Unknown so the PLY009 is the only error.
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
        assert "PLY009" in str(result.errors[0])

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
        assert any("PLY010" in str(e) for e in results[0].errors)


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
        assert any("PLY032" in str(e) for e in results[0].errors)

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
        assert any("PLY008" in str(e) for e in results[0].errors)

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
        assert any("PLY007" in str(e) for e in results[0].errors)

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
        assert any("PLY008" in str(e) and "when" in str(e) for e in results[0].errors)

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
    of raising PLY022.

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
        assert "Duration?" in str(diffs[0])
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
        assert any("PLY014" in str(e) for e in results[0].errors)

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
        assert not any("PLY010" in str(e) for e in results[0].errors)

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
        assert any("PLY010" in str(e) for e in results[0].errors)
