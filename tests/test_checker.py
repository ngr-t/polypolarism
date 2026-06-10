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
    UInt32,
    Utf8,
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
