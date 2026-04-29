"""Tests for function call type inference."""

import textwrap

import pytest

from polypolarism.analyzer import analyze_source, FunctionRegistry, FunctionInfo
from polypolarism.types import FrameType, Int64, Float64, Utf8, Nullable


PANDERA_HEADER = """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
"""


class TestFunctionRegistry:
    """Tests for FunctionRegistry."""

    def test_register_and_get(self):
        """Can register and retrieve function info."""
        registry = FunctionRegistry()
        info = FunctionInfo(
            name="test_func",
            node=None,  # Simplified for test
            signature=None,
            inferred_returns={},
        )
        registry.register(info)
        assert registry.get("test_func") == info

    def test_get_unknown_returns_none(self):
        """Getting unknown function returns None."""
        registry = FunctionRegistry()
        assert registry.get("unknown") is None

    def test_has_signature(self):
        """Can check if function has signature."""
        registry = FunctionRegistry()
        # Without signature
        info_no_sig = FunctionInfo(
            name="untyped",
            node=None,
            signature=None,
            inferred_returns={},
        )
        registry.register(info_no_sig)
        assert not registry.has_signature("untyped")


class TestBasicFunctionCall:
    """Tests for basic function call type inference."""

    def test_call_typed_function(self):
        """Calling a typed function infers return type from signature."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class IdSchema(pa.DataFrameModel):
                id: int

            class IdDoubledSchema(pa.DataFrameModel):
                id: int
                doubled: int

            def helper(df: DataFrame[IdSchema]) -> DataFrame[IdDoubledSchema]:
                return df.with_columns((pl.col("id") * 2).alias("doubled"))

            def caller(data: DataFrame[IdSchema]) -> DataFrame[IdDoubledSchema]:
                return helper(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.inferred_return_type is not None
        assert "id" in caller_analysis.inferred_return_type.columns
        assert "doubled" in caller_analysis.inferred_return_type.columns

    def test_call_typed_function_chained(self):
        """Chained function calls propagate types correctly."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class A(pa.DataFrameModel):
                a: int

            class AB(pa.DataFrameModel):
                a: int
                b: int

            class ABC(pa.DataFrameModel):
                a: int
                b: int
                c: int

            def add_b(df: DataFrame[A]) -> DataFrame[AB]:
                return df.with_columns(pl.lit(100).alias("b"))

            def add_c(df: DataFrame[AB]) -> DataFrame[ABC]:
                return df.with_columns((pl.col("a") + pl.col("b")).alias("c"))

            def pipeline(data: DataFrame[A]) -> DataFrame[ABC]:
                temp = add_b(data)
                result = add_c(temp)
                return result
        ''')
        results = analyze_source(source)

        pipeline_analysis = next(r for r in results if r.name == "pipeline")
        assert pipeline_analysis.inferred_return_type is not None
        assert set(pipeline_analysis.inferred_return_type.columns.keys()) == {"a", "b", "c"}

    def test_forward_reference(self):
        """Can call function defined later in the file."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class XSchema(pa.DataFrameModel):
                x: int

            class XYSchema(pa.DataFrameModel):
                x: int
                y: int

            def caller(data: DataFrame[XSchema]) -> DataFrame[XYSchema]:
                return helper(data)

            def helper(df: DataFrame[XSchema]) -> DataFrame[XYSchema]:
                return df.with_columns((pl.col("x") * 2).alias("y"))
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.inferred_return_type is not None
        assert "y" in caller_analysis.inferred_return_type.columns


class TestUntypedFunctionCall:
    """Tests for calling untyped functions."""

    def test_untyped_passthrough(self):
        """Untyped function that passes through infers correct type."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class IdSchema(pa.DataFrameModel):
                id: int

            def untyped_passthrough(df):
                return df

            def caller(data: DataFrame[IdSchema]) -> DataFrame[IdSchema]:
                return untyped_passthrough(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.inferred_return_type is not None
        assert "id" in caller_analysis.inferred_return_type.columns

    def test_untyped_with_transform(self):
        """Untyped function that transforms infers correct type."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class IdSchema(pa.DataFrameModel):
                id: int

            class IdNewSchema(pa.DataFrameModel):
                id: int
                new_col: int

            def untyped_add_column(df):
                return df.with_columns(pl.lit(100).alias("new_col"))

            def caller(data: DataFrame[IdSchema]) -> DataFrame[IdNewSchema]:
                return untyped_add_column(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.inferred_return_type is not None
        assert "id" in caller_analysis.inferred_return_type.columns
        assert "new_col" in caller_analysis.inferred_return_type.columns


class TestVariableAnnotation:
    """Tests for variable type annotations."""

    def test_variable_annotation_basic(self):
        """Variable annotation provides type for unknown source."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class IdName(pa.DataFrameModel):
                id: int
                name: str

            def process() -> DataFrame[IdName]:
                df: DataFrame[IdName] = get_data()
                return df
        ''')
        results = analyze_source(source)

        process_analysis = next(r for r in results if r.name == "process")
        assert process_analysis.inferred_return_type is not None
        assert "id" in process_analysis.inferred_return_type.columns
        assert "name" in process_analysis.inferred_return_type.columns

    def test_variable_annotation_with_chain(self):
        """Variable annotation followed by method chain."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class InSchema(pa.DataFrameModel):
                id: int
                value: int

            class OutSchema(pa.DataFrameModel):
                id: int
                doubled: int

            def process() -> DataFrame[OutSchema]:
                df: DataFrame[InSchema] = get_data()
                result = df.select(
                    pl.col("id"),
                    (pl.col("value") * 2).alias("doubled"),
                )
                return result
        ''')
        results = analyze_source(source)

        process_analysis = next(r for r in results if r.name == "process")
        assert process_analysis.inferred_return_type is not None
        assert "id" in process_analysis.inferred_return_type.columns
        assert "doubled" in process_analysis.inferred_return_type.columns


class TestArgumentTypeCheck:
    """Tests for argument type checking."""

    def test_missing_column_error(self):
        """Error when argument is missing required column."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class TwoColSchema(pa.DataFrameModel):
                id: int
                name: str

            class IdOnlySchema(pa.DataFrameModel):
                id: int

            def requires_two(df: DataFrame[TwoColSchema]) -> DataFrame[TwoColSchema]:
                return df

            def caller(data: DataFrame[IdOnlySchema]) -> DataFrame[TwoColSchema]:
                return requires_two(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.has_errors
        assert any("name" in err.lower() for err in caller_analysis.errors)

    def test_type_mismatch_error(self):
        """Error when argument column has wrong type."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class IntSchema(pa.DataFrameModel):
                id: int

            class StrSchema(pa.DataFrameModel):
                id: str

            def expects_int(df: DataFrame[IntSchema]) -> DataFrame[IntSchema]:
                return df

            def caller(data: DataFrame[StrSchema]) -> DataFrame[IntSchema]:
                return expects_int(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.has_errors
        assert any("id" in err.lower() or "type" in err.lower() for err in caller_analysis.errors)

    def test_nullable_mismatch_error(self):
        """Error when nullable passed where non-nullable expected."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class NonNull(pa.DataFrameModel):
                value: int

            class WithNull(pa.DataFrameModel):
                value: int = pa.Field(nullable=True)

            def expects_non_nullable(df: DataFrame[NonNull]) -> DataFrame[NonNull]:
                return df

            def caller(data: DataFrame[WithNull]) -> DataFrame[NonNull]:
                return expects_non_nullable(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.has_errors
