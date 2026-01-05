"""Tests for function call type inference."""

import textwrap

import pytest

from polypolarism.analyzer import analyze_source, FunctionRegistry, FunctionInfo
from polypolarism.types import FrameType, Int64, Float64, Utf8, Nullable


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
        source = textwrap.dedent('''
            from polypolarism import DF

            def helper(df: DF["{id: Int64}"]) -> DF["{id: Int64, doubled: Int64}"]:
                return df.with_columns((pl.col("id") * 2).alias("doubled"))

            def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64, doubled: Int64}"]:
                return helper(data)
        ''')
        results = analyze_source(source)

        # Find caller analysis
        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.inferred_return_type is not None
        assert "id" in caller_analysis.inferred_return_type.columns
        assert "doubled" in caller_analysis.inferred_return_type.columns

    def test_call_typed_function_chained(self):
        """Chained function calls propagate types correctly."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def add_b(df: DF["{a: Int64}"]) -> DF["{a: Int64, b: Int64}"]:
                return df.with_columns(pl.lit(100).alias("b"))

            def add_c(df: DF["{a: Int64, b: Int64}"]) -> DF["{a: Int64, b: Int64, c: Int64}"]:
                return df.with_columns((pl.col("a") + pl.col("b")).alias("c"))

            def pipeline(data: DF["{a: Int64}"]) -> DF["{a: Int64, b: Int64, c: Int64}"]:
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
        source = textwrap.dedent('''
            from polypolarism import DF

            def caller(data: DF["{x: Int64}"]) -> DF["{x: Int64, y: Int64}"]:
                return helper(data)

            def helper(df: DF["{x: Int64}"]) -> DF["{x: Int64, y: Int64}"]:
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
        source = textwrap.dedent('''
            from polypolarism import DF

            def untyped_passthrough(df):
                return df

            def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64}"]:
                return untyped_passthrough(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.inferred_return_type is not None
        assert "id" in caller_analysis.inferred_return_type.columns

    def test_untyped_with_transform(self):
        """Untyped function that transforms infers correct type."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def untyped_add_column(df):
                return df.with_columns(pl.lit(100).alias("new_col"))

            def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64, new_col: Int64}"]:
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
        source = textwrap.dedent('''
            from polypolarism import DF

            def process() -> DF["{id: Int64, name: Utf8}"]:
                df: DF["{id: Int64, name: Utf8}"] = get_data()
                return df
        ''')
        results = analyze_source(source)

        process_analysis = next(r for r in results if r.name == "process")
        assert process_analysis.inferred_return_type is not None
        assert "id" in process_analysis.inferred_return_type.columns
        assert "name" in process_analysis.inferred_return_type.columns

    def test_variable_annotation_with_chain(self):
        """Variable annotation followed by method chain."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def process() -> DF["{id: Int64, doubled: Int64}"]:
                df: DF["{id: Int64, value: Int64}"] = get_data()
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
        source = textwrap.dedent('''
            from polypolarism import DF

            def requires_two(df: DF["{id: Int64, name: Utf8}"]) -> DF["{id: Int64, name: Utf8}"]:
                return df

            def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64, name: Utf8}"]:
                return requires_two(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.has_errors
        assert any("name" in err.lower() for err in caller_analysis.errors)

    def test_type_mismatch_error(self):
        """Error when argument column has wrong type."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def expects_int(df: DF["{id: Int64}"]) -> DF["{id: Int64}"]:
                return df

            def caller(data: DF["{id: Utf8}"]) -> DF["{id: Int64}"]:
                return expects_int(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.has_errors
        assert any("id" in err.lower() or "type" in err.lower() for err in caller_analysis.errors)

    def test_nullable_mismatch_error(self):
        """Error when nullable passed where non-nullable expected."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def expects_non_nullable(df: DF["{value: Int64}"]) -> DF["{value: Int64}"]:
                return df

            def caller(data: DF["{value: Int64?}"]) -> DF["{value: Int64}"]:
                return expects_non_nullable(data)
        ''')
        results = analyze_source(source)

        caller_analysis = next(r for r in results if r.name == "caller")
        assert caller_analysis.has_errors
