"""Tests for issue #3: method-call return type propagation.

``self.foo()``, ``Class().foo()``, ``obj.foo()`` etc. should pick up the
callee's ``DataFrame[Schema]`` return annotation, mirroring the existing
behaviour for module-level function calls.
"""

from __future__ import annotations

import textwrap

from polypolarism.checker import check_source

COMMON = """
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame
from typing import cast


class Sales(pa.DataFrameModel):
    sku: str
    qty: int


class Forecast(pa.DataFrameModel):
    sku: str
    forecast: pl.Float64
"""


def _check(body: str) -> list:
    src = textwrap.dedent(COMMON) + textwrap.dedent(body)
    return check_source(src)


def _all_pass(body: str) -> bool:
    results = _check(body)
    return all(r.passed for r in results)


class TestSelfDispatch:
    def test_self_method_call(self):
        # self.method() inside the same class should resolve to the
        # method's annotated return type.
        assert _all_pass(
            """
            class Loader:
                def _load(self) -> DataFrame[Sales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[Sales], Sales.validate(df))

                def run(self) -> DataFrame[Sales]:
                    return self._load()
            """
        )

    def test_cls_method_call(self):
        # cls.method() (inside a @classmethod) should resolve too.
        assert _all_pass(
            """
            class Loader:
                @classmethod
                def _load(cls) -> DataFrame[Sales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[Sales], Sales.validate(df))

                @classmethod
                def run(cls) -> DataFrame[Sales]:
                    return cls._load()
            """
        )

    def test_self_method_chain(self):
        # ``self._load()`` returning a frame, with a frame method chained
        # on top — the receiver type from the method call should be
        # available to the frame-method dispatch.
        assert _all_pass(
            """
            class Loader:
                def _load(self) -> DataFrame[Sales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[Sales], Sales.validate(df))

                def run(self) -> DataFrame[Sales]:
                    return self._load().select(pl.col("sku"), pl.col("qty"))
            """
        )


class TestInstanceDispatch:
    def test_inline_instantiation(self):
        # ``Class()._method()`` — receiver class taken from the call's
        # function name.
        assert _all_pass(
            """
            class Loader:
                def _load(self) -> DataFrame[Sales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[Sales], Sales.validate(df))


            def run() -> DataFrame[Sales]:
                return Loader()._load()
            """
        )

    def test_var_assigned_instance(self):
        # ``var = Class(); var.method()`` — assignment tracks the class so
        # the later call resolves.
        assert _all_pass(
            """
            class Loader:
                def _load(self) -> DataFrame[Sales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[Sales], Sales.validate(df))


            def run() -> DataFrame[Sales]:
                obj = Loader()
                return obj._load()
            """
        )

    def test_class_method_without_instantiation(self):
        # ``Class.method()`` — static/classmethod called on the class,
        # no instantiation. Should still resolve.
        assert _all_pass(
            """
            class Loader:
                @staticmethod
                def _load() -> DataFrame[Sales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[Sales], Sales.validate(df))


            def run() -> DataFrame[Sales]:
                return Loader._load()
            """
        )


class TestOrchestratorPattern:
    def test_load_transform_chain(self):
        # The pattern from the issue: orchestrator method that calls two
        # other methods to load and transform.
        assert _all_pass(
            """
            class Step:
                def _load(self) -> DataFrame[Sales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[Sales], Sales.validate(df))

                def _transform(self, df: DataFrame[Sales]) -> DataFrame[Forecast]:
                    return df.select(
                        pl.col("sku"),
                        pl.col("qty").cast(pl.Float64).alias("forecast"),
                    )

                def run(self) -> DataFrame[Forecast]:
                    return self._transform(self._load())
            """
        )


class TestMismatchStillDetected:
    """The new path must not become a rubber stamp — when the inner
    method's annotated return doesn't actually match the outer
    function's annotation, we still want a mismatch error."""

    def test_inner_returns_wrong_schema(self):
        # The inner schema is STRICT so its return binds closed and the
        # mismatch stays provable. (A non-strict inner schema may carry
        # extra columns through check_types — including Forecast's — so
        # the call result binds open and the mismatch is genuinely not
        # provable; issue #81.)
        results = _check(
            """
            class StrictSales(pa.DataFrameModel):
                sku: str
                qty: int

                class Config:
                    strict = True


            class Loader:
                def _load(self) -> DataFrame[StrictSales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[StrictSales], StrictSales.validate(df))

                def run(self) -> DataFrame[Forecast]:
                    return self._load()       # returns StrictSales, declared Forecast
            """
        )
        # Loader._load and Loader.run should both be checked. _load passes,
        # run fails with a column/schema mismatch.
        run_result = next(r for r in results if r.function_name == "Loader.run")
        assert not run_result.passed

    def test_nonstrict_inner_schema_passes_with_leniency(self):
        # The issue #81 counterpart of the strict case above: a
        # non-strict inner return may legitimately carry the outer
        # schema's columns among its extras, so the mismatch is not
        # provable — the check records open-frame leniency instead.
        results = _check(
            """
            class Loader:
                def _load(self) -> DataFrame[Sales]:
                    df = pl.read_parquet("x.parquet")
                    return cast(DataFrame[Sales], Sales.validate(df))

                def run(self) -> DataFrame[Forecast]:
                    return self._load()
            """
        )
        run_result = next(r for r in results if r.function_name == "Loader.run")
        assert run_result.passed
        assert any("not provably absent" in note for note in run_result.leniency), (
            run_result.leniency
        )


class TestUnknownMethodFallsThrough:
    """When the receiver doesn't resolve to a known class, the new method
    resolution must not interfere — frame methods (``df.select(...)``)
    still work."""

    def test_frame_method_unaffected(self):
        assert _all_pass(
            """
            def run(df: DataFrame[Sales]) -> DataFrame[Sales]:
                return df.select(pl.col("sku"), pl.col("qty"))
            """
        )

    def test_unknown_attribute_call(self):
        # ``foo.bar()`` where foo is unbound — should fail to infer, not
        # crash.
        results = _check(
            """
            def run() -> DataFrame[Sales]:
                return foo.bar()
            """
        )
        run_result = next(r for r in results if r.function_name == "run")
        assert not run_result.passed
