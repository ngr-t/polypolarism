"""Cross-module resolution of imported typed helpers (row-poly follow-up #1).

A typed helper defined in a project-local module and imported into the
caller's module should have its signature resolved at the call site:

- a plain ``DataFrame[Schema]``-returning helper's return type is inferred
  (no more PLW003 / "could not infer return type");
- a ``@rowpoly`` helper threads the caller's extra columns into the result
  with their real dtypes, exactly like the same-module case.

Genuinely external imports (stdlib / third-party / unresolvable) MUST stay
on the old path (PLW003) — we never guess at code we can't read.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from polypolarism.cli import check_file


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip("\n"))


def _project_marker(tmp_path: Path) -> None:
    """Bound the import resolver's upward walk at ``tmp_path``."""
    _write(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "demo"
        """,
    )


_HELPERS = """
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame
from polypolarism import rowpoly


class InId(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class OutScore(pa.DataFrameModel):
    id: int
    score: float

    class Config:
        strict = False


@rowpoly("R")
def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.with_columns(score=pl.col("id").cast(pl.Float64))


# Same shape WITHOUT the decorator — the plain typed helper.
def add_score_plain(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.with_columns(score=pl.col("id").cast(pl.Float64))
"""


class TestImportedRowpolyHelper:
    def test_imported_rowpoly_threads_caller_extra(self, tmp_path: Path) -> None:
        # The imported @rowpoly helper preserves the caller's `region`
        # column (declared in Result) — the verified gap, now resolved.
        _project_marker(tmp_path)
        _write(tmp_path / "helpers.py", _HELPERS)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from helpers import add_score


            class Caller(pa.DataFrameModel):
                id: int
                region: str


            class Result(pa.DataFrameModel):
                id: int
                score: float
                region: str


            def use(c: DataFrame[Caller]) -> DataFrame[Result]:
                out = add_score(c)
                return out.select("id", "score", "region")
            """,
        )
        results = check_file(tmp_path / "app.py")
        use = next(r for r in results if r.function_name == "use")
        # No PLW003 — the imported helper resolved.
        assert not any("PLW003" in str(w) for w in use.warnings), use.warnings
        assert use.passed, [str(e) for e in use.errors]

    def test_imported_rowpoly_wrong_extra_dtype_fails_ply040(self, tmp_path: Path) -> None:
        # `region` is really Utf8; declaring it int must FAIL with PLY040 —
        # proving the extra was threaded with its REAL dtype, not Unknown.
        _project_marker(tmp_path)
        _write(tmp_path / "helpers.py", _HELPERS)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from helpers import add_score


            class Caller(pa.DataFrameModel):
                id: int
                region: str


            class ResultWrong(pa.DataFrameModel):
                id: int
                score: float
                region: int


            def use_wrong(c: DataFrame[Caller]) -> DataFrame[ResultWrong]:
                out = add_score(c)
                return out.select("id", "score", "region")
            """,
        )
        results = check_file(tmp_path / "app.py")
        use = next(r for r in results if r.function_name == "use_wrong")
        assert not use.passed
        assert any("PLY040" in str(e) for e in use.errors), use.errors

    def test_imported_rowpoly_via_alias_threads(self, tmp_path: Path) -> None:
        # ``from helpers import add_score as scorer`` binds `scorer`.
        _project_marker(tmp_path)
        _write(tmp_path / "helpers.py", _HELPERS)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from helpers import add_score as scorer


            class Caller(pa.DataFrameModel):
                id: int
                region: str


            class ResultWrong(pa.DataFrameModel):
                id: int
                score: float
                region: int


            def use_alias(c: DataFrame[Caller]) -> DataFrame[ResultWrong]:
                out = scorer(c)
                return out.select("id", "score", "region")
            """,
        )
        results = check_file(tmp_path / "app.py")
        use = next(r for r in results if r.function_name == "use_alias")
        # Threaded under the alias binding -> wrong dtype is a precise PLY040.
        assert not use.passed
        assert any("PLY040" in str(e) for e in use.errors), use.errors


class TestImportedPlainHelper:
    def test_imported_plain_helper_return_is_inferred(self, tmp_path: Path) -> None:
        # The plain (non-rowpoly) typed helper's return type is inferred —
        # no PLW003, and the declared OutScore columns are readable.
        _project_marker(tmp_path)
        _write(tmp_path / "helpers.py", _HELPERS)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from helpers import add_score_plain


            class Caller(pa.DataFrameModel):
                id: int


            class OutScore(pa.DataFrameModel):
                id: int
                score: float

                class Config:
                    strict = False


            def use_plain(c: DataFrame[Caller]) -> DataFrame[OutScore]:
                out = add_score_plain(c)
                return out.select("id", "score")
            """,
        )
        results = check_file(tmp_path / "app.py")
        use = next(r for r in results if r.function_name == "use_plain")
        assert not any("PLW003" in str(w) for w in use.warnings), use.warnings
        assert use.passed, [str(e) for e in use.errors]


class TestExternalImportStillUnresolved:
    def test_stdlib_import_still_warns_plw003(self, tmp_path: Path) -> None:
        # A genuinely external helper (here a stdlib name) must NOT resolve —
        # the old PLW003 path is preserved (no false resolution).
        _project_marker(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from functools import reduce


            class Caller(pa.DataFrameModel):
                id: int


            def use_external(c: DataFrame[Caller]) -> DataFrame[Caller]:
                out = reduce(c)
                return out
            """,
        )
        results = check_file(tmp_path / "app.py")
        use = next(r for r in results if r.function_name == "use_external")
        assert any("PLW003" in str(w) for w in use.warnings), use.warnings


# A @rowpoly helper that does NOT preserve its row variable: select("id")
# drops the caller's extra columns. Its OWN file flags PLY043; the caller
# must NOT trust the marker and thread columns that are gone at runtime
# (issue #112).
_HELPERS_DROP = """
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame
from polypolarism import rowpoly


class InId(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class OutScore(pa.DataFrameModel):
    id: int
    score: float

    class Config:
        strict = False


# select("id") drops every caller extra — does NOT preserve "R".
@rowpoly("R")
def bad_drop(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.select("id").with_columns(score=pl.col("id").cast(pl.Float64))


# select(pl.exclude("^tmp_.*$")) pattern-drops — also non-preserving.
@rowpoly("R")
def bad_pattern_drop(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.select(pl.exclude("^tmp_.*$")).with_columns(
        score=pl.col("id").cast(pl.Float64)
    )
"""


class TestImportedRowpolyHelperPreservation:
    """Issue #112: threading an imported @rowpoly helper must be gated on the
    helper provably PRESERVING its row variable. A non-preserving imported
    helper must not silently thread columns it drops at runtime."""

    def test_imported_dropping_helper_not_threaded_caller_flagged(self, tmp_path: Path) -> None:
        # The #112 reproducer: bad_drop drops `label`, so the caller's
        # downstream pl.col("label") references a column gone at runtime.
        # Analyzing the CALLER alone must no longer be a silent OK.
        _project_marker(tmp_path)
        _write(tmp_path / "helpers.py", _HELPERS_DROP)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from helpers import bad_drop


            class Wide(pa.DataFrameModel):
                id: int
                label: str

                class Config:
                    strict = False


            class OutScore(pa.DataFrameModel):
                id: int
                score: float

                class Config:
                    strict = False


            def c_trusts_bad(wide: DataFrame[Wide]) -> DataFrame[OutScore]:
                s = bad_drop(wide)
                return s.with_columns(up=pl.col("label").str.to_uppercase())
            """,
        )
        results = check_file(tmp_path / "app.py")
        use = next(r for r in results if r.function_name == "c_trusts_bad")
        # Surfaced (not silent): a PLW014 warning that the imported helper's
        # row variable is unverified / not threaded.
        assert any("PLW014" in str(w) for w in use.warnings), use.warnings
        # `label` was NOT threaded, so it isn't trusted downstream. Whether
        # this lands as an error or merely degrades, the key invariant is:
        # the function must NOT be a clean silent OK with no diagnostic.
        assert not (use.passed and not use.warnings), "c_trusts_bad must not be a silent OK"

    def test_imported_pattern_dropping_helper_not_threaded(self, tmp_path: Path) -> None:
        # select(pl.exclude(...)) pattern-drops — reuses the PLY043
        # pattern-drop detection; must also not be trusted at the call site.
        _project_marker(tmp_path)
        _write(tmp_path / "helpers.py", _HELPERS_DROP)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from helpers import bad_pattern_drop


            class Wide(pa.DataFrameModel):
                id: int
                label: str

                class Config:
                    strict = False


            class OutScore(pa.DataFrameModel):
                id: int
                score: float

                class Config:
                    strict = False


            def c_trusts_pattern(wide: DataFrame[Wide]) -> DataFrame[OutScore]:
                s = bad_pattern_drop(wide)
                return s.with_columns(up=pl.col("label").str.to_uppercase())
            """,
        )
        results = check_file(tmp_path / "app.py")
        use = next(r for r in results if r.function_name == "c_trusts_pattern")
        assert any("PLW014" in str(w) for w in use.warnings), use.warnings
        assert not (use.passed and not use.warnings), "c_trusts_pattern must not be a silent OK"

    def test_imported_preserving_helper_still_threads_precisely(self, tmp_path: Path) -> None:
        # NO REGRESSION: a genuinely-preserving imported helper (with_columns)
        # must STILL thread the caller's extra with its real dtype — a
        # wrong-dtype declaration is still a precise PLY040, and no PLW014.
        _project_marker(tmp_path)
        _write(tmp_path / "helpers.py", _HELPERS)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from helpers import add_score


            class Caller(pa.DataFrameModel):
                id: int
                region: str


            class ResultWrong(pa.DataFrameModel):
                id: int
                score: float
                region: int


            def use_wrong(c: DataFrame[Caller]) -> DataFrame[ResultWrong]:
                out = add_score(c)
                return out.select("id", "score", "region")
            """,
        )
        results = check_file(tmp_path / "app.py")
        use = next(r for r in results if r.function_name == "use_wrong")
        # Still threaded precisely -> wrong dtype is PLY040, and no PLW014.
        assert not use.passed
        assert any("PLY040" in str(e) for e in use.errors), use.errors
        assert not any("PLW014" in str(w) for w in use.warnings), use.warnings
