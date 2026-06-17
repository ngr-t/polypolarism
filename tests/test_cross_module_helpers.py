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
