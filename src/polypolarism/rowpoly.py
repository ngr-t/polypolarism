"""Runtime-inert marker for static row-polymorphic helpers (backlog C-14).

``@rowpoly("R")`` is a **no-op at runtime**: it returns the decorated
function unchanged (after best-effort stamping ``fn.__pp_rowpoly__`` for
optional introspection). Its purpose is purely *static* — it names a row
variable that the polypolarism analyzer reads from the AST to thread a
function's "extra columns" from input to output.

Crucially the marker sits **beside** the type annotation, which stays a
bare, Pandera-validated ``DataFrame[Schema]``:

    @pa.check_types
    @rowpoly("R")
    def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
        ...

Pandera (and mypy / pyright) see the function unchanged, so the runtime
schema check is untouched. Wrapping the annotation in ``Annotated`` instead
would make Pandera stop recognizing the parameter and skip validation
entirely (backlog C-14 Tier 1 de-risk, probed against pandera 0.31). The
row variable is therefore a static-only layer that never deviates from
Pandera at runtime.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TypeVar

__all__ = ["rowpoly"]

_F = TypeVar("_F", bound=Callable[..., object])


def rowpoly(name: str) -> Callable[[_F], _F]:
    """Tag the decorated function with row variable ``name`` (runtime no-op).

    The polypolarism static analyzer reads the row-variable name from this
    decorator in the AST; at runtime the decorator returns the function
    unchanged so Pandera validation and ordinary calls are unaffected.
    """

    def decorate(fn: _F) -> _F:
        # Builtins / ``__slots__`` objects reject attribute writes — the
        # AST-level read is what the analyzer relies on, so swallow those.
        with contextlib.suppress(AttributeError, TypeError):
            fn.__pp_rowpoly__ = name  # type: ignore[attr-defined]
        return fn

    return decorate
