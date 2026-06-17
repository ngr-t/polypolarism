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
from typing import TypeVar, overload

__all__ = ["rowpoly"]

_F = TypeVar("_F", bound=Callable[..., object])


@overload
def rowpoly(fn: _F, /) -> _F: ...  # bare ``@rowpoly`` (no parens)
@overload
def rowpoly(
    *args: str, **kwargs: object
) -> Callable[[_F], _F]: ...  # @rowpoly("R") / (a="R1") / (drops=<selector>)
def rowpoly(*args: object, **kwargs: object) -> object:
    """Tag the decorated function with row variable(s) (runtime no-op).

    Accepts every surface and is inert under each:

    - ``@rowpoly("R")`` — a single shared row variable,
    - ``@rowpoly(a="R1", b="R2")`` — one row variable per parameter (C-14
      Tier 5), and
    - ``@rowpoly("R", drops=<selector>)`` — declares an intended pattern
      restriction of the row variable (the helper removes the matching caller
      extras). ``drops=`` is a STATIC-only declaration the analyzer reads from
      the AST; the selector object passed here is accepted and discarded at
      runtime (the ``**kwargs: object`` signature is what keeps a real
      ``cs.starts_with(...)`` selector import-inert).

    The polypolarism static analyzer reads the names from this decorator in
    the AST; at runtime the decorator returns the function unchanged so Pandera
    validation and ordinary calls are unaffected. It deliberately accepts any
    arguments (the signature is never enforced at runtime) so neither surface
    raises at import time.

    A bare ``@rowpoly`` (no parentheses) applies the decorator directly to the
    function, i.e. ``rowpoly(fn)`` — return ``fn`` unchanged so the function is
    not silently rebound to the inner closure. (The analyzer ignores a bare
    ``@rowpoly`` anyway; this only keeps the runtime behavior inert.)
    """
    if len(args) == 1 and not kwargs and callable(args[0]):
        return args[0]

    def decorate(fn: _F) -> _F:
        # Builtins / ``__slots__`` objects reject attribute writes — the
        # AST-level read is what the analyzer relies on, so swallow those.
        with contextlib.suppress(AttributeError, TypeError):
            fn.__pp_rowpoly__ = args[0] if args else kwargs  # type: ignore[attr-defined]
        return fn

    return decorate
