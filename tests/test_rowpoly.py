"""The ``rowpoly`` decorator must be a runtime no-op (backlog C-14 Tier 1).

The row variable it names is a *static* concept read from the AST. At
runtime the decorator must leave the function completely untouched so that
Pandera's ``@pa.check_types`` (applied on the same function) keeps
validating. The pandera-interaction half of this guarantee is pinned in
``tests/test_runtime_differential.py``; here we pin the decorator's own
inertness without needing pandera installed.
"""

from polypolarism import rowpoly


def test_returns_the_same_function_object() -> None:
    def f(x):
        return x + 1

    decorated = rowpoly("R")(f)
    # Identity: no wrapper, so call semantics are byte-for-byte unchanged.
    assert decorated is f
    assert decorated(41) == 42


def test_stamps_the_row_variable_name_for_introspection() -> None:
    @rowpoly("Rest")
    def g(df):
        return df

    assert g.__pp_rowpoly__ == "Rest"


def test_is_importable_from_package_root() -> None:
    import polypolarism

    assert polypolarism.rowpoly is rowpoly
    assert "rowpoly" in polypolarism.__all__


def test_bare_decorator_without_parens_returns_function_unchanged() -> None:
    # ``@rowpoly`` (no parens) applies the decorator directly: rowpoly(fn).
    # It must return fn unchanged, not rebind the name to the inner closure.
    def f(x):
        return x + 1

    decorated = rowpoly(f)
    assert decorated is f
    assert decorated(41) == 42


def test_keyword_form_is_runtime_inert() -> None:
    # The Tier 5 surface @rowpoly(a="R1", b="R2") must not raise at import or
    # alter the function (it carries the kwargs as metadata).
    @rowpoly(a="R1", b="R2")
    def g(a, b):
        return (a, b)

    assert g(1, 2) == (1, 2)
    assert g.__pp_rowpoly__ == {"a": "R1", "b": "R2"}


def test_does_not_crash_on_unwritable_target() -> None:
    # An object whose attributes cannot be set must still pass through; the
    # analyzer reads the name from the AST, not from a runtime attribute.
    class Slotted:
        __slots__ = ()

        def __call__(self):
            return "ok"

    obj = Slotted()
    decorated = rowpoly("R")(obj)
    assert decorated is obj
    assert decorated() == "ok"
