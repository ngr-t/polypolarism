"""Security invariant: the analyzer must never execute the code it analyzes.

WHY THIS TEST EXISTS
====================
polypolarism is a *static* checker. Its whole input is untrusted Python
source — a user (or CI) points it at arbitrary ``.py`` files, including
files written by someone else. The entire safety story rests on one
architectural property:

    polypolarism only ever ``ast.parse``-es that source and walks the
    resulting tree as data. It never imports it, never ``eval``/``exec``-es
    it, never spawns a process, and never deserializes attacker-controlled
    bytes.

As long as that holds, pointing the checker at a hostile file is harmless:
a malicious module body (``os.system(...)``, ``eval(...)``, an import with
side effects) is just inert AST nodes that are inspected, not run. The
moment a dynamic-execution / process-spawn / deserialization sink appears
anywhere in ``src/``, that guarantee silently breaks and "analyze this
file" can become remote code execution.

That property is currently maintained only by code review. This test turns
it into an *enforced, regression-tested invariant*: it parses every module
under ``src/polypolarism`` and fails if any of the following appear:

  * the dynamic-eval builtins ``eval`` / ``exec`` / ``compile`` /
    ``__import__`` (no import needed — banned as bare-name calls),
  * dynamic import via ``importlib.import_module`` (note: ``importlib.
    metadata.version(...)``, which only *reads* package metadata and is
    used by the version detector, is deliberately still allowed),
  * process spawning (``os.system`` / ``os.popen`` / any ``subprocess.*``),
  * untrusted-bytes deserialization (``pickle`` / ``marshal`` / ``yaml.load``).

If a future change genuinely needs one of these, do not edit the file to
silence the test blindly: that is exactly the change a reviewer must see.
Add a narrow, commented exemption here *and* call it out in review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "polypolarism"

# Bare builtins that evaluate/execute a string of code. These need no
# import, so they are matched as direct ``name(...)`` calls.
_BANNED_BUILTIN_CALLS = frozenset({"eval", "exec", "compile", "__import__"})

# Dotted calls banned by their fully-qualified spelling. ``importlib.
# metadata.*`` is intentionally NOT here — reading a dist's version string
# does not import or run the package (see version_check.py).
_BANNED_DOTTED_CALLS = frozenset(
    {
        "importlib.import_module",  # dynamic import == running module code
        "os.system",
        "os.popen",
        "pickle.load",
        "pickle.loads",
        "marshal.load",
        "marshal.loads",
        "yaml.load",  # unsafe loader can construct arbitrary objects
    }
)

# Any call whose dotted target starts with one of these is banned too —
# covers the whole subprocess surface (run/call/Popen/check_output/...).
_BANNED_DOTTED_PREFIXES = ("subprocess.",)

# Modules whose only role would be exec / process-spawn / deserialization.
# ``import X`` of any of these (or a submodule) is banned. ``os`` and
# ``importlib`` are absent on purpose: both have legitimate read-only uses
# (paths, ``importlib.metadata``) — their dangerous *members* are caught by
# the call/from-import checks instead.
_BANNED_IMPORT_MODULES = frozenset(
    {"subprocess", "pickle", "marshal", "shelve", "socket", "pty", "ctypes", "dill"}
)

# ``from os import <name>`` / ``from importlib import <name>`` smuggle the
# dangerous member in under a bare (possibly aliased) name, dodging the
# dotted-call check. Ban the members directly at the import site.
_BANNED_FROM_IMPORTS = {
    "os": frozenset({"system", "popen", "execl", "execv", "execve", "execvp", "spawnl", "spawnv"}),
    "importlib": frozenset({"import_module"}),
}


def _dotted_name(node: ast.expr) -> str | None:
    """Reconstruct ``a.b.c`` for a Name/Attribute chain, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _violations_in(tree: ast.AST, rel: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = _dotted_name(node.func)
            if target is None:
                continue
            if (
                target in _BANNED_BUILTIN_CALLS
                or target in _BANNED_DOTTED_CALLS
                or target.startswith(_BANNED_DOTTED_PREFIXES)
            ):
                found.append(f"{rel}:{node.lineno}: call to `{target}(...)`")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BANNED_IMPORT_MODULES:
                    found.append(f"{rel}:{node.lineno}: `import {alias.name}`")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top in _BANNED_IMPORT_MODULES:
                found.append(f"{rel}:{node.lineno}: `from {module} import ...`")
            banned_members = _BANNED_FROM_IMPORTS.get(module, frozenset())
            for alias in node.names:
                if alias.name in banned_members:
                    found.append(f"{rel}:{node.lineno}: `from {module} import {alias.name}`")
    return found


def test_src_has_no_dynamic_execution_sinks() -> None:
    """No module under src/ may import or call a code-execution sink.

    Guards the "AST-only, never executes the analyzed code" property that
    makes running polypolarism on untrusted files safe. See the module
    docstring for the full rationale.
    """
    src_files = sorted(_SRC_ROOT.rglob("*.py"))
    # Guard against a broken glob silently passing the whole suite.
    assert src_files, f"no source files found under {_SRC_ROOT}"

    violations: list[str] = []
    for path in src_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        violations.extend(
            _violations_in(tree, path.relative_to(_SRC_ROOT.parent.parent).as_posix())
        )

    assert not violations, (
        "Dynamic code-execution / process-spawn / deserialization sink found in src/.\n"
        "polypolarism must only ast.parse untrusted source, never run it — a sink here\n"
        "can turn `analyze this file` into code execution. If a new sink is truly\n"
        "required, add a narrow commented exemption in tests/test_no_dynamic_execution.py\n"
        "and flag it in review.\n  " + "\n  ".join(violations)
    )


if __name__ == "__main__":  # pragma: no cover - convenience runner
    pytest.main([__file__, "-v"])
