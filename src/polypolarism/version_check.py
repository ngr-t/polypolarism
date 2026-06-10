"""Detect and validate the polars / pandera versions a target project uses.

Walks upward from a target path to find ``pyproject.toml`` and consults
(in priority order): an explicit CLI override, ``[tool.polypolarism]`` in
the target's ``pyproject.toml``, the project's ``uv.lock`` (exact version),
or the floor specified in ``[project.dependencies]`` /
``[dependency-groups.*]``.

Detection is best-effort and never fails the run — when a version cannot
be inferred, no warning is emitted and the analyzer proceeds with its
default assumptions.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from polypolarism.diagnostics import PLW010, tag


@dataclass(order=True, frozen=True)
class Version:
    major: int
    minor: int = 0
    patch: int = 0

    @classmethod
    def parse(cls, s: str) -> Version | None:
        """Best-effort parse of ``"X"``, ``"X.Y"``, ``"X.Y.Z"`` (with optional
        trailing pre-release/build segments, which are discarded). Returns
        ``None`` for unrecognized input."""
        if not s:
            return None
        s = s.strip()
        parts = s.split(".", 2)
        try:
            major = int(parts[0])
        except ValueError:
            return None
        minor = 0
        patch = 0
        if len(parts) >= 2:
            digits = _leading_digits(parts[1])
            if digits is None:
                return None
            minor = digits
        if len(parts) >= 3:
            digits = _leading_digits(parts[2])
            if digits is None:
                return None
            patch = digits
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _leading_digits(s: str) -> int | None:
    buf = ""
    for c in s:
        if c.isdigit():
            buf += c
        else:
            break
    if not buf:
        return None
    return int(buf)


# Supported ranges. Updated alongside ADR-0001.
#
# Polars: the supported window is the latest two 1.x minor releases. When
# polars ships a new minor, bump ``POLARS_LATEST_KNOWN`` and
# ``POLARS_FLOOR`` follows automatically (latest minor minus one). Anything
# below ``POLARS_FLOOR`` triggers a PLW010 warning.
POLARS_LATEST_KNOWN = Version(1, 41, 0)
POLARS_FLOOR = Version(POLARS_LATEST_KNOWN.major, POLARS_LATEST_KNOWN.minor - 1, 0)
POLARS_SUPPORT_NOTE = (
    f"polypolarism supports polars >= {POLARS_FLOOR} (the latest two 1.x minor "
    "releases — older minors are best-effort)"
)

# Pandera: the AST-relevant surface is just class-name matching
# (DataFrameModel / SchemaModel), which has been stable since 0.19. We keep
# the floor at 0.19 rather than tracking a "latest two minors" window
# because there is no per-minor variation for polypolarism to test against.
PANDERA_FLOOR = Version(0, 19, 0)
PANDERA_SUPPORT_NOTE = (
    "polypolarism supports pandera >= 0.19.0 (both DataFrameModel and "
    "legacy SchemaModel are accepted)"
)


@dataclass(frozen=True)
class DetectedVersion:
    """A version that was either detected or explicitly supplied.

    ``source`` is a human-readable origin: ``"cli"``,
    ``"pyproject.toml [tool.polypolarism]"``, ``"uv.lock"``,
    ``"pyproject.toml dependencies"``, or ``"pyproject.toml dependency-groups"``.
    ``exact`` is True when the source pins an exact version (CLI / lockfile /
    explicit config); False when it was extracted from a ``>=`` floor.
    """

    package: str
    version: Version
    source: str
    exact: bool


@dataclass(frozen=True)
class VersionInfo:
    polars: DetectedVersion | None = None
    pandera: DetectedVersion | None = None


@dataclass(frozen=True)
class VersionWarning:
    package: str
    detected: DetectedVersion
    floor: Version
    message: str


def find_project_root(start: Path) -> Path | None:
    """Walk upward from ``start`` looking for a ``pyproject.toml``.

    Returns the directory containing it, or ``None`` if not found before
    the filesystem root.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def detect_versions(
    project_root: Path | None,
    polars_override: str | None = None,
    pandera_override: str | None = None,
) -> VersionInfo:
    """Resolve polars / pandera versions for a target project.

    Priority per package:
      1. ``polars_override`` / ``pandera_override`` (CLI flag).
      2. ``[tool.polypolarism] polars_version | pandera_version`` in the
         target's ``pyproject.toml``.
      3. ``uv.lock`` package entry (exact version).
      4. ``pyproject.toml`` ``[project.dependencies]`` floor.
      5. ``pyproject.toml`` ``[dependency-groups.*]`` floor.
    """
    polars: DetectedVersion | None = None
    pandera: DetectedVersion | None = None

    if polars_override is not None:
        v = Version.parse(polars_override)
        if v is not None:
            polars = DetectedVersion("polars", v, "cli", exact=True)
    if pandera_override is not None:
        v = Version.parse(pandera_override)
        if v is not None:
            pandera = DetectedVersion("pandera", v, "cli", exact=True)

    if project_root is None:
        return VersionInfo(polars=polars, pandera=pandera)

    pyproject_path = project_root / "pyproject.toml"
    pyproject_data: dict | None = None
    if pyproject_path.is_file():
        try:
            pyproject_data = tomllib.loads(pyproject_path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            pyproject_data = None

    if pyproject_data is not None:
        tool_section = pyproject_data.get("tool", {}).get("polypolarism", {})
        if polars is None:
            cfg = tool_section.get("polars_version")
            if isinstance(cfg, str):
                v = Version.parse(cfg)
                if v is not None:
                    polars = DetectedVersion(
                        "polars", v, "pyproject.toml [tool.polypolarism]", exact=True
                    )
        if pandera is None:
            cfg = tool_section.get("pandera_version")
            if isinstance(cfg, str):
                v = Version.parse(cfg)
                if v is not None:
                    pandera = DetectedVersion(
                        "pandera", v, "pyproject.toml [tool.polypolarism]", exact=True
                    )

    lockfile = project_root / "uv.lock"
    if lockfile.is_file() and (polars is None or pandera is None):
        try:
            lock_data = tomllib.loads(lockfile.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            lock_data = {}
        for pkg in lock_data.get("package", []):
            name = pkg.get("name")
            ver_str = pkg.get("version")
            if not isinstance(name, str) or not isinstance(ver_str, str):
                continue
            v = Version.parse(ver_str)
            if v is None:
                continue
            if name == "polars" and polars is None:
                polars = DetectedVersion("polars", v, "uv.lock", exact=True)
            elif name == "pandera" and pandera is None:
                pandera = DetectedVersion("pandera", v, "uv.lock", exact=True)

    if pyproject_data is not None and (polars is None or pandera is None):
        deps = pyproject_data.get("project", {}).get("dependencies", [])
        if polars is None:
            v = _floor_for("polars", deps)
            if v is not None:
                polars = DetectedVersion("polars", v, "pyproject.toml dependencies", exact=False)
        if pandera is None:
            v = _floor_for("pandera", deps)
            if v is not None:
                pandera = DetectedVersion("pandera", v, "pyproject.toml dependencies", exact=False)

        groups = pyproject_data.get("dependency-groups", {})
        if polars is None or pandera is None:
            for group_deps in groups.values():
                if not isinstance(group_deps, list):
                    continue
                if polars is None:
                    v = _floor_for("polars", group_deps)
                    if v is not None:
                        polars = DetectedVersion(
                            "polars",
                            v,
                            "pyproject.toml dependency-groups",
                            exact=False,
                        )
                if pandera is None:
                    v = _floor_for("pandera", group_deps)
                    if v is not None:
                        pandera = DetectedVersion(
                            "pandera",
                            v,
                            "pyproject.toml dependency-groups",
                            exact=False,
                        )
                if polars is not None and pandera is not None:
                    break

    return VersionInfo(polars=polars, pandera=pandera)


_DEP_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_\-.]+)\s*(\[[^\]]*\])?\s*(.*)$",
)
_VERSION_OP_PATTERN = re.compile(
    r"(>=|==|~=|\^|>)\s*([0-9][0-9A-Za-z._\-]*)",
)


def _floor_for(package: str, deps: list) -> Version | None:
    """Extract the floor version of ``package`` from a list of PEP-508-ish
    dependency strings.

    Picks the lowest candidate when multiple operators are present (so
    ``"polars>=1.0,<2"`` yields ``1.0`` and ``"polars>1.30"`` yields ``1.30``
    — close enough for support-warning purposes).
    """
    for entry in deps:
        if not isinstance(entry, str):
            continue
        m = _DEP_PATTERN.match(entry)
        if m is None:
            continue
        name = m.group(1).lower().replace("_", "-")
        if name != package.lower():
            continue
        rest = m.group(3)
        candidates: list[Version] = []
        for op, ver_str in _VERSION_OP_PATTERN.findall(rest):
            v = Version.parse(ver_str)
            if v is None:
                continue
            if op in (">=", "==", "~=", "^", ">"):
                candidates.append(v)
        if candidates:
            return min(candidates)
    return None


def check_versions(info: VersionInfo) -> list[VersionWarning]:
    """Return one VersionWarning per detected-but-below-floor package.

    Versions that were not detected at all yield no warning — we can't
    warn about something we don't know.
    """
    warnings: list[VersionWarning] = []
    if info.polars is not None and info.polars.version < POLARS_FLOOR:
        warnings.append(
            VersionWarning(
                package="polars",
                detected=info.polars,
                floor=POLARS_FLOOR,
                message=tag(
                    PLW010,
                    f"detected polars {info.polars.version} (from {info.polars.source}); "
                    f"{POLARS_SUPPORT_NOTE}. Type-check accuracy is best-effort below this — "
                    "use --polars-version to override.",
                ),
            )
        )
    if info.pandera is not None and info.pandera.version < PANDERA_FLOOR:
        warnings.append(
            VersionWarning(
                package="pandera",
                detected=info.pandera,
                floor=PANDERA_FLOOR,
                message=tag(
                    PLW010,
                    f"detected pandera {info.pandera.version} (from {info.pandera.source}); "
                    f"{PANDERA_SUPPORT_NOTE}. Use --pandera-version to override.",
                ),
            )
        )
    return warnings
