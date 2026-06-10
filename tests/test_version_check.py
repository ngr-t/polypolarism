"""Tests for version detection and the unsupported-version warning."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

from polypolarism import version_check
from polypolarism.version_check import (
    PANDERA_FLOOR,
    POLARS_FLOOR,
    POLARS_LATEST_KNOWN,
    DetectedVersion,
    Version,
    VersionInfo,
    check_versions,
    detect_versions,
    find_project_root,
)


@pytest.fixture
def no_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend neither polars nor pandera is installed in the running
    environment, so tests exercise the lockfile / dependency-floor paths
    regardless of what the dev environment actually has installed."""
    monkeypatch.setattr(version_check, "_installed_version", lambda package: None)


def _patch_installed(monkeypatch: pytest.MonkeyPatch, versions: dict[str, str]) -> None:
    """Make the installed-environment lookup report exactly ``versions``."""

    def fake(package: str) -> Version | None:
        ver = versions.get(package)
        return Version.parse(ver) if ver is not None else None

    monkeypatch.setattr(version_check, "_installed_version", fake)


class TestVersionParse:
    def test_full_triple(self):
        assert Version.parse("1.32.1") == Version(1, 32, 1)

    def test_double(self):
        assert Version.parse("1.0") == Version(1, 0, 0)

    def test_single(self):
        assert Version.parse("1") == Version(1, 0, 0)

    def test_with_prerelease_suffix(self):
        assert Version.parse("1.32.1rc2") == Version(1, 32, 1)
        assert Version.parse("1.32rc2") == Version(1, 32, 0)

    def test_invalid(self):
        assert Version.parse("not-a-version") is None
        assert Version.parse("") is None
        assert Version.parse("1.x.y") is None

    def test_ordering(self):
        assert Version(0, 19, 0) < Version(1, 0, 0)
        assert Version(1, 0, 0) < Version(1, 0, 1)
        assert Version(1, 0, 0) == Version(1, 0, 0)

    def test_str(self):
        assert str(Version(1, 32, 1)) == "1.32.1"
        assert str(Version(1, 0, 0)) == "1.0.0"


class TestFindProjectRoot:
    def test_from_file(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("")
        sub = tmp_path / "src" / "pkg"
        sub.mkdir(parents=True)
        f = sub / "x.py"
        f.write_text("")
        assert find_project_root(f) == tmp_path

    def test_from_directory(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("")
        sub = tmp_path / "src"
        sub.mkdir()
        assert find_project_root(sub) == tmp_path

    def test_not_found(self, tmp_path: Path):
        assert find_project_root(tmp_path) is None


class TestDetectFromCli:
    def test_cli_override_polars(self, tmp_path: Path):
        info = detect_versions(tmp_path, polars_override="1.5.0")
        assert info.polars is not None
        assert info.polars.version == Version(1, 5, 0)
        assert info.polars.source == "cli"
        assert info.polars.exact is True

    @pytest.mark.usefixtures("no_installed")
    def test_cli_override_invalid_string_ignored(self, tmp_path: Path):
        info = detect_versions(tmp_path, polars_override="garbage")
        assert info.polars is None

    def test_cli_override_short_form(self, tmp_path: Path):
        info = detect_versions(tmp_path, polars_override="1.0")
        assert info.polars is not None
        assert info.polars.version == Version(1, 0, 0)


class TestDetectFromPyprojectConfig:
    def _write(self, tmp_path: Path, body: str):
        (tmp_path / "pyproject.toml").write_text(body)

    def test_tool_polypolarism_section(self, tmp_path: Path):
        self._write(
            tmp_path,
            """
[tool.polypolarism]
polars_version = "1.32.0"
pandera_version = "0.20.0"
""",
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 32, 0)
        assert "tool.polypolarism" in info.polars.source
        assert info.pandera is not None
        assert info.pandera.version == Version(0, 20, 0)

    def test_cli_beats_tool_section(self, tmp_path: Path):
        self._write(
            tmp_path,
            """
[tool.polypolarism]
polars_version = "1.32.0"
""",
        )
        info = detect_versions(tmp_path, polars_override="1.0.0")
        assert info.polars is not None
        assert info.polars.version == Version(1, 0, 0)
        assert info.polars.source == "cli"


class TestDetectFromUvLock:
    def test_uv_lock_exact_version(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "uv.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.1"

[[package]]
name = "pandera"
version = "0.21.0"
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 40, 1)
        assert info.polars.source == "uv.lock"
        assert info.polars.exact is True
        assert info.pandera is not None
        assert info.pandera.version == Version(0, 21, 0)

    def test_tool_section_beats_uv_lock(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            """
[tool.polypolarism]
polars_version = "1.10.0"
"""
        )
        (tmp_path / "uv.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.1"
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 10, 0)


@pytest.mark.usefixtures("no_installed")
class TestDetectFromDependencies:
    def test_project_dependencies_floor(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=1.32.0", "pandera[polars]>=0.20"]
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 32, 0)
        assert info.polars.exact is False
        assert "dependencies" in info.polars.source
        assert info.pandera is not None
        assert info.pandera.version == Version(0, 20, 0)

    def test_dependency_groups_floor(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            """
[dependency-groups]
dev = ["polars>=1.0.0", "pytest>=8.0.0"]
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 0, 0)
        assert "dependency-groups" in info.polars.source

    def test_no_polars_dep(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["requests>=2.0.0"]
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is None
        assert info.pandera is None

    def test_uv_lock_beats_dependencies(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=1.0.0"]
"""
        )
        (tmp_path / "uv.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.1"
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 40, 1)
        assert info.polars.source == "uv.lock"

    def test_bounded_range_picks_floor(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=1.20.0,<1.40.0"]
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 20, 0)


@pytest.mark.usefixtures("no_installed")
class TestDetectMissingFiles:
    def test_no_project_root(self):
        info = detect_versions(None)
        assert info.polars is None
        assert info.pandera is None

    def test_malformed_pyproject(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("not = valid = toml")
        info = detect_versions(tmp_path)
        assert info.polars is None


class TestDetectFromPoetryLock:
    def test_poetry_lock_exact_version(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "poetry.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.1"

[[package]]
name = "pandera"
version = "0.21.0"
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 40, 1)
        assert info.polars.source == "poetry.lock"
        assert info.polars.exact is True
        assert info.pandera is not None
        assert info.pandera.version == Version(0, 21, 0)
        assert info.pandera.source == "poetry.lock"

    def test_uv_lock_beats_poetry_lock(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "uv.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.0"
"""
        )
        (tmp_path / "poetry.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.10.0"
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 40, 0)
        assert info.polars.source == "uv.lock"

    @pytest.mark.usefixtures("no_installed")
    def test_poetry_lock_beats_dependency_floor(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=1.0.0"]
"""
        )
        (tmp_path / "poetry.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.1"
"""
        )
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 40, 1)
        assert info.polars.source == "poetry.lock"

    @pytest.mark.usefixtures("no_installed")
    def test_malformed_poetry_lock_ignored(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "poetry.lock").write_text("not = valid = toml")
        info = detect_versions(tmp_path)
        assert info.polars is None


class TestDetectFromInstalledEnvironment:
    def test_installed_env_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        (tmp_path / "pyproject.toml").write_text("")
        _patch_installed(monkeypatch, {"polars": "1.41.2", "pandera": "0.21.0"})
        info = detect_versions(tmp_path)
        assert info.polars == DetectedVersion(
            "polars", Version(1, 41, 2), "installed environment", exact=True
        )
        assert info.pandera == DetectedVersion(
            "pandera", Version(0, 21, 0), "installed environment", exact=True
        )

    def test_no_project_root_uses_installed_env(self, monkeypatch: pytest.MonkeyPatch):
        _patch_installed(monkeypatch, {"polars": "1.41.2"})
        info = detect_versions(None)
        assert info.polars is not None
        assert info.polars.version == Version(1, 41, 2)
        assert info.polars.source == "installed environment"
        assert info.pandera is None

    def test_cli_override_beats_installed_env(self, monkeypatch: pytest.MonkeyPatch):
        _patch_installed(monkeypatch, {"polars": "1.41.2"})
        info = detect_versions(None, polars_override="1.0.0")
        assert info.polars is not None
        assert info.polars.version == Version(1, 0, 0)
        assert info.polars.source == "cli"

    def test_uv_lock_beats_installed_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "uv.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.0"
"""
        )
        _patch_installed(monkeypatch, {"polars": "1.41.2"})
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 40, 0)
        assert info.polars.source == "uv.lock"

    def test_poetry_lock_beats_installed_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "poetry.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.0"
"""
        )
        _patch_installed(monkeypatch, {"polars": "1.41.2"})
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 40, 0)
        assert info.polars.source == "poetry.lock"

    def test_installed_env_beats_dependency_floor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=1.0.0"]
"""
        )
        _patch_installed(monkeypatch, {"polars": "1.41.2"})
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 41, 2)
        assert info.polars.source == "installed environment"
        assert info.polars.exact is True

    def test_falls_back_to_floor_when_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=1.32.0"]
"""
        )
        _patch_installed(monkeypatch, {})
        info = detect_versions(tmp_path)
        assert info.polars is not None
        assert info.polars.version == Version(1, 32, 0)
        assert info.polars.source == "pyproject.toml dependencies"
        assert info.polars.exact is False


class TestInstalledVersionLookup:
    """The real importlib.metadata-backed lookup."""

    def test_not_installed_returns_none(self):
        assert version_check._installed_version("definitely-not-a-real-package-xyz") is None

    def test_installed_package_parses(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.41.2")
        assert version_check._installed_version("polars") == Version(1, 41, 2)

    def test_unparseable_version_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "garbage")
        assert version_check._installed_version("polars") is None


class TestCheckVersions:
    def test_below_polars_floor_warns(self):
        info = VersionInfo(
            polars=DetectedVersion("polars", Version(0, 20, 0), "uv.lock", exact=True)
        )
        warnings = check_versions(info)
        assert len(warnings) == 1
        assert warnings[0].package == "polars"
        assert "PLW010" in warnings[0].message
        assert "0.20.0" in warnings[0].message

    def test_at_polars_floor_silent(self):
        info = VersionInfo(polars=DetectedVersion("polars", POLARS_FLOOR, "uv.lock", exact=True))
        assert check_versions(info) == []

    def test_above_polars_floor_silent(self):
        info = VersionInfo(
            polars=DetectedVersion("polars", Version(1, 40, 1), "uv.lock", exact=True)
        )
        assert check_versions(info) == []

    def test_below_pandera_floor_warns(self):
        info = VersionInfo(
            pandera=DetectedVersion("pandera", Version(0, 17, 0), "uv.lock", exact=True)
        )
        warnings = check_versions(info)
        assert len(warnings) == 1
        assert warnings[0].package == "pandera"

    def test_at_pandera_floor_silent(self):
        info = VersionInfo(pandera=DetectedVersion("pandera", PANDERA_FLOOR, "uv.lock", exact=True))
        assert check_versions(info) == []

    def test_no_detection_no_warning(self):
        assert check_versions(VersionInfo()) == []

    def test_both_below_floor(self):
        info = VersionInfo(
            polars=DetectedVersion("polars", Version(0, 20, 0), "uv.lock", exact=True),
            pandera=DetectedVersion("pandera", Version(0, 18, 0), "uv.lock", exact=True),
        )
        warnings = check_versions(info)
        assert len(warnings) == 2
        assert {w.package for w in warnings} == {"polars", "pandera"}


class TestNoWarningFromInexactDetection:
    """A ``>=``-floor extracted from dependency specs is not evidence the
    project actually runs that version, so it must never trigger PLW010.
    Exact sources (lockfiles, installed environment, CLI, config) still do."""

    def test_polars_floor_below_floor_is_silent(self):
        info = VersionInfo(
            polars=DetectedVersion(
                "polars", Version(1, 0, 0), "pyproject.toml dependencies", exact=False
            )
        )
        assert check_versions(info) == []

    def test_pandera_floor_below_floor_is_silent(self):
        info = VersionInfo(
            pandera=DetectedVersion(
                "pandera", Version(0, 17, 0), "pyproject.toml dependency-groups", exact=False
            )
        )
        assert check_versions(info) == []

    def test_exact_below_floor_still_warns(self):
        info = VersionInfo(
            polars=DetectedVersion("polars", Version(1, 0, 0), "uv.lock", exact=True)
        )
        assert len(check_versions(info)) == 1

    def test_installed_environment_below_floor_warns(self):
        info = VersionInfo(
            polars=DetectedVersion("polars", Version(1, 10, 0), "installed environment", exact=True)
        )
        warnings = check_versions(info)
        assert len(warnings) == 1
        assert "installed environment" in warnings[0].message

    def test_mixed_only_exact_side_warns(self):
        info = VersionInfo(
            polars=DetectedVersion(
                "polars", Version(1, 0, 0), "pyproject.toml dependencies", exact=False
            ),
            pandera=DetectedVersion("pandera", Version(0, 17, 0), "uv.lock", exact=True),
        )
        warnings = check_versions(info)
        assert len(warnings) == 1
        assert warnings[0].package == "pandera"


@pytest.mark.usefixtures("no_installed")
class TestCliIntegration:
    def test_no_version_check_flag_suppresses(self, tmp_path: Path, capsys, monkeypatch):
        from polypolarism.cli import main

        py = tmp_path / "x.py"
        py.write_text("")
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=0.19.0"]
"""
        )
        rc = main([str(py), "--no-version-check"])
        captured = capsys.readouterr()
        assert "PLW010" not in captured.err
        assert rc == 0

    def test_dependency_floor_alone_does_not_warn(self, tmp_path: Path, capsys):
        """A ">=" floor in pyproject dependencies is the only source — even
        though it is below the supported window, no PLW010 fires: the floor
        of a range is not the version in use (issue #13)."""
        from polypolarism.cli import main

        py = tmp_path / "x.py"
        py.write_text("")
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=0.19.0"]
"""
        )
        main([str(py), "--no-color"])
        captured = capsys.readouterr()
        assert "PLW010" not in captured.err

    def test_old_poetry_lock_emits_warning(self, tmp_path: Path, capsys):
        from polypolarism.cli import main

        py = tmp_path / "x.py"
        py.write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "poetry.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.10.0"
"""
        )
        main([str(py), "--no-color"])
        captured = capsys.readouterr()
        assert "PLW010" in captured.err
        assert "poetry.lock" in captured.err

    def test_old_installed_env_emits_warning(
        self, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        from polypolarism.cli import main

        py = tmp_path / "x.py"
        py.write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        _patch_installed(monkeypatch, {"polars": "1.10.0"})
        main([str(py), "--no-color"])
        captured = capsys.readouterr()
        assert "PLW010" in captured.err
        assert "installed environment" in captured.err

    def test_cli_override_silences_warning(self, tmp_path: Path, capsys):
        from polypolarism.cli import main

        py = tmp_path / "x.py"
        py.write_text("")
        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=0.19.0"]
"""
        )
        main([str(py), "--no-color", "--polars-version", str(POLARS_FLOOR)])
        captured = capsys.readouterr()
        assert "PLW010" not in captured.err

    def test_polars_1_0_now_warns(self, tmp_path: Path, capsys):
        """Polars 1.0 is below the supported window (latest two minors) and
        should emit PLW010, even though it's a 1.x release."""
        from polypolarism.cli import main

        py = tmp_path / "x.py"
        py.write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        main([str(py), "--no-color", "--polars-version", "1.0.0"])
        captured = capsys.readouterr()
        assert "PLW010" in captured.err


# Specific Polars 1.x minors that mark known landmarks in the churn doc.
# Each entry is paired with the expected warn/silent state under the
# current supported window.
#   - 0.20: pre-1.0, definitely warns (analyzer doesn't know the legacy
#     spellings in the first place).
#   - 1.0:  big-bang minor, below window.
#   - 1.18: introduced Int128, below window.
#   - 1.25: Enum stabilized, below window.
#   - 1.27: hist bin-closure shift, below window.
#   - 1.32: selector-as-DSL change, below window.
#   - 1.38: one minor below floor, below window.
#   - 1.39: floor (==POLARS_LATEST_KNOWN.minor - 1), silent.
#   - 1.40: latest known, silent.
#   - 1.41: hypothetical future minor, silent (we don't reject the
#     unknown-future direction; only "too old" is unsupported).
@pytest.mark.parametrize(
    ("version_str", "should_warn"),
    [
        ("0.20.0", True),
        ("1.0.0", True),
        ("1.18.0", True),
        ("1.25.0", True),
        ("1.27.0", True),
        ("1.32.0", True),
        ("1.38.0", True),
        ("1.39.0", False),
        ("1.40.0", False),
        ("1.41.0", False),
    ],
)
@pytest.mark.usefixtures("no_installed")
class TestPolarsVersionLandmarks:
    def test_via_cli_override(self, version_str: str, should_warn: bool, tmp_path: Path, capsys):
        from polypolarism.cli import main

        py = tmp_path / "x.py"
        py.write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        main([str(py), "--no-color", "--polars-version", version_str])
        captured = capsys.readouterr()
        if should_warn:
            assert "PLW010" in captured.err, f"polars {version_str} should warn but didn't"
            assert "polars" in captured.err.lower()
        else:
            assert "PLW010" not in captured.err, f"polars {version_str} should be silent but warned"

    def test_via_check_versions_directly(self, version_str: str, should_warn: bool):
        v = Version.parse(version_str)
        assert v is not None
        info = VersionInfo(polars=DetectedVersion("polars", v, "test", exact=True))
        warnings = check_versions(info)
        if should_warn:
            assert len(warnings) == 1
            assert warnings[0].package == "polars"
            assert warnings[0].detected.version == v
            assert warnings[0].floor == POLARS_FLOOR
        else:
            assert warnings == []


# Specific Pandera versions paired with expected state. Pandera's floor is
# 0.19.0 (where polars validation landed); below that we warn.
#   - 0.17: pre-DataFrameModel, pre-polars-support, warns.
#   - 0.18: still pre-polars-support, warns.
#   - 0.19: floor, silent.
#   - 0.20: DataFrameModel rename era, silent.
#   - 0.22: hypothetical recent minor, silent.
@pytest.mark.parametrize(
    ("version_str", "should_warn"),
    [
        ("0.17.0", True),
        ("0.18.0", True),
        ("0.19.0", False),
        ("0.20.0", False),
        ("0.22.0", False),
    ],
)
class TestPanderaVersionLandmarks:
    def test_via_cli_override(self, version_str: str, should_warn: bool, tmp_path: Path, capsys):
        from polypolarism.cli import main

        py = tmp_path / "x.py"
        py.write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        main(
            [
                str(py),
                "--no-color",
                "--polars-version",
                str(POLARS_LATEST_KNOWN),
                "--pandera-version",
                version_str,
            ]
        )
        captured = capsys.readouterr()
        if should_warn:
            assert "PLW010" in captured.err
            assert "pandera" in captured.err.lower()
        else:
            assert "PLW010" not in captured.err

    def test_via_check_versions_directly(self, version_str: str, should_warn: bool):
        v = Version.parse(version_str)
        assert v is not None
        info = VersionInfo(pandera=DetectedVersion("pandera", v, "test", exact=True))
        warnings = check_versions(info)
        if should_warn:
            assert len(warnings) == 1
            assert warnings[0].package == "pandera"
            assert warnings[0].floor == PANDERA_FLOOR
        else:
            assert warnings == []


class TestWarningMessageContent:
    """The warning surfaces enough info for a user to act on it: the
    detected version, the source it came from, and the override flag."""

    def _info_polars(self, v: Version, source: str) -> VersionInfo:
        return VersionInfo(polars=DetectedVersion("polars", v, source, exact=True))

    def test_includes_diagnostic_code(self):
        warnings = check_versions(self._info_polars(Version(1, 0, 0), "uv.lock"))
        assert "[PLW010]" in warnings[0].message

    def test_includes_detected_version(self):
        warnings = check_versions(self._info_polars(Version(1, 18, 0), "uv.lock"))
        assert "1.18.0" in warnings[0].message

    def test_includes_source_for_uv_lock(self):
        warnings = check_versions(self._info_polars(Version(1, 0, 0), "uv.lock"))
        assert "uv.lock" in warnings[0].message

    def test_includes_source_for_dependencies(self):
        warnings = check_versions(
            self._info_polars(Version(1, 0, 0), "pyproject.toml dependencies")
        )
        assert "pyproject.toml dependencies" in warnings[0].message

    def test_includes_floor_value(self):
        warnings = check_versions(self._info_polars(Version(1, 0, 0), "cli"))
        assert str(POLARS_FLOOR) in warnings[0].message

    def test_includes_override_hint(self):
        warnings = check_versions(self._info_polars(Version(1, 0, 0), "cli"))
        assert "--polars-version" in warnings[0].message

    def test_pandera_message_points_at_pandera_flag(self):
        info = VersionInfo(
            pandera=DetectedVersion("pandera", Version(0, 17, 0), "uv.lock", exact=True)
        )
        warnings = check_versions(info)
        assert "--pandera-version" in warnings[0].message
        assert "0.17.0" in warnings[0].message


@pytest.mark.usefixtures("no_installed")
class TestUvLockExactPriority:
    """When uv.lock pins a specific version, that wins over the floor in
    pyproject.toml dependencies — even when the floor would have produced
    a different warn/silent decision."""

    def test_uv_lock_modern_silences_old_pyproject_floor(self, tmp_path: Path, capsys):
        from polypolarism.cli import main

        (tmp_path / "pyproject.toml").write_text(
            """
[project]
dependencies = ["polars>=1.0.0"]
"""
        )
        (tmp_path / "uv.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.40.0"
"""
        )
        py = tmp_path / "x.py"
        py.write_text("")
        main([str(py), "--no-color"])
        captured = capsys.readouterr()
        # uv.lock at 1.40 should win — no warning even though pyproject says >=1.0.
        assert "PLW010" not in captured.err

    def test_uv_lock_old_warns_even_if_pyproject_floor_modern(self, tmp_path: Path, capsys):
        from polypolarism.cli import main

        (tmp_path / "pyproject.toml").write_text(
            f"""
[project]
dependencies = ["polars>={POLARS_LATEST_KNOWN}"]
"""
        )
        (tmp_path / "uv.lock").write_text(
            """
[[package]]
name = "polars"
version = "1.10.0"
"""
        )
        py = tmp_path / "x.py"
        py.write_text("")
        main([str(py), "--no-color"])
        captured = capsys.readouterr()
        # uv.lock at 1.10 wins — should warn despite "modern" pyproject floor.
        assert "PLW010" in captured.err
        assert "1.10.0" in captured.err
