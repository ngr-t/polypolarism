"""Tests for version detection and the unsupported-version warning."""

from __future__ import annotations

from pathlib import Path

from polypolarism.version_check import (
    PANDERA_FLOOR,
    POLARS_FLOOR,
    DetectedVersion,
    Version,
    VersionInfo,
    check_versions,
    detect_versions,
    find_project_root,
)


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


class TestDetectMissingFiles:
    def test_no_project_root(self):
        info = detect_versions(None)
        assert info.polars is None
        assert info.pandera is None

    def test_malformed_pyproject(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("not = valid = toml")
        info = detect_versions(tmp_path)
        assert info.polars is None


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

    def test_below_floor_emits_warning(self, tmp_path: Path, capsys):
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
        assert "PLW010" in captured.err
        assert "polars" in captured.err.lower()

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
        main([str(py), "--no-color", "--polars-version", "1.32.0"])
        captured = capsys.readouterr()
        assert "PLW010" not in captured.err
