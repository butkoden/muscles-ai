from __future__ import annotations

from pathlib import Path
import tomllib


def test_core_dependency_matches_current_muscles_version() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    dependencies = project["project"]["dependencies"]

    assert "muscles>=1.0.0" in dependencies
    assert not any(item.startswith("muscles>=3.") for item in dependencies)
