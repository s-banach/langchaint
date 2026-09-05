"""Verify dependency rules against temporary source files without importing them."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts.check_architecture import check_architecture


def _check_dependencies(tmp_path: Path, files: Mapping[str, str]) -> int:
    package = tmp_path / "src" / "langchaint"
    package.mkdir(parents=True)
    _ = (package / "__init__.py").write_text("")
    for relative_path, source in files.items():
        path = package / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        for directory in path.parents:
            if directory == package:
                break
            (directory / "__init__.py").touch()
        _ = path.write_text(source)
    return check_architecture(project_root=tmp_path)


@pytest.mark.parametrize(
    "files",
    [
        {
            "new_a.py": "import langchaint.new_b\n",
            "new_b.py": "import langchaint.new_a\n",
        },
        {
            "new_a.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import langchaint.new_b\n",
            "new_b.py": "import langchaint.new_a\n",
        },
        {
            "new_a.py": "def run():\n    import langchaint.new_b\n",
            "new_b.py": "import langchaint.new_a\n",
        },
        {
            "generation/a.py": "import langchaint.generation.b\n",
            "generation/b.py": "import langchaint.generation.c\n",
            "generation/c.py": "import langchaint.generation.a\n",
        },
        {
            "common/a.py": "import langchaint.common.b\n",
            "common/b.py": "import langchaint.common.a\n",
        },
        {
            "generation/__init__.py": "import langchaint.generation.a\n",
            "generation/a.py": "import langchaint.generation\n",
        },
        {
            "new_a.py": "from langchaint.generation import b\n",
            "generation/b.py": "import langchaint.new_a\n",
        },
        {
            "new_a.py": "import os, langchaint.new_b as b\n",
            "new_b.py": "import langchaint.new_a\n",
        },
        {
            "__init__.py": "import langchaint.new_a\n",
            "new_a.py": "from langchaint import exported\n",
        },
    ],
    ids=[
        "new-files",
        "type-checking",
        "function-local",
        "same-package",
        "common-cycle",
        "package-init",
        "named-submodule",
        "multiple-aliases",
        "root-init",
    ],
)
def test_architecture_rejects_cycles(
    tmp_path: Path, files: Mapping[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject cycles even when imports are deferred or files share a package."""
    result = _check_dependencies(tmp_path, files)
    output = capsys.readouterr()
    assert result != 0, output.out + output.err
    assert "Circular file dependency" in output.err


@pytest.mark.parametrize(
    "relative_path",
    [
        "common/new_file.py",
        "common/__init__.py",
        "common/nested/new_file.py",
        "common/nested/__init__.py",
    ],
)
def test_architecture_rejects_common_importing_generation(
    tmp_path: Path, relative_path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject imports out of common, including package entrypoints and new descendants."""
    result = _check_dependencies(
        tmp_path,
        {relative_path: "import langchaint.generation.llm\n", "generation/llm.py": ""},
    )
    output = capsys.readouterr()
    assert result != 0, output.out + output.err
    assert "langchaint.generation.llm" in output.out + output.err


def test_architecture_accepts_one_way_dependencies_without_executing_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accept a dependency chain while leaving top-level Python statements unexecuted."""
    result = _check_dependencies(
        tmp_path,
        {
            "generation/llm.py": "import langchaint.common.messages\nraise AssertionError('must not execute')\n",
            "common/messages.py": "import langchaint.common.checked_copy\n",
            "common/checked_copy.py": "",
        },
    )
    output = capsys.readouterr()
    assert result == 0, output.out + output.err


def test_architecture_rejects_common_importing_public_exports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject a dependency through the top-level application exports."""
    result = _check_dependencies(
        tmp_path,
        {
            "__init__.py": "from langchaint.generation.llm import LLM\n",
            "common/messages.py": "from langchaint import LLM\n",
            "generation/llm.py": "class LLM: pass\n",
        },
    )
    output = capsys.readouterr()
    assert result != 0, output.out + output.err
    assert "langchaint" in output.out + output.err


def test_architecture_rejects_package_cycles_without_file_cycles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject reciprocal package dependencies assembled from separate file pairs."""
    result = _check_dependencies(
        tmp_path,
        {
            "generation/a.py": "import langchaint.openai.b\n",
            "openai/b.py": "",
            "openai/c.py": "import langchaint.generation.d\n",
            "generation/d.py": "",
        },
    )
    output = capsys.readouterr()
    assert result != 0, output.out + output.err
    assert "Circular package dependency" in output.err


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        ("import langchaint.missing\n", "Unresolved internal import"),
        ("from langchaint.missing import Symbol\n", "Unresolved internal import"),
        ("from . import existing\n", "Relative imports are forbidden"),
    ],
)
def test_architecture_rejects_invalid_source(
    tmp_path: Path, source: str, diagnostic: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail visibly when source cannot produce a complete import graph."""
    result = _check_dependencies(tmp_path, {"new_a.py": source, "existing.py": ""})
    output = capsys.readouterr()
    assert result != 0
    assert diagnostic in output.err


def test_architecture_ignores_external_imports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Do not require third-party dependencies to be installed."""
    result = _check_dependencies(
        tmp_path,
        {"common/a.py": "import unavailable_sdk\nfrom langchaint_extra import Symbol\n"},
    )
    output = capsys.readouterr()
    assert result == 0, output.out + output.err
