"""
File: test_repository_hygiene.py

Description:
Repository-wide encoding and syntax checks for every authored Python file.

Responsibilities:
- Reject UTF-8 byte-order marks that tools may misread.
- Verify every source, script, and test module parses as UTF-8 Python.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _python_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for directory in ("src", "scripts", "tests"):
        for path in sorted((project_root / directory).rglob("*.py")):
            if "__pycache__" not in path.parts:
                files.append(path)
    return files


def test_every_python_file_has_no_bom_and_parses(project_root: Path) -> None:
    files = _python_files(project_root)
    assert files
    problems: list[str] = []
    for path in files:
        relative = path.relative_to(project_root)
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            problems.append(f"{relative}: UTF-8 BOM")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"{relative}: not UTF-8 ({exc})")
            continue
        try:
            ast.parse(text, filename=str(relative))
        except SyntaxError as exc:
            problems.append(f"{relative}: {exc}")
    assert not problems, "\n".join(problems)
