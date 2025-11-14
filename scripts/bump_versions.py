#!/usr/bin/env python
"""
Synchronize version across all packages.

Usage:
  python scripts/bump_versions.py -v 0.2.0 [--dry-run]

Targets:
  - Root: setup.py, pyproject.toml (if [project].version present)
  - Packages: packages/*/setup.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple


ROOT = Path(__file__).resolve().parents[1]


def find_files() -> Iterable[Path]:
    # Root setup/pyproject
    for name in ("setup.py", "pyproject.toml"):
        p = ROOT / name
        if p.exists():
            yield p

    # Package setup.py files
    pkgs_dir = ROOT / "packages"
    if pkgs_dir.exists():
        for setup in pkgs_dir.glob("*/setup.py"):
            yield setup


def replace_version_in_setup_py(text: str, version: str) -> Tuple[str, bool]:
    # Match version="x.y.z" with optional spaces
    pattern = r"version\s*=\s*([\"\'])\d+\.\d+\.\d+\1"
    new_text, n = re.subn(pattern, f'version="{version}"', text, count=1)
    return new_text, n > 0


def replace_version_in_pyproject(text: str, version: str) -> Tuple[str, bool]:
    # Try to update only inside [project] section if it exists
    lines = text.splitlines()
    in_project = False
    changed = False
    for i, line in enumerate(lines):
        if re.match(r"\s*\[project\]\s*$", line):
            in_project = True
            continue
        if in_project and re.match(r"\s*\[.+\]\s*$", line):
            # Reached next section without finding version
            in_project = False
        if in_project and re.match(r"\s*version\s*=\s*['\"]\d+\.\d+\.\d+['\"]\s*$", line):
            lines[i] = re.sub(r"(['\"])\d+\.\d+\.\d+(['\"])", f'"{version}"', line)
            changed = True
            break
    if not changed:
        # Fallback: change the first top-level version assignment
        for i, line in enumerate(lines):
            if re.match(r"\s*version\s*=\s*['\"]\d+\.\d+\.\d+['\"]\s*$", line):
                lines[i] = re.sub(r"(['\"])\d+\.\d+\.\d+(['\"])", f'"{version}"', line)
                changed = True
                break
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), changed


def bump_file(path: Path, version: str, dry_run: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    if path.name == "setup.py":
        new_text, changed = replace_version_in_setup_py(text, version)
    elif path.name == "pyproject.toml":
        new_text, changed = replace_version_in_pyproject(text, version)
    else:
        return False

    if changed and not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--version", required=True, help="Target version, e.g., 0.2.0")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would change")
    args = ap.parse_args()

    changed_any = False
    for fp in find_files():
        changed = bump_file(fp, args.version, args.dry_run)
        status = "UPDATED" if changed else "SKIPPED"
        print(f"[{status}] {fp.relative_to(ROOT)}")
        changed_any = changed_any or changed

    if not changed_any:
        print("No version fields matched. Verify your setup.py and pyproject.toml formats.")
    else:
        print(f"Done. Version set to {args.version}.")


if __name__ == "__main__":
    main()
