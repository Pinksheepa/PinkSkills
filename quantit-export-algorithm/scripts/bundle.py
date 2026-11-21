#!/usr/bin/env python3
"""Verify or run a command from an exported algorithm skill."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath


def source_path(root, value):
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value:
        raise ValueError(f"Expected a normalized relative source path: {value!r}")
    relative = root
    for part in path.parts:
        relative = relative / part
        if relative.is_symlink():
            raise ValueError(f"Source paths must not traverse symlinks: {relative}")
    if not relative.is_file():
        raise ValueError(f"Source file missing: {relative}")
    return relative


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(skill):
    spec = json.loads((skill / "algorithm.json").read_bytes())
    root = skill / "source"
    files = spec["files"]
    if spec["schema"] != 1 or not files or spec["entrypoint"] not in files:
        raise ValueError("Invalid algorithm bundle")
    if root.is_symlink():
        raise ValueError("Bundle source directory is a symlink")
    for path, expected in files.items():
        if fingerprint(source_path(root, path)) != expected:
            raise ValueError(f"Source drift: {path}")
    discovered = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        if path.is_file():
            discovered.add(path.relative_to(root).as_posix())
    if discovered != files.keys():
        raise ValueError("Source file set differs from the bundle manifest")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("verify")
    runner = sub.add_parser("run")
    runner.add_argument("command")
    runner.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    skill = Path(__file__).resolve().parents[1]
    try:
        spec = verify(skill)
        if args.action == "verify":
            print(f"Verified {len(spec['files'])} source files for {spec['name']}")
        else:
            argv = spec["commands"].get(args.command)
            if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
                raise ValueError(f"Unknown or invalid command: {args.command}")
            extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            subprocess.run(
                [*argv, *extra], cwd=skill / "source", env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True,
            )
    except (KeyError, TypeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"algorithm bundle: {error}\n")


if __name__ == "__main__":
    main()
