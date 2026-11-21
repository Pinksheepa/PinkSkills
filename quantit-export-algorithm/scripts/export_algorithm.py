#!/usr/bin/env python3
"""Export declared algorithm source and recipe as an installable Codex skill."""

import argparse
import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from bundle import fingerprint, source_path, verify


def export(spec_path, destination):
    source = spec_path.resolve().parent
    spec = json.loads(spec_path.read_bytes())
    name = spec["name"]
    if not isinstance(name, str) or len(name) > 64 or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Use a lowercase skill name up to 64 characters")
    description = spec["description"]
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError("Specify a descriptive skill description up to 1024 characters")
    if any(char in description for char in "<>"):
        raise ValueError("Skill description must not contain angle brackets")
    entrypoint = spec["entrypoint"]
    files = spec["files"]
    if (not isinstance(files, list) or not files or not all(isinstance(path, str) for path in files)
            or len(files) != len(set(files)) or entrypoint not in files):
        raise ValueError("Declare unique source files including the entrypoint")
    declared = {path: source_path(source, path) for path in files}
    if not isinstance(spec["recipe"], dict) or not spec["recipe"]:
        raise ValueError("recipe must be a nonempty JSON object")
    commands = spec.get("commands", {})
    if not isinstance(commands, dict) or any(
        not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv)
        for argv in commands.values()
    ):
        raise ValueError("commands must map names to nonempty argument arrays")

    target = destination.resolve() / name
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix=".quantit-export-") as temporary:
        stage = Path(temporary)
        hashes = {}
        for path, original in sorted(declared.items()):
            copy = stage / "source" / path
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copy)
            hashes[path] = fingerprint(copy)
        manifest = {
            "schema": 1, "name": name, "entrypoint": entrypoint, "recipe": spec["recipe"],
            "files": hashes, "commands": commands,
        }
        (stage / "algorithm.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (stage / "scripts").mkdir()
        shutil.copy2(Path(__file__).with_name("bundle.py"), stage / "scripts" / "bundle.py")
        (stage / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {json.dumps(description, ensure_ascii=False)}\n---\n\n"
            f"# {name}\n\n"
            "Read [algorithm.json](algorithm.json) for the entrypoint, recipe, source hashes, and optional commands. "
            "Run the bundled verifier before integrating or executing the implementation.\n\n"
            "- Run `python scripts/bundle.py verify` from this skill directory.\n"
            "- Integrate the declared entrypoint from the source directory. Use the recorded recipe for its settings.\n"
            "- To execute a declared command, use `python scripts/bundle.py run <command> -- <arguments>`. "
            "The runner verifies the bundle and starts the command from the source directory.\n",
            encoding="utf-8",
        )
        verify(stage)
        stage.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("destination", type=Path, help="Parent directory for the generated skill")
    args = parser.parse_args()
    try:
        print(export(args.spec, args.destination))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"quantit export: {error}\n")


if __name__ == "__main__":
    main()
