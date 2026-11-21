#!/usr/bin/env python3
"""Freeze, run, and verify a model-independent quantization experiment."""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def name(value):
    check(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9-]*", value), f"Invalid name: {value!r}")
    return value


@contextmanager
def publish(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    with TemporaryDirectory(dir=destination.parent, prefix=".quantit-") as temporary:
        staged = Path(temporary)
        yield staged
        # A complete output is a nonempty directory. A concurrent publisher wins once.
        staged.rename(destination)


def init(project):
    project.mkdir(parents=True, exist_ok=True)
    with (project / "quantit.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "comparison": {
                "model": "model-id", "preprocessing": "preprocessing-id", "num_outputs": 2,
                "reference_dtype": "float32",
            },
            "inputs": ["adapter.py"],
            "targets": "targets.json",
            "suites": {"smoke": "manifests/smoke.json", "full": "manifests/full.json"},
            "reference": ["python", "adapter.py", "reference"],
            "methods": {
                "candidate": {
                    "recipe": {},
                    "calibration": {"manifest": "manifests/train.json", "command": ["python", "adapter.py", "calibrate"]},
                    "evaluate": ["python", "adapter.py", "evaluate"],
                },
            },
        }, handle, indent=2)
        handle.write("\n")


def sample_ids(manifest):
    ids = [row["sample_id"] for row in manifest["samples"]]
    check(ids and all(isinstance(key, str) and key for key in ids), "Expected nonempty string sample IDs")
    check(len(ids) == len(set(ids)), "Duplicate sample ID")
    return ids


def subset(source, destination, count, seed):
    manifest = read_json(source)
    ids = sample_ids(manifest)
    check(1 <= count <= len(ids), f"count must be between 1 and {len(ids)}")
    ranked = sorted(
        zip(ids, manifest["samples"], strict=True),
        key=lambda item: (hashlib.sha256(f"{seed}\0{item[0]}".encode()).digest(), item[0]),
    )
    payload = json.dumps({**manifest, "samples": [row for _, row in ranked[:count]]},
                         sort_keys=True, indent=2, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def command(argv):
    check(isinstance(argv, list) and argv and all(isinstance(arg, str) and arg for arg in argv),
          "An adapter command must be a nonempty JSON array of strings")
    return argv


def freeze(config_file, destination):
    project = config_file.resolve().parent
    config = read_json(config_file)
    comparison = config["comparison"]
    check(isinstance(comparison, dict) and all(
        isinstance(comparison.get(key), str) and comparison[key] for key in ("model", "preprocessing")
    ), "comparison requires model and preprocessing identifiers")
    outputs = comparison["num_outputs"]
    check(type(outputs) is int and outputs > 0, "num_outputs must be a positive integer")
    command(config["reference"])
    inputs = config["inputs"]
    check(isinstance(inputs, list) and inputs and all(isinstance(path, str) and path for path in inputs),
          "inputs must declare the adapter and any other files to freeze")
    inputs = {str((project / path).resolve()): digest(project / path) for path in inputs}
    targets = read_json(project / config["targets"])
    check(isinstance(targets, list) and all(isinstance(x, str) and x for x in targets)
          and len(targets) == len(set(targets)), "targets must be a nonempty list of unique names")
    check(config["suites"] and config["methods"], "Configure at least one suite and one method")
    suites, methods, snapshots, evaluation_ids = {}, {}, {}, set()

    for suite, path in config["suites"].items():
        name(suite)
        manifest = read_json(project / path)
        check(manifest["split"] in ("eval", "validation", "test"), f"{suite}: expected an evaluation split")
        evaluation_ids.update(sample_ids(manifest))
        suites[suite] = f"manifests/eval-{suite}.json"
        snapshots[suites[suite]] = manifest

    for method, spec in config["methods"].items():
        name(method)
        check(method != "reference", "reference is a reserved method name")
        check(isinstance(spec["recipe"], dict), f"{method}: recipe must be an object")
        command(spec["evaluate"])
        method_config = {"recipe": spec["recipe"], "evaluate": spec["evaluate"]}
        if "calibration" in spec:
            calibration = spec["calibration"]
            command(calibration["command"])
            manifest = read_json(project / calibration["manifest"])
            ids = sample_ids(manifest)
            check(manifest["split"] == "train" and evaluation_ids.isdisjoint(ids),
                  f"{method}: calibration must use a disjoint train split")
            snapshot = f"manifests/train-{method}.json"
            snapshots[snapshot] = manifest
            method_config["calibration"] = {"manifest": snapshot, "command": calibration["command"]}
        methods[method] = method_config

    with publish(destination) as staged:
        (staged / "manifests").mkdir()
        for path, data in snapshots.items():
            write_json(staged / path, data)
        write_json(staged / "experiment.lock.json", {
            "schema": 1, "project": str(project), "comparison": comparison, "inputs": inputs, "targets": targets,
            "suites": suites, "reference": config["reference"], "methods": methods,
            "manifests": {path: digest(staged / path) for path in snapshots},
        })


def load_experiment(root):
    lock = root / "experiment.lock.json"
    config = read_json(lock)
    check(config["schema"] == 1, "Unsupported experiment schema")
    for path, expected in config["inputs"].items():
        check(digest(Path(path)) == expected, f"Frozen input changed: {path}")
    for path, expected in config["manifests"].items():
        check(digest(root / path) == expected, f"Frozen manifest changed: {path}")
    return config, digest(lock)


def call_adapter(config, argv, stage, **context):
    path = stage / "context.json"
    write_json(path, {**context, "comparison": config["comparison"], "targets": config["targets"]})
    subprocess.run(
        [*argv, str(path)], cwd=config["project"], env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True,
    )
    path.unlink()


def calibration_files(path):
    files = {}
    for item in sorted(path.rglob("*")):
        check(not item.is_symlink(), f"Calibration payload contains a symlink: {item}")
        if item.is_file():
            files[item.relative_to(path).as_posix()] = digest(item)
    check(files, "Adapter produced an empty calibration payload")
    return files


def calibrate(root, config, lock_hash, method):
    spec = config["methods"][method]
    spec_cal = spec.get("calibration")
    if spec_cal is None:
        return None, None
    manifest = spec_cal["manifest"]
    output = root / "calibration" / method
    binding = {"lock_sha256": lock_hash, "method": method, "manifest_sha256": config["manifests"][manifest]}
    if not output.exists():
        try:
            with publish(output) as stage:
                payload = stage / "payload"
                payload.mkdir()
                local = stage / "manifest.json"
                local.write_bytes((root / manifest).read_bytes())
                call_adapter(config, spec_cal["command"], stage, operation="calibrate", method=method,
                             recipe=spec["recipe"], manifest=str(local), output=str(payload))
                check(digest(local) == binding["manifest_sha256"], "Adapter modified its input manifest")
                local.unlink()
                write_json(stage / "calibration.json", {**binding, "payload": calibration_files(payload)})
        except OSError:
            # Another worker may have calibrated this method. Always validate its output.
            if not (output / "calibration.json").is_file():
                raise
    path = output / "calibration.json"
    record = read_json(path)
    check(record == {**binding, "payload": calibration_files(output / "payload")},
          f"Calibration does not match its frozen manifest or payload: {output}")
    return output / "payload", digest(path)


def load_logits(path, ids, width):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        stored = archive["sample_ids"]
        logits = archive["logits"]
    check(stored.ndim == 1 and stored.dtype.kind == "U" and stored.tolist() == ids,
          f"Sample IDs do not match the manifest in order: {path}")
    check(logits.shape == (len(ids), width) and logits.dtype == np.float32 and np.isfinite(logits).all(),
          f"Expected finite FP32 logits of shape {(len(ids), width)}: {path}")
    return logits


def compare(stage, base, ids, logits):
    import numpy as np

    mse = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(ids), 4096):
        end = start + 4096
        delta = logits[start:end].astype(np.float64) - base[start:end]
        mse[start:end] = (delta ** 2).mean(axis=1)
    maximum = int(mse.argmax())
    write_json(stage / "metrics.json", {
        "sample_count": len(ids), "mean_mse": float(mse.mean()), "p95_mse": float(np.percentile(mse, 95)),
        "max_mse": float(mse[maximum]), "max_mse_sample_id": ids[maximum],
    })
    with (stage / "per_sample_mse.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "mse"))
        writer.writerows(zip(ids, mse, strict=True))


def run_record(path, binding):
    files = ["logits.npz", "applied_targets.json"]
    if binding["method"] != "reference" and "shard" not in binding:
        files.extend(("metrics.json", "per_sample_mse.csv"))
    return {**binding, "files": {file: digest(path / file) for file in files}}


def load_run(path, binding, ids, width):
    record = read_json(path / "run.json")
    check(record == run_record(path, binding), f"Run metadata or output files changed: {path}")
    return load_logits(path / "logits.npz", ids, width)


def finalize(stage, base, binding, ids, logits):
    if base is not None:
        compare(stage, base, ids, logits)
    write_json(stage / "run.json", run_record(stage, binding))


def infer(path, base, config, manifest, ids, binding, calibration):
    width = config["comparison"]["num_outputs"]
    if path.exists():
        load_run(path, binding, ids, width)
        return
    method = binding["method"]
    is_reference = method == "reference"
    spec = {} if is_reference else config["methods"][method]
    argv = config["reference"] if is_reference else spec["evaluate"]
    with publish(path) as stage:
        local = stage / "manifest.json"
        write_json(local, manifest)
        call_adapter(config, argv, stage, operation="reference" if is_reference else "evaluate", method=method,
                     recipe=spec.get("recipe", {}), calibration=str(calibration) if calibration else None,
                     manifest=str(local), output=str(stage))
        check(read_json(local) == manifest, "Adapter modified its input manifest")
        local.unlink()
        targets = read_json(stage / "applied_targets.json")
        check(targets == ([] if is_reference else config["targets"]), f"Applied target list does not match: {path}")
        logits = load_logits(stage / "logits.npz", ids, width)
        finalize(stage, base, binding, ids, logits)


def run(args):
    root = args.experiment.resolve()
    config, lock_hash = load_experiment(root)
    if args.action == "calibrate":
        check(args.method in config["methods"], f"Unknown candidate method: {args.method}")
        path, _ = calibrate(root, config, lock_hash, args.method)
        print(path or "Method has no calibration stage")
        return

    suite, method, shards = args.suite, args.method, args.shards
    check(suite in config["suites"], f"Unknown suite: {suite}")
    check(method == "reference" or method in config["methods"], f"Unknown method: {method}")
    manifest_path = config["suites"][suite]
    manifest = read_json(root / manifest_path)
    ids = sample_ids(manifest)
    check(1 <= shards <= len(ids), f"shards must be between 1 and {len(ids)}")
    if args.action == "merge":
        check(shards > 1, "merge requires at least two shards")
    elif args.shard_index is not None:
        check(shards > 1 and 0 <= args.shard_index < shards, "shard index must be within the declared shard count")

    suite_root = root / "suites" / suite
    reference = suite_root / "reference"
    output = reference if method == "reference" else suite_root / "runs" / method
    shard_only = args.action == "run" and args.shard_index is not None
    common = {"lock_sha256": lock_hash, "suite": suite, "validation_sha256": config["manifests"][manifest_path]}
    width = config["comparison"]["num_outputs"]
    base = None
    if method != "reference" and not shard_only:
        check((reference / "run.json").is_file(), f"Generate the reference first: {reference}")
        base = load_run(reference, {**common, "method": "reference", "calibration_sha256": None}, ids, width)
    calibration, cal_hash = (None, None) if method == "reference" else calibrate(root, config, lock_hash, method)
    binding = {**common, "method": method, "calibration_sha256": cal_hash}
    final_binding = binding if base is None else {**binding, "reference_run_sha256": digest(reference / "run.json")}
    if output.exists() and not shard_only:
        load_run(output, final_binding, ids, width)
    elif shards == 1:
        infer(output, base, config, manifest, ids, final_binding, calibration)
    else:
        def shard(index):
            start, end = index * len(ids) // shards, (index + 1) * len(ids) // shards
            path = suite_root / "shards" / method / str(shards) / str(index)
            info = {"count": shards, "index": index, "start": start, "end": end}
            return path, start, end, {**binding, "shard": info}

        if args.action == "run":
            indices = [args.shard_index] if shard_only else range(shards)
            for index in indices:
                path, start, end, part = shard(index)
                infer(path, None, config, {**manifest, "samples": manifest["samples"][start:end]},
                      ids[start:end], part, calibration)
            if shard_only:
                print(path)
                return

        import numpy as np

        logits = np.empty((len(ids), width), dtype=np.float32)
        targets = [] if method == "reference" else config["targets"]
        for index in range(shards):
            path, start, end, part = shard(index)
            check(read_json(path / "applied_targets.json") == targets, f"Shard {index} has different targets")
            logits[start:end] = load_run(path, part, ids[start:end], width)
        with publish(output) as stage:
            np.savez_compressed(stage / "logits.npz", sample_ids=np.array(ids), logits=logits)
            write_json(stage / "applied_targets.json", targets)
            finalize(stage, base, final_binding, ids, logits)

    print(output)
    if method != "reference":
        print((output / "metrics.json").read_text().strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init").add_argument("project", type=Path)
    selecting = sub.add_parser("subset")
    selecting.add_argument("source", type=Path)
    selecting.add_argument("destination", type=Path)
    selecting.add_argument("--count", required=True, type=int)
    selecting.add_argument("--seed", type=int, default=0)
    freezing = sub.add_parser("freeze")
    freezing.add_argument("config", type=Path)
    freezing.add_argument("experiment", type=Path)
    for action in ("calibrate", "run", "merge"):
        child = sub.add_parser(action)
        child.add_argument("experiment", type=Path)
        child.add_argument("--method", required=True)
        if action != "calibrate":
            child.add_argument("--suite", required=True)
            child.add_argument("--shards", type=int, default=1)
        if action == "run":
            child.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    try:
        if args.action == "init":
            init(args.project)
        elif args.action == "subset":
            subset(args.source, args.destination, args.count, args.seed)
        elif args.action == "freeze":
            freeze(args.config, args.experiment.resolve())
        else:
            run(args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"quantit: {error}\n")


if __name__ == "__main__":
    main()
