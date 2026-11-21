# Configuration and adapter contract

Run the CLI with Python 3.11 or later. Install NumPy in the evaluation environment. Use the absolute path to the skill script and put the project config beside the adapters.

```bash
python "$QUANTIT" init "$PROJECT"
python "$QUANTIT" subset "$PROJECT/manifests/train.json" "$PROJECT/manifests/cal.json" --count "$CALIBRATION_COUNT" --seed "$SEED"
python "$QUANTIT" freeze "$PROJECT/quantit.json" "$EXPERIMENT"
python "$QUANTIT" run "$EXPERIMENT" --suite smoke --method reference
python "$QUANTIT" run "$EXPERIMENT" --suite smoke --method candidate
```

`subset` ranks unique sample IDs by SHA-256 with the seed. The same seed yields nested prefixes across calibration sizes, independently of the source row order. Use consistent source sample IDs across splits so the freeze check can catch overlap. A config can include full, smoke, and expanded suites that share IDs.

## Files to supply

- `quantit.json` defines a comparison, input files, target list, suite manifests, a reference command, and candidate methods. Path fields resolve relative to the config directory. Commands are argument arrays and run from that directory. Specify an absolute interpreter path if using a separate Python environment.
- `comparison` requires `model`, `preprocessing`, and `num_outputs`. Record output meaning and order, precision, backend, seed, and checkpoint revision there as needed. Commands control actual backend behavior. Declare the adapter, method code, and relevant model, data, and environment artifacts as `inputs` so Quantit can check their bytes. Inputs are individual files. An asset manifest, when used as an input, pins its own bytes. Have the adapter verify any asset hashes it lists.
- `targets.json` is an ordered JSON array of unique target names. Candidates in one comparison use that exact list. If coverage is intentionally different, use a separate comparison.
- A sample manifest has the form `{"split":"train","samples":[{"sample_id":"example-0001","path":"data/sample.bin"}]}`. Give evaluation suites the `eval`, `validation`, or `test` split. Interpret payload fields such as `path` from the project directory or use absolute locators. The runner preserves payloads and order.
- Add one entry under `methods` for each candidate. Each entry has a `recipe` object and an `evaluate` command. For calibrated methods, add `"calibration":{"manifest":"manifests/cal.json","command":["python","adapter.py","calibrate"]}`. Candidate definitions without that field run without a calibration cache. To compare calibration sizes under one lock, declare separate method entries and use the same recipe with the corresponding nested manifests.

## Adapter inputs and outputs

Quantit appends one absolute path to a JSON context file to the configured command. The context contains `operation`, `method`, `recipe`, `comparison`, `targets`, `manifest`, and `output`. Evaluation and reference calls also get `calibration`, which is the verified cache directory or null. Read the manifest in order and keep inputs read-only.

- For `calibrate`, write the calibration artifacts to `output`. Quantit hashes the files and binds the cache to the lock, method, and calibration manifest. The same cache is used for all suites in that experiment.
- For `reference` and `evaluate`, write `logits.npz` and `applied_targets.json` to `output`. The NPZ contains `sample_ids` as a 1-D NumPy Unicode string array in manifest order and `logits` as a finite FP32 array of shape `(sample_count, num_outputs)`. A fixed-width embedding or other comparable output vector can use the same field. Cast for storage only, and record the actual reference and candidate compute precision in the configuration.
- Write the targets actually quantized, in the configured order, to `applied_targets.json`. The reference writes `[]`. Use identical weights, preprocessing, and output semantics for the reference and each candidate.

## Evaluation

Run the reference for a suite before finalizing candidates. Use `run --shards N` to run and merge all shards locally, or launch each `run --shards N --shard-index I` separately with indices from zero and finish with `merge --shards N`. Each command also takes the same experiment, suite, and method. Results are published atomically.

The runner verifies the manifests, exact sample order, output shape and dtype, target coverage, reference, and cache bindings. Candidate results land in `suites/<suite>/runs/<method>/` under the experiment. Each sample's MSE is the mean squared output difference against its reference, calculated in float64. Report `metrics.json` with the sample count and `per_sample_mse.csv` for individual errors. Freeze a new experiment when inputs or settings change.
