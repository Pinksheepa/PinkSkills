---
name: quantit
description: Build reproducible post-training quantization experiments for any model or quantization method. Use to set up FP32 references, freeze calibration and evaluation data, compare quantized outputs with MSE, run smoke and full validation, shard evaluation, or prepare an algorithm implementation for export as a child skill.
---

# Quantit

Use the model's existing inference environment and data loader. Supply thin adapters for reference inference, optional calibration, and candidate inference. Keep quantization math and backend selection in the adapter. Follow the [configuration and adapter contract](references/workflow.md).

1. Run `scripts/quantit.py init PROJECT` to create the config. Set the reference precision, comparable output vector, shared target list, input files, adapter commands, and evaluation suites. Use `subset` when a smoke set or calibration prefix needs deterministic sampling.
2. Run `freeze CONFIG EXPERIMENT`. Use the resulting lock and manifests for every method in the comparison.
3. Run `run EXPERIMENT --suite smoke --method reference`, followed by one smoke run per method. Run `calibrate EXPERIMENT --method METHOD` separately when useful. Candidate runs automatically produce or verify the cache when calibration is configured.
4. After smoke passes, run the reference and candidates for the requested evaluation suite. Use `--shards N --shard-index I` for independent workers and `merge` after all shards finish.
5. Report the actual sample count and mean, P95, and maximum MSE from each candidate's `metrics.json`. Include the experiment lock and method recipe when presenting results.

Use [Quantit Export Algorithm](../quantit-export-algorithm/SKILL.md) to package a reusable algorithm and its recipe as a standalone skill.
