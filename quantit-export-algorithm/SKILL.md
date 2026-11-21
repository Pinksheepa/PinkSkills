---
name: quantit-export-algorithm
description: Export an existing quantization algorithm, explicit source files, and its recipe as a standalone skill. Use when packaging a quantization method for reuse or verifying a previously exported Quantit algorithm bundle.
---

# Quantit Export Algorithm

Create a JSON spec in the root of the algorithm source to be exported. Declare the smallest file set that includes the entrypoint and its local dependencies. Set the recipe to the implementation's actual quantization settings. Commands are optional, and their argument arrays execute from the bundled source directory.

```json
{
  "name": "example-quantizer",
  "description": "Apply the example quantizer using its calibrated recipe.",
  "entrypoint": "algorithm/quantize.py",
  "files": ["algorithm/quantize.py", "requirements.txt"],
  "recipe": {"weight_bits": 4, "activation_bits": 8, "weight_granularity": "per-channel"},
  "commands": {"quantize": ["python", "algorithm/quantize.py"]}
}
```

Use Python 3.11 or later. Set `EXPORTER` to the absolute path to this skill's `scripts/export_algorithm.py`. Set `SPEC` to the spec file and `DESTINATION` to the parent directory for the new skill.

```bash
python "$EXPORTER" "$SPEC" "$DESTINATION"
python "$DESTINATION/example-quantizer/scripts/bundle.py" verify
```

The exporter copies exactly the declared sources, records their SHA-256 hashes with the recipe, and creates a `SKILL.md` and a verifier. Local paths must be normalized relative paths inside the spec directory. To run a declared command, follow the generated skill and use the environment and arguments required by that algorithm. A generated skill can be installed and verified by itself.
