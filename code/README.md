# Code map

- `run_inference.sh`: complete VascFusion inference and submission entry point.
- `assemble_biomarkers.py`: seven-field VascFusion-Quant assembly.
- `gave2_solution/gave2v1/`: datasets, losses, metrics, and model definitions.
- `gave2_solution/train*.py`: cross-validation and full-data training.
- `gave2_solution/predict*.py`: checkpoint-specific probability inference.
- `gave2_solution/register_ffa_minima.py`: label-free FFA-to-CFP registration.
- `biomarker_tools/`: optic-disc inference and SIVA-style biomarker extraction.
- `external/MINIMA/`: upstream MINIMA source snapshot and license.

Set the module path before invoking Python entry points directly:

```bash
export PYTHONPATH="$PWD/code/gave2_solution"
```

For most users, the root README and `run_inference.sh` are the only
required entry points.
