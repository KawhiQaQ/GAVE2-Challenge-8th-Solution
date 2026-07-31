# Training details

This document complements the quick-start commands in the root README. All
commands assume:

```bash
export PYTHONPATH="$PWD/code/gave2_solution"
export DATA=/absolute/path/GAVE2_preliminary
export REGISTERED=$PWD/cache/registered_ffa
export RUNS=$PWD/runs
```

## Cross-validation

`train.py` creates density/complexity-balanced folds with seed 77. For a full
five-fold V1 study:

```bash
for fold in 0 1 2 3 4; do
  python code/gave2_solution/train.py \
    --task 1 --fold "$fold" --n-folds 5 --data-root "$DATA" \
    --output-dir "$RUNS/cv/task1/fold${fold}" \
    --size 768x1152 --epochs 80 --patience 15 --seed 77 --device cuda

  python code/gave2_solution/train.py \
    --task 2 --fold "$fold" --n-folds 5 --data-root "$DATA" \
    --output-dir "$RUNS/cv/task2/fold${fold}" \
    --size 768x1152 --epochs 80 --patience 15 --seed 77 --device cuda
done
```

TJ009 uses the Fold-0 checkpoints as architecture-compatible initializers for
full-data training; it is not a five-model inference ensemble.

## Task 1 and Task 2 full training

Use the two V54 commands in the root README. Their complete configurations are
also stored in `configs/training/task1_v54.json` and
`configs/training/task2_v54.json`.

## Optic disc and C-zone preparation

Task 3 requires SIVA C-zone masks. After placing the released MNet/DeepCDR HDF5
checkpoint under `weights/external/`, run:

```bash
python code/biomarker_tools/local_optic_disc.py \
  --data-root "$DATA" \
  --weights weights/external/Model_DiscSeg_ORIGA.h5 \
  --output-dir "$PWD/cache/disc_masks" \
  --split all --device cuda

python code/gave2_solution/prepare_zone_c_masks.py \
  --disc-dir "$PWD/cache/disc_masks/training" \
  --output-dir "$PWD/cache/zone_c/training" \
  --report-output "$PWD/cache/zone_c/training_report.json"

python code/gave2_solution/prepare_zone_c_masks.py \
  --disc-dir "$PWD/cache/disc_masks/validation" \
  --output-dir "$PWD/cache/zone_c/validation" \
  --report-output "$PWD/cache/zone_c/validation_report.json"
```

The C zone is the annulus from 1.5 to 2.5 optic-disc diameters around the
minimum-enclosing-circle center.

## Task 3 source training

The same Task 2 V1 Fold-0 checkpoint initializes V2, V3, V8, and V12. Set:

```bash
export INIT_T2="$RUNS/init/task2/fold0/best.pt"
```

### V2: caliber source using original FFA

Do not pass `--ffa-root`; Task 2 then reads the original FFA under `$DATA`.

```bash
python code/gave2_solution/train_full_v2.py \
  --task 2 --data-root "$DATA" --output-dir "$RUNS/task3/v2" \
  --epochs 47 --size 1024x1536 --batch-size 1 \
  --accumulation-steps 2 --learning-rate 1.5e-4 \
  --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0 \
  --seed 77 --device cuda --num-workers 0 --no-pretrained \
  --init-checkpoint "$INIT_T2"
```

### V3: artery-density source using original FFA

```bash
python code/gave2_solution/train_full_v2.py \
  --task 2 --data-root "$DATA" --output-dir "$RUNS/task3/v3" \
  --epochs 47 --size 1024x1536 --batch-size 1 \
  --accumulation-steps 2 --learning-rate 1.5e-4 \
  --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 \
  --seed 77 --device cuda --num-workers 0 --no-pretrained \
  --init-checkpoint "$INIT_T2"
```

The released density factor is derived from out-of-fold training predictions.
It is intentionally fixed in `configs/v3_density_calibration.json` and must not
be estimated from test data.

### V8: registered-FFA parent for vein density

```bash
python code/gave2_solution/train_full_v2.py \
  --task 2 --data-root "$DATA" --ffa-root "$REGISTERED" \
  --output-dir "$RUNS/task3/v8" \
  --epochs 48 --size 1024x1536 --batch-size 1 \
  --accumulation-steps 2 --learning-rate 1.5e-4 \
  --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 \
  --seed 77 --device cuda --num-workers 0 --no-pretrained \
  --init-checkpoint "$INIT_T2"
```

### V12: registered-FFA fractal-dimension source

```bash
python code/gave2_solution/train_full_v2.py \
  --task 2 --v12-phase-geometry --data-root "$DATA" \
  --ffa-root "$REGISTERED" --output-dir "$RUNS/task3/v12" \
  --epochs 48 --size 1024x1536 --batch-size 1 \
  --accumulation-steps 2 --learning-rate 1.5e-4 \
  --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 \
  --seed 77 --device cuda --num-workers 0 \
  --init-checkpoint "$INIT_T2"
```

V12 intentionally retains ImageNet initialization for its new phase encoder;
therefore do not add `--no-pretrained` to this command.

### D0044: low-capacity vein-density refiner

```bash
python code/gave2_solution/train_full_v17_task3_vein.py \
  --data-root "$DATA" --ffa-root "$REGISTERED" \
  --zone-c-dir "$PWD/cache/zone_c/training" \
  --parent-checkpoint "$RUNS/task3/v8/final.pt" \
  --output-dir "$RUNS/task3/d0044" \
  --size 1024x1536 --epochs 24 --accumulation-steps 2 \
  --learning-rate 3e-4 --weight-decay 1e-4 --warmup-epochs 2 \
  --ema-decay 0.995 --seed 77 --device cuda --num-workers 0 \
  --loss-policy vessel_argmax
```

Only the compact refiner is optimized; the V8 parent remains frozen.

## Inference after self-training

Copy or symlink your final checkpoints into the paths listed in
`weights/README.md`, verify their configuration, and run
`code/run_tj009_inference.sh`. The unified runner handles all prediction,
geometry, routing, compaction, and packaging stages.

For a new model rather than an exact TJ009 reproduction, update both
`configs/tj009_route.json` and the expected weight manifest instead of silently
replacing files under an old version name.

