<div align="center">

# VascFusion — 8th-Place Solution for the MICCAI 2026 GAVE2 Challenge

[English](README.md) | [简体中文](README_zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-ee4c2c.svg)](https://pytorch.org/)
[![Challenge](https://img.shields.io/badge/MICCAI%202026-GAVE2-6f42c1.svg)](https://aistudio.baidu.com/competition/detail/1463/0/introduction)

</div>

This repository contains the training and inference code for **VascFusion**,
our 8th-place solution to the MICCAI 2026 GAVE2 Challenge.
VascFusion covers all three tasks:

1. CFP-only retinal artery/vein segmentation;
2. CFP + FFA cross-modal artery/vein segmentation;
3. CRAE, CRVE, AVR, artery/vein density, and artery/vein fractal dimension.

## Results

| Method | Task 1 | Task 2 | Task 3 | Overall |
|---|---:|---:|---:|---:|
| VascFusion | 8.29660 | 8.31706 | 7.39370 | **7.94362** |

## Method overview

VascFusion combines modality-adaptive vessel segmentation with anatomy-guided
biomarker measurement. It preserves full-resolution vascular detail while
using angiographic phase information only where the task permits it.

### VascFusion-Seg: modality-adaptive A/V segmentation

- ConvNeXt-Tiny multi-scale CFP encoder;
- U-Net-like decoder with a native-resolution detail stem;
- semantic, generic-vessel, and class-centerline heads;
- five shared probability-space recursive refinement steps;
- artery/vein topology-aware training losses;
- CFP-only optical-density detail features for Task 1;
- gated multi-scale fusion of registered early/late FFA for Task 2.

Task 1 is **strictly CFP-only**. Neither training nor inference reads FFA for
Task 1.

### VascFusion-Quant: anatomy-guided biomarker quantification

Task 3 is not a black-box scalar regressor. We first extract biomarkers from
segmentation probabilities using SIVA-style geometry. Specialized measurement
branches are optimized for vessel caliber, density, and network complexity:

| Biomarker | Measurement strategy |
|---|---|
| CRAE, CRVE, AVR | caliber-oriented A/V masks and the revised Knudtson–Hubbard formulas |
| artery density | vessel-aware artery masks with a training-only OOF calibration |
| vein density | a compact C-zone vein refinement branch |
| artery fractal dimension | branch-consistent artery skeleton followed by box counting |
| vein fractal dimension | phase-aware vein geometry marginalized over three fixed thresholds |

## Repository layout

```text
.
├── README.md / README_zh-CN.md
├── code/
│   ├── run_inference.sh             # end-to-end inference and packaging
│   ├── assemble_biomarkers.py       # seven-field biomarker assembly
│   ├── biomarker_tools/             # optic-disc and biomarker geometry
│   ├── external/MINIMA/             # Apache-2.0 MINIMA source snapshot
│   └── gave2_solution/              # models, training, prediction, postprocess
├── configs/
│   ├── training/                    # released training configurations
│   ├── inference_route.json
│   ├── artery_density_calibration.json
│   └── weights_manifest.json
├── weights/                         # intentionally empty; expected paths only
├── environment.yml
└── requirements.txt
```

## Installation

The verified training/inference machine used Python 3.11, PyTorch 2.4.0,
CUDA 12.1, and an RTX 4090.

```bash
git clone git@github.com:KawhiQaQ/GAVE2-Challenge-8th-Solution.git
cd GAVE2-Challenge-8th-Solution

conda env create -f environment.yml
conda activate gave2
export PYTHONPATH="$PWD/code/gave2_solution"
```

Alternatively, install PyTorch for your CUDA version first and then run:

```bash
pip install -r requirements.txt
```

## Dataset preparation

Download the GAVE2 data from the official challenge page and organize it as
follows. The code never downloads or redistributes challenge data.

```text
GAVE2_preliminary/
├── training/
│   ├── images/g_xxx.png
│   ├── masks/g_xxx.png
│   ├── av/g_xxx.png
│   ├── FFA_A/g_xxx.png
│   ├── FFA_AV/g_xxx.png
│   └── biomarker/g_xxx.txt
└── validation/
    ├── images/g_xxx.png
    ├── masks/g_xxx.png
    ├── FFA_A/g_xxx.png
    └── FFA_AV/g_xxx.png
```

`masks/` denotes the released retinal field-of-view mask, while `av/` denotes
the RGB artery/vessel/vein annotation. Images are expected at the official
1536×1024 resolution.

## Checkpoints

The released checkpoints are available from
[Baidu Netdisk](https://pan.baidu.com/s/1tqCe97lvXGM0zn1VvUlPMA?pwd=29v6)
(extraction code: `29v6`). Download the complete `weights/` directory and place
it directly under the repository root. The download preserves the original
training archive names, so create the semantic aliases expected by the public
inference code:

```bash
ln -sf v1_task1_fold0/best.pt weights/initialization/task1_fold0.pt
ln -sf v1_task2_fold0/best.pt weights/initialization/task2_fold0.pt
ln -sf tj009_v54_full/final.pt weights/task1/segmentation_cfp.pt
ln -sf tj009_v54_full/final.pt weights/task2/segmentation_multimodal.pt
ln -sf v2_task2_full/final.pt weights/task3/caliber_expert.pt
ln -sf v3_task2_full/final.pt weights/task3/artery_density_expert.pt
ln -sf v8_task2_full/final.pt weights/task3/vein_density_parent.pt
ln -sf v21_v12_fd_full/final.pt weights/task3/phase_geometry_expert.pt
ln -sf d0044_vein_density_full/final.pt weights/task3/vein_density_refiner.pt
```

The seven Task 1/2/3 checkpoints are required for inference. The two
initialization checkpoints are needed only to reproduce full training. The
expected paths and SHA256 digests are listed in
[configs/weights_manifest.json](configs/weights_manifest.json) and summarized
in [weights/README.md](weights/README.md).

Two third-party weights are not included in the Netdisk directory. Download
MINIMA-LoFTR from its
[official release](https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt)
to `weights/external/minima_loftr.ckpt`, and place the MNet/DeepCDR optic-disc
checkpoint at `weights/external/Model_DiscSeg_ORIGA.h5`.

## Reproduce VascFusion inference

Once the dataset and checkpoints are in place, run:

```bash
bash code/run_inference.sh \
  --data-root /absolute/path/GAVE2_private \
  --work-root /absolute/path/vascfusion_work \
  --output-zip /absolute/path/kawhi00.zip \
  --python "$(which python)" \
  --device cuda
```

The input root must contain
`validation/{images,masks,FFA_A,FFA_AV}`. Case IDs must form one contiguous
`g_###` interval. The script performs:

1. independent early/late FFA-to-CFP registration with MINIMA-LoFTR;
2. optic-disc detection and SIVA C-zone construction;
3. VascFusion-Seg inference for Task 1 and Task 2;
4. caliber-, density-, and phase-geometry probability inference;
5. deterministic VascFusion-Quant biomarker extraction and assembly;
6. threshold-preserving probability compaction;
7. submission validation and ZIP packaging.

The result contains exactly:

```text
Task1/g_xxx.png  # RGB = artery, all vessel, vein probabilities
Task2/g_xxx.png  # RGB = artery, all vessel, vein probabilities
Task3/g_xxx.txt  # seven biomarker values
```

The runner refuses to overwrite an existing work directory or output ZIP. If
you already have audited registration/disc/zone caches, see
[`code/run_inference.sh --help`](code/run_inference.sh) for the
optional cache arguments.

## Training

The following commands expose both cross-validation and full-data training.
They are the actual entry points used by the released system; paths may be
changed, but the hyperparameters should remain fixed for exact reproduction.

### 1. Train the stage-one initialization checkpoints

The final VascFusion-Seg models are warm-started from corresponding Fold-0
stage-one checkpoints. Train Fold 0 as follows (use folds 0–4 for full CV):

```bash
export DATA=/absolute/path/GAVE2_preliminary
export RUNS=$PWD/runs

python code/gave2_solution/train.py \
  --task 1 --fold 0 --n-folds 5 --data-root "$DATA" \
  --output-dir "$RUNS/init/task1/fold0" \
  --size 768x1152 --epochs 80 --patience 15 --seed 77 --device cuda

python code/gave2_solution/train.py \
  --task 2 --fold 0 --n-folds 5 --data-root "$DATA" \
  --output-dir "$RUNS/init/task2/fold0" \
  --size 768x1152 --epochs 80 --patience 15 --seed 77 --device cuda
```

### 2. Register FFA for Task 2

```bash
export REGISTERED=$PWD/cache/registered_ffa

python code/gave2_solution/register_ffa_minima.py \
  --data-root "$DATA" \
  --output-root "$REGISTERED" \
  --minima-root code/external/MINIMA \
  --checkpoint weights/external/minima_loftr.ckpt \
  --splits training validation
```

Review the generated QA images and `registration_manifest.json` before
training. Registration uses no vessel or biomarker labels.

### 3. Train the final Task 1 and Task 2 models

```bash
python code/gave2_solution/train_segmentation.py \
  --task 1 --data-root "$DATA" \
  --output-dir "$RUNS/vascfusion/task1" \
  --init-checkpoint "$RUNS/init/task1/fold0/best.pt" \
  --epochs 48 --size 1024x1536 --batch-size 1 \
  --accumulation-steps 2 --learning-rate 1.5e-4 \
  --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 \
  --ema-decay 0.995 --refinement-steps 5 --hard-gap-weight 0.25 \
  --seed 77 --device cuda --num-workers 0

python code/gave2_solution/train_segmentation.py \
  --task 2 --data-root "$DATA" --ffa-root "$REGISTERED" \
  --output-dir "$RUNS/vascfusion/task2" \
  --init-checkpoint "$RUNS/init/task2/fold0/best.pt" \
  --epochs 37 --size 1024x1536 --batch-size 1 \
  --accumulation-steps 2 --learning-rate 1.5e-4 \
  --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 \
  --ema-decay 0.995 --refinement-steps 5 --hard-gap-weight 0.25 \
  --seed 77 --device cuda --num-workers 0
```

This corresponds to 1,200 optimizer updates for Task 1 and 925 for Task 2.

### 4. Prepare the optic disc and SIVA C zone

```bash
python code/biomarker_tools/local_optic_disc.py \
  --data-root "$DATA" \
  --weights weights/external/Model_DiscSeg_ORIGA.h5 \
  --output-dir "$PWD/cache/disc_masks" --split all --device cuda

python code/gave2_solution/prepare_zone_c_masks.py \
  --disc-dir "$PWD/cache/disc_masks/training" \
  --output-dir "$PWD/cache/zone_c/training" \
  --report-output "$PWD/cache/zone_c/training_report.json"
```

### 5. Train VascFusion-Quant

The quantification system uses several specialized segmentation-derived
measurement branches. All branches start from the Task 2 Fold-0 initialization:

```bash
export INIT_T2="$RUNS/init/task2/fold0/best.pt"
```

**Caliber branch — original FFA geometry:**

```bash
python code/gave2_solution/train_quantification.py \
  --task 2 --data-root "$DATA" --output-dir "$RUNS/quant/caliber" \
  --epochs 47 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

**Artery-density branch — original FFA geometry:**

```bash
python code/gave2_solution/train_quantification.py \
  --task 2 --data-root "$DATA" --output-dir "$RUNS/quant/artery_density" \
  --epochs 47 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

The released factor in `configs/artery_density_calibration.json` is estimated
from training-only OOF predictions and remains fixed during inference.

**Registered-FFA parent for vein density:**

```bash
python code/gave2_solution/train_quantification.py \
  --task 2 --data-root "$DATA" --ffa-root "$REGISTERED" \
  --output-dir "$RUNS/quant/vein_parent" \
  --epochs 48 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

**Phase-geometry branch for fractal dimension:**

```bash
python code/gave2_solution/train_quantification.py \
  --task 2 --phase-geometry --data-root "$DATA" \
  --ffa-root "$REGISTERED" --output-dir "$RUNS/quant/phase_geometry" \
  --epochs 48 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --init-checkpoint "$INIT_T2"
```

**Compact C-zone vein refiner:**

```bash
python code/gave2_solution/train_vein_refiner.py \
  --data-root "$DATA" --ffa-root "$REGISTERED" \
  --zone-c-dir "$PWD/cache/zone_c/training" \
  --parent-checkpoint "$RUNS/quant/vein_parent/final.pt" \
  --output-dir "$RUNS/quant/vein_refiner" \
  --size 1024x1536 --epochs 24 --accumulation-steps 2 \
  --learning-rate 3e-4 --weight-decay 1e-4 --warmup-epochs 2 \
  --ema-decay 0.995 --seed 77 --device cuda --num-workers 0 \
  --loss-policy vessel_argmax
```

Only the compact C-zone refiner is optimized in the last stage; its parent
network remains frozen.

## Evaluation and usage constraints

For labeled local predictions, reproduce the released A/V metrics with:

```bash
python code/gave2_solution/evaluate_av.py \
  --predictions-dir /path/to/probability_pngs \
  --labels-dir "$DATA/training/av" \
  --masks-dir "$DATA/training/masks" \
  --output /path/to/metrics.json
```

For Task 3, `local_biomarker_eval.py` reports raw MAE and SMAPE on labeled
training folds. The organizer's hidden P/Q normalization is not guessed by the
local evaluator.

- Task 1 must never access FFA.
- Task 1/2 PNG channels are `[artery, all vessel, vein]`.
- Task 3 always uses the same field route for every case.
- No test labels or per-case leaderboard-driven selection are used.
- Do not refit calibration factors on validation/private-test images.

## Third-party code

The repository includes an unmodified source snapshot of
[MINIMA](https://github.com/LSXI7/MINIMA) for reproducible FFA registration;
its Apache-2.0 license is retained. The optic-disc weight originates from
MNet/DeepCDR and is not redistributed here. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

Our challenge report citation will be added after publication. If this code is
useful, please also cite the GAVE2 challenge and the upstream methods listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Our original code is released under the [MIT License](LICENSE). Third-party
components remain subject to their respective licenses, as documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
