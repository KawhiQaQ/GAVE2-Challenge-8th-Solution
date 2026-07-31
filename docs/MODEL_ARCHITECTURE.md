# Model architecture

## System overview

TJ009 consists of two final segmentation networks and one deterministic
biomarker system:

1. **Task 1:** GAVEV54, CFP only;
2. **Task 2:** GAVEV54, CFP plus registered early/late FFA;
3. **Task 3:** a fixed field route over V2, V3, V12, and V17 probability maps.

All segmentation images follow the challenge channel convention:
`R = artery`, `G = all vessels`, and `B = vein`.

## GAVEV54 for Task 1 and Task 2

The shared backbone contains:

- a ConvNeXt-Tiny CFP encoder;
- a multi-scale U-Net-like decoder;
- a native-resolution detail stem;
- semantic, generic-vessel, and class-centerline prediction heads;
- a shared deep probability U-Net that recursively refines artery/vein
  classification for five steps while conditioning on the frozen generic
  vessel probability.

New residual outputs are zero-initialized, preserving the parent model at the
start of warm-start training.

### Task 1 specialization

- only CFP is accepted by the data loader and prediction entry point;
- the detail stem receives normalized RGB and local red/green optical-density
  features;
- a native-resolution A/V refiner targets thin vessels and branch continuity.

### Task 2 specialization

- CFP is encoded by ConvNeXt-Tiny;
- early FFA, late FFA, and their positive difference form the three-channel
  angiography input;
- a lightweight FFA pyramid is injected at every encoder scale through gated
  cross-modal fusion;
- early and late FFA are independently registered to CFP with MINIMA-LoFTR.

### Final training protocol

| Parameter | Task 1 | Task 2 |
|---|---:|---:|
| Input resolution | 1024×1536 | 1024×1536 |
| Epochs | 48 | 37 |
| Optimizer updates | 1,200 | 925 |
| Batch / accumulation | 1 / 2 | 1 / 2 |
| Learning rate | 1.5e-4 | 1.5e-4 |
| Encoder LR ratio | 0.10 | 0.10 |
| Weight decay | 1e-4 | 1e-4 |
| Warm-up | 2 epochs | 2 epochs |
| Encoder freeze | 1 epoch | 1 epoch |
| EMA | 0.995 | 0.995 |
| Seed | 77 | 77 |

Both models are warm-started from the matching V1 Fold-0 checkpoint and then
trained on all 50 labeled preliminary cases. No public R2-V2 competition
checkpoint is loaded.

## Task 3 geometry system

Task 3 does not use one end-to-end regression head. It extracts clinically
defined measurements from predicted vessel geometry and uses one route fixed
before test inference.

| Field | Model and processing |
|---|---|
| CRAE / CRVE / AVR | V2 Task 2 full model, raw FFA, vessel-argmax masks and revised caliber formula |
| artery density | V3 Task 2 full model, raw FFA, global OOF scale calibration |
| vein density | frozen V8 parent plus D0044 C-zone vein-logit residual refiner, registered FFA |
| artery fractal dimension | deterministic branch-consistency postprocess over V2 maps, skeletonization and box counting |
| vein fractal dimension | arithmetic mean of V12 scalar FD estimates at thresholds 0.4, 0.5 and 0.6 |

V12 adds a full ConvNeXt-Tiny FFA phase encoder to the V2 CFP path and fuses
registered phase geometry at four scales. D0044 freezes V8 and trains only a
small U-Net-style residual head; it can change the vein probability only in the
SIVA C zone.

The route is global and case-independent. It does not access ground truth,
scan test thresholds, or switch models based on individual cases.

## Postprocessing and packaging

- Task 1/2 use radius-3 structural repair;
- probability maps use step-4 uint8 quantization that preserves every
  `>= 128` decision;
- Task 3 files contain seven keys in a fixed order;
- the final archive contains `Task1/`, `Task2/`, and `Task3/` directly.

