# Checkpoint placement

Model weights are intentionally excluded from Git. A Baidu Netdisk download
link will be added after the competition. Extract the future bundle at the
repository root so that the following files exist:

```text
weights/
├── initialization/
│   ├── task1_fold0.pt
│   └── task2_fold0.pt
├── task1/segmentation_cfp.pt
├── task2/segmentation_multimodal.pt
├── task3/
│   ├── caliber_expert.pt
│   ├── artery_density_expert.pt
│   ├── vein_density_parent.pt
│   ├── phase_geometry_expert.pt
│   └── vein_density_refiner.pt
└── external/
    ├── minima_loftr.ckpt
    └── Model_DiscSeg_ORIGA.h5
```

Only the seven final inference checkpoints and two external weights are needed
to run `code/run_inference.sh`. The two initialization checkpoints are
needed only to reproduce full training.

Verify every file against `configs/weights_manifest.json` before inference.
For example:

```bash
shasum -a 256 weights/task1/segmentation_cfp.pt
```

The MINIMA-LoFTR checkpoint is also available from the
[official MINIMA release](https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt).
