# Checkpoint placement

Model weights are excluded from Git and released through
[Baidu Netdisk](https://pan.baidu.com/s/1tqCe97lvXGM0zn1VvUlPMA?pwd=29v6)
(extraction code: `29v6`). Download the shared `weights/` directory into the
repository root, then create the semantic aliases documented in the root
README so that the following files exist:

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
