# Checkpoint placement

Model weights are intentionally excluded from Git. A Baidu Netdisk download
link will be added after the competition. Extract the future bundle at the
repository root so that the following files exist:

```text
weights/
├── initialization/
│   ├── v1_task1_fold0/best.pt
│   └── v1_task2_fold0/best.pt
├── task1/tj009_v54_full/final.pt
├── task2/tj009_v54_full/final.pt
├── task3/
│   ├── v2_task2_full/final.pt
│   ├── v3_task2_full/final.pt
│   ├── v8_task2_full/final.pt
│   ├── v21_v12_fd_full/final.pt
│   └── d0044_vein_density_full/final.pt
└── external/
    ├── minima_loftr.ckpt
    └── Model_DiscSeg_ORIGA.h5
```

Only the seven final inference checkpoints and two external weights are needed
to run `code/run_tj009_inference.sh`. The two initialization checkpoints are
needed only to reproduce full training.

Verify every file against `configs/weights_manifest.json` before inference.
For example:

```bash
shasum -a 256 weights/task1/tj009_v54_full/final.pt
```

The MINIMA-LoFTR checkpoint is also available from the
[official MINIMA release](https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt).

