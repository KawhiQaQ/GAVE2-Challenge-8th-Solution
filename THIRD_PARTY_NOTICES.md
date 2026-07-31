# Third-party notices

## MINIMA

- Project: [LSXI7/MINIMA](https://github.com/LSXI7/MINIMA)
- Use in this repository: FFA-to-CFP registration with MINIMA-LoFTR
- License: Apache License 2.0
- Included material: source snapshot under `code/external/MINIMA/`; the upstream
  license is retained at `code/external/MINIMA/LICENSE`
- Checkpoint: not included; download from the upstream release or the future
  VascFusion checkpoint bundle

Please cite the MINIMA paper when using the registration component:

> Jiangwei Ren et al., “MINIMA: Modality Invariant Image Matching,” CVPR 2025.

## MNet / DeepCDR

- Project: “Joint Optic Disc and Cup Segmentation Based on Multi-label Deep
  Network and Polar Transformation”
- Use in this repository: optic-disc localization for SIVA-zone construction
- Upstream code license: CC BY-NC-SA 4.0 for non-commercial use
- Included material: a PyTorch inference-compatible architecture/weight loader
  in `code/biomarker_tools/local_optic_disc.py`
- Upstream HDF5 checkpoint: not redistributed in this repository

Relevant citation:

> Huazhu Fu, Jun Cheng, Yanwu Xu, Damon Wing Kee Wong, Jiang Liu, and Xiaochun
> Cao, “Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network
> and Polar Transformation,” IEEE Transactions on Medical Imaging, 2018.

## ConvNeXt and PyTorch

The CFP encoders are built through PyTorch/torchvision ConvNeXt components.
Their source is not vendored here. Please follow the respective upstream
licenses and cite ConvNeXt where appropriate.

Third-party components remain governed by their original terms, independently
of the license eventually selected for our original code.
