from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from sklearn.model_selection import StratifiedKFold
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def case_ids(data_root: Path, split: str) -> list[str]:
    return sorted(path.stem for path in (data_root / split / "images").glob("g_*.png"))


def parse_biomarker(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()[:2]
        values[key] = float(value)
    return values


def make_balanced_folds(
    data_root: Path,
    n_folds: int = 5,
    seed: int = 77,
) -> list[dict[str, list[str]]]:
    """Balance folds by annotated vessel density/network complexity."""
    ids = case_ids(data_root, "training")
    complexity = []
    for case_id in ids:
        values = parse_biomarker(data_root / "training" / "biomarker" / f"{case_id}.txt")
        complexity.append(
            values["artery_density"]
            + values["vein_density"]
            + 0.02
            * (
                values["artery_fractal_dimension"]
                + values["vein_fractal_dimension"]
            )
        )
    order = np.argsort(np.asarray(complexity))
    strata = np.empty(len(ids), dtype=np.int64)
    for rank, index in enumerate(order):
        strata[index] = min(n_folds - 1, rank * n_folds // len(ids))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    indices = np.arange(len(ids))
    for train_indices, validation_indices in splitter.split(indices, strata):
        folds.append(
            {
                "training": sorted(ids[index] for index in train_indices),
                "validation": sorted(ids[index] for index in validation_indices),
            }
        )
    return folds


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _label_to_classes(label: np.ndarray) -> np.ndarray:
    classes = np.zeros(label.shape[:2], dtype=np.uint8)
    classes[label[..., 0] > 127] = 1
    classes[label[..., 2] > 127] = 2
    classes[label[..., 1] > 127] = 3
    return classes


def class_to_av(classes: Tensor) -> Tensor:
    artery = (classes == 1) | (classes == 3)
    vessel = classes != 0
    vein = (classes == 2) | (classes == 3)
    return torch.stack((artery, vessel, vein), dim=0).float()


def _centerlines(classes: np.ndarray) -> np.ndarray:
    artery = (classes == 1) | (classes == 3)
    vessel = classes != 0
    vein = (classes == 2) | (classes == 3)
    return np.stack(
        [skeletonize(mask).astype(np.uint8) for mask in (artery, vessel, vein)],
        axis=0,
    )


def _robust_ffa(channel: np.ndarray, roi: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    values = channel[(roi > 0) & (channel > 0)]
    if values.size < 100:
        return channel / 255.0
    low, high = np.percentile(values, (1.0, 99.5))
    if high <= low:
        return channel / 255.0
    return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)


class GAVE2Dataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        ids: Sequence[str],
        task: int,
        split: str = "training",
        size: tuple[int, int] = (768, 1152),
        augment: bool = False,
        cache: bool = True,
        ffa_jitter: int = 8,
        ffa_root: str | Path | None = None,
        region_dir: str | Path | None = None,
    ) -> None:
        if task not in (1, 2):
            raise ValueError("task must be 1 or 2")
        self.data_root = Path(data_root)
        self.ids = list(ids)
        self.task = task
        self.split = split
        self.size = size
        self.augment = augment
        self.cache = cache
        self.ffa_jitter = ffa_jitter
        self.ffa_root = Path(ffa_root) if ffa_root is not None else None
        self.region_dir = Path(region_dir) if region_dir is not None else None
        self.samples: dict[str, dict[str, np.ndarray]] = {}
        if cache:
            for case_id in self.ids:
                self.samples[case_id] = self._read_case(case_id)

    def _read_case(self, case_id: str) -> dict[str, np.ndarray]:
        base = self.data_root / self.split
        sample: dict[str, np.ndarray] = {
            "rgb": _read_rgb(base / "images" / f"{case_id}.png"),
            "roi": _read_gray(base / "masks" / f"{case_id}.png"),
        }
        if self.region_dir is not None:
            sample["region"] = _read_gray(self.region_dir / f"{case_id}.png")
        if self.task == 2:
            ffa_base = (
                self.ffa_root / self.split
                if self.ffa_root is not None
                else base
            )
            early = _read_gray(ffa_base / "FFA_A" / f"{case_id}.png")
            late = _read_gray(ffa_base / "FFA_AV" / f"{case_id}.png")
            early = _robust_ffa(early, sample["roi"])
            late = _robust_ffa(late, sample["roi"])
            delta = np.clip(late - early, 0.0, 1.0)
            sample["ffa"] = np.stack((early, late, delta), axis=0).astype(np.float32)
        if self.split == "training":
            label = _read_rgb(base / "av" / f"{case_id}.png")
            classes = _label_to_classes(label)
            sample["classes"] = classes
            sample["centerlines"] = _centerlines(classes)
        return sample

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _affine(
        tensor: Tensor,
        angle: float,
        translate: list[int],
        scale: float,
        shear: list[float],
        interpolation: InterpolationMode,
    ) -> Tensor:
        return TF.affine(
            tensor,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=0,
            center=None,
        )

    def _shared_geometry(
        self,
        rgb: Tensor,
        ffa: Tensor | None,
        classes: Tensor | None,
        roi: Tensor,
        centerlines: Tensor | None,
        region: Tensor | None,
    ) -> tuple[
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor,
        Tensor | None,
        Tensor | None,
    ]:
        rgb = TF.resize(
            rgb,
            self.size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        roi = TF.resize(roi, self.size, interpolation=InterpolationMode.NEAREST)
        if region is not None:
            region = TF.resize(
                region,
                self.size,
                interpolation=InterpolationMode.NEAREST,
            )
        if ffa is not None:
            ffa = TF.resize(
                ffa,
                self.size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        if classes is not None:
            classes = TF.resize(
                classes[None].float(),
                self.size,
                interpolation=InterpolationMode.NEAREST,
            )[0].long()
        if centerlines is not None:
            centerlines = TF.resize(
                centerlines.float(),
                self.size,
                interpolation=InterpolationMode.NEAREST,
            )

        if not self.augment:
            return rgb, ffa, classes, roi, centerlines, region

        height, width = self.size
        angle = random.uniform(-30.0, 30.0)
        translate = [
            round(random.uniform(-0.04, 0.04) * width),
            round(random.uniform(-0.04, 0.04) * height),
        ]
        scale = random.uniform(0.9, 1.12)
        shear = [random.uniform(-8.0, 8.0), random.uniform(-5.0, 5.0)]
        rgb = self._affine(
            rgb, angle, translate, scale, shear, InterpolationMode.BILINEAR
        )
        roi = self._affine(
            roi, angle, translate, scale, shear, InterpolationMode.NEAREST
        )
        if region is not None:
            region = self._affine(
                region,
                angle,
                translate,
                scale,
                shear,
                InterpolationMode.NEAREST,
            )
        if ffa is not None:
            ffa = self._affine(
                ffa, angle, translate, scale, shear, InterpolationMode.BILINEAR
            )
        if classes is not None:
            classes = self._affine(
                classes[None].float(),
                angle,
                translate,
                scale,
                shear,
                InterpolationMode.NEAREST,
            )[0].long()
        if centerlines is not None:
            centerlines = self._affine(
                centerlines,
                angle,
                translate,
                scale,
                shear,
                InterpolationMode.NEAREST,
            )

        if random.random() < 0.5:
            rgb = TF.hflip(rgb)
            roi = TF.hflip(roi)
            if region is not None:
                region = TF.hflip(region)
            if ffa is not None:
                ffa = TF.hflip(ffa)
            if classes is not None:
                classes = TF.hflip(classes)
            if centerlines is not None:
                centerlines = TF.hflip(centerlines)
        if random.random() < 0.5:
            rgb = TF.vflip(rgb)
            roi = TF.vflip(roi)
            if region is not None:
                region = TF.vflip(region)
            if ffa is not None:
                ffa = TF.vflip(ffa)
            if classes is not None:
                classes = TF.vflip(classes)
            if centerlines is not None:
                centerlines = TF.vflip(centerlines)
        return rgb, ffa, classes, roi, centerlines, region

    @staticmethod
    def _rgb_photometric(rgb: Tensor) -> Tensor:
        if random.random() < 0.8:
            rgb = TF.adjust_brightness(rgb, random.uniform(0.75, 1.25))
            rgb = TF.adjust_contrast(rgb, random.uniform(0.75, 1.3))
            rgb = TF.adjust_saturation(rgb, random.uniform(0.75, 1.3))
            rgb = TF.adjust_hue(rgb, random.uniform(-0.035, 0.035))
        if random.random() < 0.4:
            rgb = TF.adjust_gamma(rgb, random.uniform(0.75, 1.35))
        if random.random() < 0.3:
            rgb = rgb + torch.randn_like(rgb) * random.uniform(0.005, 0.025)
        return rgb.clamp_(0.0, 1.0)

    def _ffa_photometric(self, ffa: Tensor) -> Tensor:
        if random.random() < 0.8:
            gain = torch.empty(3, 1, 1).uniform_(0.8, 1.25)
            gamma = random.uniform(0.75, 1.35)
            ffa = (ffa.clamp_min(0.0).pow(gamma) * gain).clamp_(0.0, 1.0)
        if random.random() < 0.5 and self.ffa_jitter > 0:
            dx = random.randint(-self.ffa_jitter, self.ffa_jitter)
            dy = random.randint(-self.ffa_jitter, self.ffa_jitter)
            shifted = torch.roll(ffa, shifts=(dy, dx), dims=(-2, -1))
            if dy > 0:
                shifted[:, :dy] = 0
            elif dy < 0:
                shifted[:, dy:] = 0
            if dx > 0:
                shifted[:, :, :dx] = 0
            elif dx < 0:
                shifted[:, :, dx:] = 0
            ffa = shifted
        # Modality dropout makes the fusion robust to poor angiography phases.
        if random.random() < 0.08:
            ffa[random.randrange(3)] = 0
        return ffa

    @staticmethod
    def _random_occlusion(rgb: Tensor, ffa: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if random.random() >= 0.25:
            return rgb, ffa
        height, width = rgb.shape[-2:]
        box_h = random.randint(max(4, height // 40), max(5, height // 12))
        box_w = random.randint(max(4, width // 40), max(5, width // 12))
        top = random.randint(0, height - box_h)
        left = random.randint(0, width - box_w)
        rgb[:, top : top + box_h, left : left + box_w] = random.uniform(0.1, 0.9)
        if ffa is not None and random.random() < 0.5:
            ffa[:, top : top + box_h, left : left + box_w] = 0
        return rgb, ffa

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_id = self.ids[index]
        arrays = self.samples[case_id] if self.cache else self._read_case(case_id)
        rgb = torch.from_numpy(
            np.ascontiguousarray(arrays["rgb"].transpose(2, 0, 1))
        ).float() / 255.0
        roi = torch.from_numpy((arrays["roi"] > 127).astype(np.float32))[None]
        ffa = torch.from_numpy(arrays["ffa"].copy()) if self.task == 2 else None
        classes = (
            torch.from_numpy(arrays["classes"].astype(np.int64))
            if "classes" in arrays
            else None
        )
        centerlines = (
            torch.from_numpy(arrays["centerlines"].astype(np.float32))
            if "centerlines" in arrays
            else None
        )
        region = (
            torch.from_numpy((arrays["region"] > 127).astype(np.float32))[None]
            if "region" in arrays
            else None
        )

        rgb, ffa, classes, roi, centerlines, region = self._shared_geometry(
            rgb,
            ffa,
            classes,
            roi,
            centerlines,
            region,
        )
        if self.augment:
            rgb = self._rgb_photometric(rgb)
            if ffa is not None:
                ffa = self._ffa_photometric(ffa)
            rgb, ffa = self._random_occlusion(rgb, ffa)
        rgb = TF.normalize(rgb, IMAGENET_MEAN, IMAGENET_STD)

        result: dict[str, Any] = {"id": case_id, "rgb": rgb, "roi": roi}
        if region is not None:
            result["region"] = region
        if ffa is not None:
            result["ffa"] = ffa
        if classes is not None and centerlines is not None:
            result["classes"] = classes
            result["av"] = class_to_av(classes)
            result["centerlines"] = centerlines
        return result
