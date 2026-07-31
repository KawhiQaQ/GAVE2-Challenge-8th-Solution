#!/usr/bin/env python3
"""PyTorch inference port of the MNet_DeepCDR optic-disc detector.

The GAVE2 baseline points to a TensorFlow 1.x model that is no longer directly
installable on current Apple Silicon. This module reproduces its U-Net graph and
loads the released Keras HDF5 weights without TensorFlow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import label
import torch
from torch import Tensor, nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "GAVE2_preliminary"
DEFAULT_WEIGHTS = (
    REPO_ROOT.parent
    / "external"
    / "MNet_DeepCDR"
    / "mnet_deep_cdr"
    / "deep_model"
    / "Model_DiscSeg_ORIGA.h5"
)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        return F.relu(self.conv2(F.relu(self.conv1(image))))


class DiscUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = DoubleConv(3, 32)
        self.block2 = DoubleConv(32, 64)
        self.block3 = DoubleConv(64, 128)
        self.block4 = DoubleConv(128, 256)
        self.block5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.block6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.block7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.block8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.block9 = DoubleConv(64, 32)
        self.side6 = nn.Conv2d(256, 1, 1)
        self.side7 = nn.Conv2d(128, 1, 1)
        self.side8 = nn.Conv2d(64, 1, 1)
        self.side9 = nn.Conv2d(32, 1, 1)

    def forward(self, image: Tensor) -> Tensor:
        conv1 = self.block1(image)
        conv2 = self.block2(F.max_pool2d(conv1, 2))
        conv3 = self.block3(F.max_pool2d(conv2, 2))
        conv4 = self.block4(F.max_pool2d(conv3, 2))
        conv5 = self.block5(F.max_pool2d(conv4, 2))
        conv6 = self.block6(torch.cat((self.up6(conv5), conv4), dim=1))
        conv7 = self.block7(torch.cat((self.up7(conv6), conv3), dim=1))
        conv8 = self.block8(torch.cat((self.up8(conv7), conv2), dim=1))
        conv9 = self.block9(torch.cat((self.up9(conv8), conv1), dim=1))
        out6 = torch.sigmoid(self.side6(F.interpolate(conv6, scale_factor=8, mode="nearest")))
        out7 = torch.sigmoid(self.side7(F.interpolate(conv7, scale_factor=4, mode="nearest")))
        out8 = torch.sigmoid(self.side8(F.interpolate(conv8, scale_factor=2, mode="nearest")))
        out9 = torch.sigmoid(self.side9(conv9))
        return (out6 + out7 + out8 + out9) / 4.0


def keras_kernel(dataset: h5py.Dataset, transpose: bool = False) -> Tensor:
    array = np.asarray(dataset)
    if transpose:
        array = array.transpose(3, 2, 0, 1)
    else:
        array = array.transpose(3, 2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(array))


def assign_conv(
    handle: h5py.File,
    module: nn.Conv2d | nn.ConvTranspose2d,
    layer_name: str,
) -> None:
    group = handle[layer_name][layer_name]
    kernel = keras_kernel(group["kernel:0"], transpose=isinstance(module, nn.ConvTranspose2d))
    bias = torch.from_numpy(np.asarray(group["bias:0"]))
    if module.weight.shape != kernel.shape:
        raise ValueError(f"{layer_name}: expected {tuple(module.weight.shape)}, got {tuple(kernel.shape)}")
    module.weight.data.copy_(kernel)
    module.bias.data.copy_(bias)


def load_keras_weights(model: DiscUNet, path: Path) -> None:
    with h5py.File(path, "r") as handle:
        blocks = [
            (model.block1, "block1"),
            (model.block2, "block2"),
            (model.block3, "block3"),
            (model.block4, "block4"),
            (model.block5, "block5"),
            (model.block6, "block6"),
            (model.block7, "block7"),
            (model.block8, "block8"),
            (model.block9, "block9"),
        ]
        for block, prefix in blocks:
            assign_conv(handle, block.conv1, f"{prefix}_conv1")
            assign_conv(handle, block.conv2, f"{prefix}_conv2")
        for module, name in [
            (model.up6, "block6_dconv"),
            (model.up7, "block7_dconv"),
            (model.up8, "block8_dconv"),
            (model.up9, "block9_dconv"),
            (model.side6, "side_6"),
            (model.side7, "side_7"),
            (model.side8, "side_8"),
            (model.side9, "side_9"),
        ]:
            assign_conv(handle, module, name)


def largest_filled_component(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    binary = probability > threshold
    if not binary.any() and probability.max() > 0:
        binary = probability > probability.max() / 2.0
    components = label(binary, connectivity=2)
    if components.max() == 0:
        return np.zeros_like(binary)
    areas = np.bincount(components.ravel())
    areas[0] = 0
    binary = components == int(areas.argmax())
    return ndimage.binary_fill_holes(binary)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("training", "validation", "all"), default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model = DiscUNet()
    load_keras_weights(model, Path(args.weights))
    model.to(device).eval()
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    splits = ("training", "validation") if args.split == "all" else (args.split,)

    for split in splits:
        output_dir = output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted((data_root / split / "images").glob("g_*.png"))
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                original_size = image.size
                resized = image.resize((640, 640), Image.Resampling.BILINEAR)
                array = np.asarray(resized, dtype=np.float32)
            tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
            probability = model(tensor)[0, 0].cpu().numpy()
            binary = largest_filled_component(probability)
            mask = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
            mask = mask.resize(original_size, Image.Resampling.NEAREST)
            mask.save(output_dir / image_path.name)
            print(f"{split}/{image_path.name} max={probability.max():.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
