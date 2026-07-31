#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from skimage.morphology import skeletonize
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from gave2v1.data import GAVE2Dataset, make_balanced_folds
from gave2v1.engine import WarmupCosine, move_batch, seed_everything, select_device
from gave2v1.model_v17 import GAVEV17Task3, load_v8_parent


@dataclass
class V17Config:
    fold: int
    n_folds: int
    seed: int
    height: int
    width: int
    epochs: int
    patience: int
    accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    ema_decay: float
    data_root: str
    ffa_root: str
    zone_c_dir: str
    parent_checkpoint: str
    output_dir: str
    device: str
    loss_policy: str = "independent"
    selection_metric: str = "geometry_proxy"


def parse_size(value: str) -> tuple[int, int]:
    height, width = (int(part) for part in value.lower().split("x"))
    if height % 32 or width % 32:
        raise argparse.ArgumentTypeError(
            "height and width must be divisible by 32"
        )
    return height, width


def masked_vein_loss(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
) -> dict[str, Tensor]:
    valid_sum = valid.sum().clamp_min(1.0)
    bce = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        * valid
    ).sum() / valid_sum
    probability = torch.sigmoid(logits)
    intersection = (probability * target * valid).sum()
    dice = (
        2.0 * intersection + 1.0
    ) / (
        (probability * valid).sum()
        + (target * valid).sum()
        + 1.0
    )
    dice_loss = 1.0 - dice
    return {
        "total": 0.5 * bce + 0.5 * dice_loss,
        "bce": bce,
        "dice": dice_loss,
    }


def policy_aligned_vein_loss(
    outputs: dict[str, Tensor],
    target: Tensor,
    valid: Tensor,
) -> dict[str, Tensor]:
    """Train the smooth equivalent of the submitted vessel+argmax vein mask."""

    parent = outputs["parent_probabilities"]
    artery = parent[:, 0:1].clamp(1e-4, 1.0 - 1e-4)
    vessel = parent[:, 1:2].clamp(0.0, 1.0)
    artery_logits = torch.logit(artery)
    vein_fraction = torch.sigmoid(
        outputs["vein_logits"] - artery_logits
    )
    probability = (vessel * vein_fraction).clamp(1e-4, 1.0 - 1e-4)
    probability_logits = torch.logit(probability)
    valid_sum = valid.sum().clamp_min(1.0)
    bce = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            probability_logits,
            target,
            reduction="none",
        )
        * valid
    ).sum() / valid_sum
    intersection = (probability * target * valid).sum()
    dice = (
        2.0 * intersection + 1.0
    ) / (
        (probability * valid).sum()
        + (target * valid).sum()
        + 1.0
    )
    dice_loss = 1.0 - dice
    return {
        "total": 0.5 * bce + 0.5 * dice_loss,
        "bce": bce,
        "dice": dice_loss,
    }


def _cldice(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> float:
    prediction = prediction & valid
    target = target & valid
    prediction_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)
    topology_precision = (
        float((prediction_skeleton & target).sum())
        / max(1, int(prediction_skeleton.sum()))
    )
    topology_sensitivity = (
        float((target_skeleton & prediction).sum())
        / max(1, int(target_skeleton.sum()))
    )
    return (
        2.0
        * topology_precision
        * topology_sensitivity
        / max(1e-8, topology_precision + topology_sensitivity)
    )


@torch.inference_mode()
def evaluate(
    model: GAVEV17Task3,
    refiner: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    apply_refiner: bool,
    loss_policy: str = "independent",
) -> dict[str, float]:
    model.eval()
    refiner.eval()
    intersection = 0
    prediction_sum = 0
    target_sum = 0
    cldice_values: list[float] = []
    density_scores: list[float] = []
    losses: list[float] = []
    for batch in loader:
        batch_device = move_batch(batch, device)
        outputs = model(
            batch_device["rgb"],
            batch_device["ffa"],
            batch_device["region"],
            refiner=refiner,
            apply_refiner=apply_refiner,
        )
        target = batch_device["av"][:, 2:3]
        valid = batch_device["region"] * batch_device["roi"]
        if loss_policy == "vessel_argmax":
            loss = policy_aligned_vein_loss(
                outputs,
                target,
                valid,
            )
        else:
            loss = masked_vein_loss(
                outputs["vein_logits"],
                target,
                valid,
            )
        losses.append(float(loss["total"].cpu()))
        if loss_policy == "vessel_argmax":
            parent = outputs["parent_probabilities"]
            predicted = (
                (parent[:, 1:2] > 0.5)
                & (outputs["vein_probability"] > parent[:, 0:1])
            )
        else:
            predicted = outputs["vein_probability"] > 0.5
        truth = target > 0.5
        intersection += int((predicted & truth & (valid > 0.5)).sum().cpu())
        prediction_sum += int((predicted & (valid > 0.5)).sum().cpu())
        target_sum += int((truth & (valid > 0.5)).sum().cpu())
        for index in range(predicted.shape[0]):
            pred_np = predicted[index, 0].cpu().numpy()
            truth_np = truth[index, 0].cpu().numpy()
            valid_np = valid[index, 0].cpu().numpy() > 0.5
            cldice_values.append(_cldice(pred_np, truth_np, valid_np))
            pred_density = float((pred_np & valid_np).sum()) / max(
                1,
                int(valid_np.sum()),
            )
            truth_density = float((truth_np & valid_np).sum()) / max(
                1,
                int(valid_np.sum()),
            )
            density_scores.append(
                2.0
                * min(pred_density, truth_density)
                / max(1e-8, pred_density + truth_density)
            )
    dice = 2.0 * intersection / max(1, prediction_sum + target_sum)
    cldice = float(np.mean(cldice_values))
    density_score = float(np.mean(density_scores))
    proxy = 0.4 * dice + 0.3 * cldice + 0.3 * density_score
    return {
        "loss": float(np.mean(losses)),
        "dice": dice,
        "cldice": cldice,
        "density_score": density_score,
        "vein_geometry_proxy": proxy,
    }


@torch.no_grad()
def update_ema(
    ema: nn.Module,
    source: nn.Module,
    decay: float,
) -> None:
    source_state = source.state_dict()
    for name, value in ema.state_dict().items():
        source_value = source_state[name].detach()
        if value.dtype.is_floating_point:
            value.mul_(decay).add_(source_value, alpha=1.0 - decay)
        else:
            value.copy_(source_value)


def save_checkpoint(
    path: Path,
    refiner: nn.Module,
    config: V17Config,
    split: dict[str, list[str]],
    parent_report: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "refiner": refiner.state_dict(),
            "config": asdict(config),
            "split": split,
            "parent": parent_report,
            "epoch": epoch,
            "metrics": metrics,
            "architecture": GAVEV17Task3.architecture_name,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train V17 Task3 C-zone vein geometry refiner."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ffa-root", required=True)
    parser.add_argument("--zone-c-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--size", type=parse_size, default=(1024, 1536))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--loss-policy",
        choices=("independent", "vessel_argmax"),
        default="independent",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("geometry_proxy", "density_score"),
        default="geometry_proxy",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    ffa_root = Path(args.ffa_root).resolve()
    zone_c_dir = Path(args.zone_c_dir).resolve()
    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    split = make_balanced_folds(
        data_root,
        args.n_folds,
        args.seed,
    )[args.fold]
    expected_regions = {
        f"{case_id}.png"
        for case_id in split["training"] + split["validation"]
    }
    actual_regions = {path.name for path in zone_c_dir.glob("g_*.png")}
    missing_regions = sorted(expected_regions - actual_regions)
    if missing_regions:
        raise FileNotFoundError(
            f"Missing {len(missing_regions)} C-zone masks: "
            f"{missing_regions[:5]}"
        )

    seed_everything(args.seed + args.fold)
    device = select_device(args.device)
    parent, parent_report = load_v8_parent(parent_checkpoint)
    model = GAVEV17Task3(parent).to(device)
    height, width = args.size
    train_dataset = GAVE2Dataset(
        data_root,
        split["training"],
        task=2,
        size=(height, width),
        augment=True,
        cache=True,
        ffa_jitter=4,
        ffa_root=ffa_root,
        region_dir=zone_c_dir,
    )
    validation_dataset = GAVE2Dataset(
        data_root,
        split["validation"],
        task=2,
        size=(height, width),
        augment=False,
        cache=True,
        ffa_root=ffa_root,
        region_dir=zone_c_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    config = V17Config(
        fold=args.fold,
        n_folds=args.n_folds,
        seed=args.seed,
        height=height,
        width=width,
        epochs=args.epochs,
        patience=args.patience,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        ema_decay=args.ema_decay,
        data_root=str(data_root),
        ffa_root=str(ffa_root),
        zone_c_dir=str(zone_c_dir),
        parent_checkpoint=str(parent_checkpoint),
        output_dir=str(output_dir),
        device=str(device),
        loss_policy=args.loss_policy,
        selection_metric=args.selection_metric,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(split, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "parent.json").write_text(
        json.dumps(parent_report, indent=2) + "\n",
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(
        model.refiner.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / max(1, args.accumulation_steps)
    )
    scheduler = WarmupCosine(
        optimizer,
        total_steps=args.epochs * updates_per_epoch,
        warmup_steps=args.warmup_epochs * updates_per_epoch,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = copy.deepcopy(model.refiner).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

    parent_baseline = evaluate(
        model,
        ema,
        validation_loader,
        device,
        apply_refiner=False,
        loss_policy=args.loss_policy,
    )
    print(
        json.dumps(
            {
                "architecture": model.architecture_name,
                "device": str(device),
                "parent": parent_report,
                "train_cases": len(split["training"]),
                "validation_cases": len(split["validation"]),
                "refiner_parameters": sum(
                    parameter.numel()
                    for parameter in model.refiner.parameters()
                ),
                "parent_trainable_parameters": sum(
                    parameter.requires_grad
                    for parameter in model.parent.parameters()
                ),
                "parent_baseline": parent_baseline,
            },
            indent=2,
        ),
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    bad_epochs = 0
    use_amp = device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        totals = {"total": 0.0, "bce": 0.0, "dice": 0.0}
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch_device = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                outputs = model(
                    batch_device["rgb"],
                    batch_device["ffa"],
                    batch_device["region"],
                )
                target = batch_device["av"][:, 2:3]
                valid = batch_device["region"] * batch_device["roi"]
                if args.loss_policy == "vessel_argmax":
                    losses = policy_aligned_vein_loss(
                        outputs,
                        target,
                        valid,
                    )
                else:
                    losses = masked_vein_loss(
                        outputs["vein_logits"],
                        target,
                        valid,
                    )
                backward_loss = (
                    losses["total"] / args.accumulation_steps
                )
            scaler.scale(backward_loss).backward()
            for name in totals:
                totals[name] += float(losses[name].detach().cpu())
            should_update = (
                step % args.accumulation_steps == 0
                or step == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.refiner.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(ema, model.refiner, args.ema_decay)

        validation = evaluate(
            model,
            ema,
            validation_loader,
            device,
            apply_refiner=True,
            loss_policy=args.loss_policy,
        )
        training = {
            name: value / max(1, len(train_loader))
            for name, value in totals.items()
        }
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - start,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training": training,
            "validation": validation,
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        save_checkpoint(
            output_dir / "last.pt",
            ema,
            config,
            split,
            parent_report,
            epoch,
            validation,
        )
        score = validation[
            "density_score"
            if args.selection_metric == "density_score"
            else "vein_geometry_proxy"
        ]
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "seconds": record["seconds"],
                    "train_total": training["total"],
                    "val_loss": validation["loss"],
                    "vein_dice": validation["dice"],
                    "vein_cldice": validation["cldice"],
                    "density_score": validation["density_score"],
                    "score_proxy": score,
                    "parent_proxy": parent_baseline[
                        "density_score"
                        if args.selection_metric == "density_score"
                        else "vein_geometry_proxy"
                    ],
                }
            ),
            flush=True,
        )
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                output_dir / "best.pt",
                ema,
                config,
                split,
                parent_report,
                epoch,
                validation,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"early_stop epoch={epoch} best_epoch={best_epoch} "
                    f"best_score={best_score:.6f}",
                    flush=True,
                )
                break
    summary = {
        "best_epoch": best_epoch,
        "best_score_proxy": best_score,
        "selection_metric": args.selection_metric,
        "loss_policy": args.loss_policy,
        "parent_score_proxy": parent_baseline[
            "density_score"
            if args.selection_metric == "density_score"
            else "vein_geometry_proxy"
        ],
        "epochs_completed": len(history),
        "best_checkpoint": str(output_dir / "best.pt"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
