from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .engine import ModelEMA, WarmupCosine, move_batch
from .model_core import RetinalVesselNet
from .segmentation_metrics import SegmentationMetricAccumulator
from .topology_losses import TopologyAwareLoss


@dataclass
class FitConfig:
    task: int
    fold: int
    n_folds: int
    seed: int
    height: int
    width: int
    epochs: int
    patience: int
    batch_size: int
    accumulation_steps: int
    learning_rate: float
    encoder_lr_ratio: float
    weight_decay: float
    warmup_epochs: int
    freeze_encoder_epochs: int
    ema_decay: float
    device: str
    data_root: str
    output_dir: str
    pretrained: bool
    num_refinements: int
    hard_gap_weight: float = 0.0
    ffa_root: str | None = None


def build_optimizer(
    model: RetinalVesselNet,
    config: FitConfig,
) -> torch.optim.AdamW:
    encoder_parameters, new_parameters = model.parameter_groups()
    return torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": config.learning_rate * config.encoder_lr_ratio,
            },
            {"params": new_parameters, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )


def set_encoder_trainable(model: RetinalVesselNet, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(trainable)


def train_one_epoch(
    model: RetinalVesselNet,
    ema: ModelEMA,
    loss_function: TopologyAwareLoss,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: WarmupCosine,
    device: torch.device,
    accumulation_steps: int,
    scaler: torch.cuda.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda"
    update_count = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type if device.type in ("cuda", "cpu") else "cpu",
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model(batch["rgb"], batch.get("ffa"))
            losses = loss_function(outputs, batch)
            backward_loss = losses["total"] / accumulation_steps
        scaler.scale(backward_loss).backward()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())

        should_update = step % accumulation_steps == 0 or step == len(loader)
        if should_update:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(model)
            update_count += 1

    result = {
        name: value / max(1, len(loader))
        for name, value in totals.items()
    }
    result["optimizer_updates"] = float(update_count)
    return result


@torch.inference_mode()
def evaluate(
    model: RetinalVesselNet,
    loss_function: TopologyAwareLoss,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    loss_totals: dict[str, float] = {}
    accumulator = SegmentationMetricAccumulator(threshold=threshold)
    for batch in loader:
        case_names = batch["id"]
        batch_device = move_batch(batch, device)
        outputs = model(batch_device["rgb"], batch_device.get("ffa"))
        losses = loss_function(outputs, batch_device)
        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.cpu())
        probabilities = model.challenge_probabilities(outputs) * batch_device["roi"]
        for index, case_id in enumerate(case_names):
            accumulator.update(
                case_id=case_id,
                probabilities=probabilities[index],
                target=batch["av"][index],
                roi=batch["roi"][index],
                classes=batch["classes"][index],
                centerlines=batch["centerlines"][index],
            )

    metrics = accumulator.summarize()
    metrics["losses"] = {
        name: value / max(1, len(loader))
        for name, value in loss_totals.items()
    }
    return metrics


def save_checkpoint(
    path: Path,
    model: RetinalVesselNet,
    config: FitConfig,
    split: dict[str, list[str]],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "split": split,
            "epoch": epoch,
            "metrics": metrics,
            "architecture": model.architecture_name,
        },
        path,
    )


def fit_topology_model(
    model: RetinalVesselNet,
    loss_function: TopologyAwareLoss,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    split: dict[str, list[str]],
    config: FitConfig,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model.to(device)
    loss_function.to(device)
    optimizer = build_optimizer(model, config)
    updates_per_epoch = math.ceil(
        len(train_loader) / max(1, config.accumulation_steps)
    )
    scheduler = WarmupCosine(
        optimizer,
        total_steps=config.epochs * updates_per_epoch,
        warmup_steps=config.warmup_epochs * updates_per_epoch,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=config.ema_decay)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    bad_epochs = 0

    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(split, indent=2),
        encoding="utf-8",
    )

    for epoch in range(1, config.epochs + 1):
        set_encoder_trainable(model, epoch > config.freeze_encoder_epochs)
        start = time.perf_counter()
        training = train_one_epoch(
            model,
            ema,
            loss_function,
            train_loader,
            optimizer,
            scheduler,
            device,
            config.accumulation_steps,
            scaler,
        )
        validation = evaluate(
            ema.module,
            loss_function,
            validation_loader,
            device,
            threshold=0.5,
        )
        score = float(validation["task_score_proxy"])
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - start,
            "learning_rates": [
                group["lr"] for group in optimizer.param_groups
            ],
            "training": training,
            "validation": validation,
        }
        history.append(record)
        artery = validation["artery"]
        vein = validation["vein"]
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "seconds": record["seconds"],
                    "train_total": training["total"],
                    "val_total": validation["losses"]["total"],
                    "dice_vessel": validation["dice_vessel"],
                    "a_sen": artery["sensitivity"],
                    "v_sen": vein["sensitivity"],
                    "classification": validation["classification_component"],
                    "feasibility_a": validation["feasibility_artery"],
                    "feasibility_v": validation["feasibility_vein"],
                    "topology_proxy": validation["topology_proxy"],
                    "score_proxy": score,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        save_checkpoint(
            output_dir / "last.pt",
            ema.module,
            config,
            split,
            epoch,
            validation,
        )
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                output_dir / "best.pt",
                ema.module,
                config,
                split,
                epoch,
                validation,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(
                    f"early_stop epoch={epoch} best_epoch={best_epoch} "
                    f"best_score={best_score:.6f}",
                    flush=True,
                )
                break

    summary = {
        "best_epoch": best_epoch,
        "best_score_proxy": best_score,
        "epochs_completed": len(history),
        "best_checkpoint": str(output_dir / "best.pt"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
