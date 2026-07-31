from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .losses import GAVEV1Loss
from .metrics import MetricAccumulator
from .model import GAVEV1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.997) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


class WarmupCosine:
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        minimum_ratio: float = 0.03,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, warmup_steps)
        self.minimum_ratio = minimum_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_number = 0

    def step(self) -> None:
        self.step_number += 1
        if self.step_number <= self.warmup_steps:
            ratio = self.step_number / self.warmup_steps
        else:
            progress = (self.step_number - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * ratio

    def state_dict(self) -> dict[str, Any]:
        return {"step_number": self.step_number}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_number = int(state["step_number"])


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


def build_optimizer(model: GAVEV1, config: FitConfig) -> torch.optim.AdamW:
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


def set_encoder_trainable(model: GAVEV1, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(trainable)


def train_one_epoch(
    model: GAVEV1,
    ema: ModelEMA,
    loss_function: GAVEV1Loss,
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

    result = {name: value / max(1, len(loader)) for name, value in totals.items()}
    result["optimizer_updates"] = float(update_count)
    return result


@torch.inference_mode()
def evaluate(
    model: GAVEV1,
    loss_function: GAVEV1Loss,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
    model.eval()
    loss_totals: dict[str, float] = {}
    accumulator = MetricAccumulator(threshold=threshold)
    samples: dict[str, dict[str, Tensor]] = {}
    for batch in loader:
        case_names = batch["id"]
        batch_device = move_batch(batch, device)
        outputs = model(batch_device["rgb"], batch_device.get("ffa"))
        losses = loss_function(outputs, batch_device)
        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.cpu())
        probabilities = model.challenge_probabilities(outputs) * batch_device["roi"]
        for index, case_id in enumerate(case_names):
            sample = {
                "probabilities": probabilities[index].cpu(),
                "av": batch["av"][index].cpu(),
                "roi": batch["roi"][index].cpu(),
                "classes": batch["classes"][index].cpu(),
                "centerlines": batch["centerlines"][index].cpu(),
            }
            samples[case_id] = sample
            accumulator.update(
                case_id=case_id,
                probabilities=sample["probabilities"],
                target=sample["av"],
                roi=sample["roi"],
                classes=sample["classes"],
                centerlines=sample["centerlines"],
            )

    metrics = accumulator.summarize()
    metrics["losses"] = {
        name: value / max(1, len(loader)) for name, value in loss_totals.items()
    }
    return metrics, samples


def save_checkpoint(
    path: Path,
    model: GAVEV1,
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
            "architecture": "GAVEV1-ConvNeXtTiny-StructuredAV",
        },
        path,
    )


def fit(
    model: GAVEV1,
    loss_function: GAVEV1Loss,
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
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    (output_dir / "split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
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
        validation, _ = evaluate(
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
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "training": training,
            "validation": validation,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "seconds": record["seconds"],
                    "train_total": training["total"],
                    "val_total": validation["losses"]["total"],
                    "dice_vessel_0.5": validation["dice_vessel"],
                    "av_accuracy": validation["accuracy_av"],
                    "topology_proxy": validation["topology_proxy"],
                    "threshold": 0.5,
                    "score_proxy": score,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
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
        if device.type == "mps":
            torch.mps.empty_cache()

    summary = {
        "best_epoch": best_epoch,
        "best_score_proxy": best_score,
        "epochs_completed": len(history),
        "best_checkpoint": str(output_dir / "best.pt"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
