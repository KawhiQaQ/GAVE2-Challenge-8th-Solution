#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gave2v1.data import GAVE2Dataset, case_ids
from gave2v1.engine import (
    ModelEMA,
    WarmupCosine,
    seed_everything,
    select_device,
)
from gave2v1.engine_v2 import (
    V2FitConfig,
    build_optimizer,
    save_checkpoint,
    set_encoder_trainable,
    train_one_epoch,
)
from gave2v1.losses_v2 import GAVEV2Loss, V2LossWeights
from gave2v1.model_v2 import warm_start_v2
from gave2v1.model_v14 import (
    RETFOUND_INPUT_SIZE,
    RETFOUND_LAYER_INDICES,
    build_v14,
    warm_start_v14,
)
from gave2v1.model_v28 import build_v28
from gave2v1.model_v30 import build_v30, warm_start_v30
from gave2v1.model_v54 import build_v54, warm_start_v54


@dataclass
class FullVariantConfig(V2FitConfig):
    full_training: bool = True
    variant: str = ""
    native_final_refiner: bool = False
    retfound_checkpoint: str = ""
    retfound_sha256: str = ""
    retfound_input_height: int = RETFOUND_INPUT_SIZE[0]
    retfound_input_width: int = RETFOUND_INPUT_SIZE[1]
    retfound_layer_indices: tuple[int, ...] = RETFOUND_LAYER_INDICES
    retfound_frozen: bool = True
    physiology_detail: bool = False
    optical_density_detail: bool = False
    phase_balance_detail: bool = False
    r2_steps: int = 0
    public_r2_weights_loaded: bool = False
    same_fold_v1_first_stage: bool = False


def parse_size(value: str) -> tuple[int, int]:
    height, width = (int(part) for part in value.lower().split("x"))
    if height % 32 or width % 32:
        raise argparse.ArgumentTypeError(
            "height and width must be divisible by 32"
        )
    return height, width


def parse_retfound_size(value: str) -> tuple[int, int]:
    height, width = (int(part) for part in value.lower().split("x"))
    if height % 14 or width % 14:
        raise argparse.ArgumentTypeError(
            "RETFound height and width must be divisible by 14"
        )
    return height, width


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a promoted GAVE2 variant on all 50 labels."
    )
    parser.add_argument(
        "--variant",
        choices=("v14", "v28", "v30", "v54"),
        required=True,
    )
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ffa-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--retfound-checkpoint")
    parser.add_argument(
        "--retfound-size",
        type=parse_retfound_size,
        default=RETFOUND_INPUT_SIZE,
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--size", type=parse_size, default=(1024, 1536))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--encoder-lr-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--num-refinements", type=int, default=3)
    parser.add_argument("--r2-steps", type=int, default=5)
    parser.add_argument("--hard-gap-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.hard_gap_weight < 0:
        raise ValueError("--hard-gap-weight must be non-negative")
    if args.r2_steps < 1:
        raise ValueError("--r2-steps must be positive")
    if args.task == 1 and args.ffa_root:
        raise ValueError("Task1 full training is strictly CFP-only")
    if args.task == 2 and not args.ffa_root:
        raise ValueError("Task2 full training requires --ffa-root")
    if args.variant == "v14" and not args.retfound_checkpoint:
        raise ValueError("V14 requires --retfound-checkpoint")
    if args.variant != "v14" and args.retfound_checkpoint:
        raise ValueError(
            f"{args.variant.upper()} must not receive --retfound-checkpoint"
        )

    data_root = Path(args.data_root).resolve()
    ffa_root = Path(args.ffa_root).resolve() if args.ffa_root else None
    output_dir = Path(args.output_dir).resolve()
    init_checkpoint = Path(args.init_checkpoint).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    seed_everything(args.seed)
    ids = case_ids(data_root, "training")
    if len(ids) != 50:
        raise ValueError(f"Expected 50 labeled cases, found {len(ids)}")
    height, width = args.size

    dataset = GAVE2Dataset(
        data_root,
        ids,
        task=args.task,
        size=(height, width),
        augment=True,
        cache=not args.no_cache,
        ffa_root=ffa_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    retfound_report: dict[str, object] | None = None
    if args.variant == "v54":
        model = build_v54(
            task=args.task,
            pretrained=False,
            r2_steps=args.r2_steps,
        )
        initialization = {
            "source": str(init_checkpoint),
            "same_fold_v1_first_stage": True,
            "public_r2_weights_loaded": False,
            **warm_start_v54(model, init_checkpoint),
        }
    elif args.variant == "v28":
        model = build_v28(
            task=args.task,
            pretrained=False,
            num_refinements=args.num_refinements,
        )
        initialization: dict[str, object] = {
            "source": str(init_checkpoint),
            **warm_start_v2(model, init_checkpoint),
        }
    elif args.variant == "v30":
        model = build_v30(
            task=args.task,
            pretrained=False,
            num_refinements=args.num_refinements,
        )
        initialization = {
            "source": str(init_checkpoint),
            **warm_start_v30(model, init_checkpoint),
        }
    else:
        model = build_v14(
            task=args.task,
            retfound_checkpoint=args.retfound_checkpoint,
            pretrained=False,
            num_refinements=args.num_refinements,
            retfound_input_size=args.retfound_size,
        )
        retfound_report = model.retfound_encoder.load_report
        initialization = {
            "source": str(init_checkpoint),
            "retfound": retfound_report,
            "parent_warm_start": warm_start_v14(
                model,
                init_checkpoint,
            ),
        }

    config = FullVariantConfig(
        task=args.task,
        fold=-1,
        n_folds=5,
        seed=args.seed,
        height=height,
        width=width,
        epochs=args.epochs,
        patience=0,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        encoder_lr_ratio=args.encoder_lr_ratio,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        ema_decay=args.ema_decay,
        device=str(device),
        data_root=str(data_root),
        output_dir=str(output_dir),
        pretrained=False,
        num_refinements=(
            args.r2_steps
            if args.variant == "v54"
            else args.num_refinements
        ),
        hard_gap_weight=args.hard_gap_weight,
        ffa_root=str(ffa_root) if ffa_root is not None else None,
        variant=args.variant,
        native_final_refiner=(
            args.variant == "v28"
            or (args.variant == "v30" and args.task == 1)
            or (args.variant == "v54" and args.task == 1)
        ),
        physiology_detail=(
            args.variant == "v30"
            or (args.variant == "v54" and args.task == 1)
        ),
        optical_density_detail=(
            args.variant in ("v30", "v54") and args.task == 1
        ),
        phase_balance_detail=args.variant == "v30" and args.task == 2,
        retfound_checkpoint=(
            str(Path(args.retfound_checkpoint).resolve())
            if args.retfound_checkpoint
            else ""
        ),
        retfound_sha256=(
            model.retfound_encoder.checkpoint_sha256
            if args.variant == "v14"
            else ""
        ),
        retfound_input_height=args.retfound_size[0],
        retfound_input_width=args.retfound_size[1],
        retfound_layer_indices=tuple(RETFOUND_LAYER_INDICES),
        retfound_frozen=True,
        r2_steps=args.r2_steps if args.variant == "v54" else 0,
        public_r2_weights_loaded=False,
        same_fold_v1_first_stage=args.variant == "v54",
    )
    split = {"training": ids, "validation": []}
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(split, indent=2),
        encoding="utf-8",
    )
    (output_dir / "initialization.json").write_text(
        json.dumps(initialization, indent=2),
        encoding="utf-8",
    )

    model.to(device)
    loss_function = GAVEV2Loss(
        weights=V2LossWeights(hard_gap=args.hard_gap_weight),
    ).to(device)
    optimizer = build_optimizer(model, config)
    updates_per_epoch = math.ceil(
        len(loader) / max(1, args.accumulation_steps)
    )
    scheduler = WarmupCosine(
        optimizer,
        total_steps=args.epochs * updates_per_epoch,
        warmup_steps=args.warmup_epochs * updates_per_epoch,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=args.ema_decay)
    history: list[dict[str, object]] = []

    print(
        json.dumps(
            {
                "architecture": model.architecture_name,
                "variant": args.variant,
                "device": str(device),
                "total_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "task": args.task,
                "task1_ffa_loaded": False,
                "training_cases": len(ids),
                "epochs": args.epochs,
                "size": [height, width],
                "hard_gap_weight": args.hard_gap_weight,
                "r2_steps": (
                    args.r2_steps if args.variant == "v54" else None
                ),
                "ffa_root": (
                    str(ffa_root) if ffa_root is not None else None
                ),
                "updates_per_epoch": updates_per_epoch,
                "retfound": retfound_report,
                "initialization": initialization,
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        set_encoder_trainable(model, epoch > args.freeze_encoder_epochs)
        started = time.perf_counter()
        training = train_one_epoch(
            model,
            ema,
            loss_function,
            loader,
            optimizer,
            scheduler,
            device,
            args.accumulation_steps,
            scaler,
        )
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "learning_rates": [
                group["lr"] for group in optimizer.param_groups
            ],
            "training": training,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "seconds": record["seconds"],
                    "train_total": training["total"],
                    "learning_rates": record["learning_rates"],
                }
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
            {"training": training, "full_training": True},
        )

    save_checkpoint(
        output_dir / "final.pt",
        ema.module,
        config,
        split,
        args.epochs,
        {"training": history[-1]["training"], "full_training": True},
    )
    summary = {
        "variant": args.variant,
        "epochs_completed": args.epochs,
        "optimizer_updates": args.epochs * updates_per_epoch,
        "final_checkpoint": str(output_dir / "final.pt"),
        "initialization": initialization,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
