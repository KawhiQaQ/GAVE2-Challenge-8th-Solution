#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
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
from gave2v1.training_engine import (
    FitConfig,
    build_optimizer,
    save_checkpoint,
    set_encoder_trainable,
    train_one_epoch,
)
from gave2v1.model_core import (
    build_retinal_vessel_net,
    warm_start_retinal_vessel_net,
)
from gave2v1.model_phase_geometry import (
    build_phase_geometry_net,
    warm_start_phase_geometry_net,
)
from gave2v1.topology_losses import TopologyAwareLoss, TopologyLossWeights


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "GAVE2_preliminary"


def parse_size(value: str) -> tuple[int, int]:
    height, width = (int(part) for part in value.lower().split("x"))
    if height % 32 or width % 32:
        raise argparse.ArgumentTypeError(
            "height and width must be divisible by 32"
        )
    return height, width


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a VascFusion-Quant measurement branch on all labels."
    )
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument(
        "--ffa-root",
        help=(
            "Optional root containing split/FFA_A and split/FFA_AV. "
            "Task1 must omit this; Task2 may use a frozen registered cache."
        ),
    )
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument(
        "--hard-gap-weight",
        type=float,
        default=0.0,
        help="Weight for the weakest-centerline gap loss.",
    )
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--phase-geometry",
        action="store_true",
        help=(
            "Use the full FFA phase-geometry architecture. This is "
            "Task 2-only and requires --ffa-root."
        ),
    )
    parser.add_argument("--init-checkpoint", required=True)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.hard_gap_weight < 0:
        raise ValueError("--hard-gap-weight must be non-negative")
    data_root = Path(args.data_root).resolve()
    ffa_root = Path(args.ffa_root).resolve() if args.ffa_root else None
    if args.task == 1 and ffa_root is not None:
        raise ValueError("Task1 full training must not receive --ffa-root")
    if args.phase_geometry and args.task != 2:
        raise ValueError("--phase-geometry is valid only for Task 2")
    if args.phase_geometry and ffa_root is None:
        raise ValueError("--phase-geometry requires --ffa-root")
    output_dir = Path(args.output_dir).resolve()
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
    if args.phase_geometry:
        model = build_phase_geometry_net(
            phase_pretrained=not args.no_pretrained,
            num_refinements=args.num_refinements,
        )
        initialization = {
            "source": str(Path(args.init_checkpoint).resolve()),
            **warm_start_phase_geometry_net(model, args.init_checkpoint),
        }
    else:
        model = build_retinal_vessel_net(
            args.task,
            pretrained=not args.no_pretrained,
            num_refinements=args.num_refinements,
        )
        initialization = {
            "source": str(Path(args.init_checkpoint).resolve()),
            **warm_start_retinal_vessel_net(model, args.init_checkpoint),
        }
    config = FitConfig(
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
        pretrained=not args.no_pretrained,
        num_refinements=args.num_refinements,
        hard_gap_weight=args.hard_gap_weight,
        ffa_root=str(ffa_root) if ffa_root is not None else None,
    )
    split = {"training": ids, "validation": []}
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "full_training": True,
                "probability_refinement_recipe": args.hard_gap_weight == 0.0,
                "hard_gap_recipe": args.hard_gap_weight > 0.0,
                "phase_geometry": args.phase_geometry,
            },
            indent=2,
        ),
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
    loss_function = TopologyAwareLoss(
        weights=TopologyLossWeights(hard_gap=args.hard_gap_weight),
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
                "device": str(device),
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "task": args.task,
                "training_cases": len(ids),
                "epochs": args.epochs,
                "size": [height, width],
                "hard_gap_weight": args.hard_gap_weight,
                "ffa_root": (
                    str(ffa_root) if ffa_root is not None else None
                ),
                "updates_per_epoch": updates_per_epoch,
                "phase_geometry": args.phase_geometry,
                "initial_phase_fusion_strengths": (
                    model.phase_fusion_strengths().detach().tolist()
                    if args.phase_geometry
                    else None
                ),
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
