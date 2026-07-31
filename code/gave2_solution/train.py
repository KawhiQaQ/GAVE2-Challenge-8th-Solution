#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from torch.utils.data import DataLoader

from gave2v1.data import GAVE2Dataset, make_balanced_folds
from gave2v1.engine import FitConfig, fit, seed_everything, select_device
from gave2v1.losses import GAVEV1Loss
from gave2v1.model import build_v1, warm_start_from_checkpoint


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "GAVE2_preliminary"


def parse_size(value: str) -> tuple[int, int]:
    height, width = (int(part) for part in value.lower().split("x"))
    if height % 32 or width % 32:
        raise argparse.ArgumentTypeError("height and width must be divisible by 32")
    return height, width


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the strong GAVE2 V1 model.")
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir")
    parser.add_argument("--size", type=parse_size, default=(768, 1152))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-ratio", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=2)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not 0 <= args.fold < args.n_folds:
        raise ValueError("--fold must be in [0, n_folds)")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else ROOT / "runs" / f"task{args.task}" / f"fold{args.fold}"
    )
    device = select_device(args.device)
    seed_everything(args.seed + args.fold)
    folds = make_balanced_folds(data_root, args.n_folds, args.seed)
    split = folds[args.fold]
    height, width = args.size

    train_dataset = GAVE2Dataset(
        data_root,
        split["training"],
        task=args.task,
        size=(height, width),
        augment=True,
        cache=not args.no_cache,
    )
    validation_dataset = GAVE2Dataset(
        data_root,
        split["validation"],
        task=args.task,
        size=(height, width),
        augment=False,
        cache=not args.no_cache,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_v1(args.task, pretrained=not args.no_pretrained)
    initialization: dict[str, object] = {"source": "ImageNet"}
    if args.init_checkpoint:
        initialization = {
            "source": str(Path(args.init_checkpoint).resolve()),
            **warm_start_from_checkpoint(model, args.init_checkpoint),
        }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = FitConfig(
        task=args.task,
        fold=args.fold,
        n_folds=args.n_folds,
        seed=args.seed,
        height=height,
        width=width,
        epochs=args.epochs,
        patience=args.patience,
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
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": parameter_count,
                "task": args.task,
                "fold": args.fold,
                "train_cases": len(split["training"]),
                "validation_cases": len(split["validation"]),
                "size": [height, width],
                "estimated_optimizer_updates_per_epoch": math.ceil(
                    len(train_loader) / args.accumulation_steps
                ),
                "initialization": initialization,
            },
            indent=2,
        ),
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "initialization.json").write_text(
        json.dumps(initialization, indent=2), encoding="utf-8"
    )
    summary = fit(
        model,
        GAVEV1Loss(),
        train_loader,
        validation_loader,
        split,
        config,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
