#!/usr/bin/env python3
"""Train the frozen-parent V17 vein-density refiner on all labeled cases."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from gave2v1.data import GAVE2Dataset, case_ids
from gave2v1.engine import (
    WarmupCosine,
    move_batch,
    seed_everything,
    select_device,
)
from gave2v1.model_v17 import GAVEV17Task3, load_v8_parent
from train_v17_task3_vein import (
    V17Config,
    masked_vein_loss,
    policy_aligned_vein_loss,
    update_ema,
)


def parse_size(value: str) -> tuple[int, int]:
    height, width = (int(part) for part in value.lower().split("x"))
    if height % 32 or width % 32:
        raise argparse.ArgumentTypeError(
            "height and width must be divisible by 32"
        )
    return height, width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ffa-root", required=True)
    parser.add_argument("--zone-c-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=parse_size, default=(1024, 1536))
    parser.add_argument("--epochs", type=int, required=True)
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
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    ffa_root = Path(args.ffa_root).resolve()
    zone_c_dir = Path(args.zone_c_dir).resolve()
    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    ids = case_ids(data_root, "training")
    missing_regions = sorted(
        f"{case_id}.png"
        for case_id in ids
        if not (zone_c_dir / f"{case_id}.png").is_file()
    )
    if missing_regions:
        raise FileNotFoundError(
            f"Missing {len(missing_regions)} C-zone masks"
        )

    seed_everything(args.seed)
    device = select_device(args.device)
    parent, parent_report = load_v8_parent(parent_checkpoint)
    model = GAVEV17Task3(parent).to(device)
    height, width = args.size
    dataset = GAVE2Dataset(
        data_root,
        ids,
        task=2,
        split="training",
        size=(height, width),
        augment=True,
        cache=True,
        ffa_jitter=4,
        ffa_root=ffa_root,
        region_dir=zone_c_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    config = V17Config(
        fold=-1,
        n_folds=5,
        seed=args.seed,
        height=height,
        width=width,
        epochs=args.epochs,
        patience=0,
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
        selection_metric="full_training_only",
    )
    split = {"training": ids, "validation": []}
    output_dir.mkdir(parents=True)
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
        len(loader) / max(1, args.accumulation_steps)
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

    history: list[dict[str, float | int]] = []
    use_amp = device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        totals = {"total": 0.0, "bce": 0.0, "dice": 0.0}
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            batch_device = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
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
                or step == len(loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.refiner.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(ema, model.refiner, args.ema_decay)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "train_total": totals["total"] / len(loader),
            "train_bce": totals["bce"] / len(loader),
            "train_dice": totals["dice"] / len(loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record), flush=True)

    checkpoint = {
        "refiner": ema.state_dict(),
        "config": asdict(config),
        "split": split,
        "parent": parent_report,
        "epoch": args.epochs,
        "metrics": {"train_total": history[-1]["train_total"]},
        "architecture": GAVEV17Task3.architecture_name,
    }
    torch.save(checkpoint, output_dir / "final.pt")
    summary = {
        "epochs_completed": args.epochs,
        "optimizer_updates": args.epochs * updates_per_epoch,
        "loss_policy": args.loss_policy,
        "final_checkpoint": str(output_dir / "final.pt"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
