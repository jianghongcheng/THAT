"""Supervised L1 training; validation selects best checkpoint, test is separate."""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from that import THAT
from that.data import MatDataset, require_disjoint
from that.cli.evaluate import evaluate, json_safe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--bands", type=int, required=True)
    p.add_argument("--factor", type=int, choices=[2, 4, 8], default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--num-groups",
        type=int,
        default=3,
        help="Number of Pivotal Token Selective Groups",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path("runs/that.pt"))
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    torch.manual_seed(args.seed)
    train, val = MatDataset(args.data), MatDataset(args.val_data)
    require_disjoint(train, val)
    development_scene_ids = sorted(train.scene_ids() | val.scene_ids())
    config = dict(
        in_ch=args.bands,
        factor=args.factor,
        embed_dim=180,
        num_heads=6,
        num_groups=args.num_groups,
    )
    model = THAT(**config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    criterion = torch.nn.L1Loss()
    best = float("-inf")
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        lr = optimizer.param_groups[0]["lr"]
        for ms, pan, gt in loader:
            ms, pan, gt = (t.to(args.device) for t in (ms, pan, gt))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ms, pan), gt)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * ms.shape[0]
        scheduler.step()
        report = evaluate(model, val, args.device)
        score = report["mean"]["PSNR"]
        record = dict(
            epoch=epoch + 1,
            loss=total / len(train),
            learning_rate=lr,
            validation=report["mean"],
        )
        history.append(record)
        print(json.dumps(json_safe(record)), flush=True)
        checkpoint = dict(
            model=model.state_dict(),
            model_config=config,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            epoch=epoch + 1,
            development_scene_ids=development_scene_ids,
            format_version="that-v1",
            training_config={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
        )
        torch.save(checkpoint, args.output.with_name(args.output.stem + ".last.pt"))
        if score > best:
            best = score
            torch.save(checkpoint, args.output)
        args.output.with_suffix(".history.json").write_text(
            json.dumps(json_safe(history), indent=2, allow_nan=False)
        )


if __name__ == "__main__":
    main()
