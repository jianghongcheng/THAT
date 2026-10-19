"""Held-out per-image metrics and macro-average; no implicit prediction clipping."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from that import THAT
from that.data import MatDataset, require_disjoint
from that.metrics import metrics


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


@torch.no_grad()
def evaluate(model, dataset, device, clip=False):
    model.eval()
    rows = []
    for i, (ms, pan, gt) in enumerate(DataLoader(dataset, batch_size=1)):
        pred = model(ms.to(device), pan.to(device))[0].permute(1, 2, 0).cpu().numpy()
        if clip:
            pred = pred.clip(0, 1)
        row = metrics(pred, gt[0].permute(1, 2, 0).numpy(), model.factor)
        row["file"] = str(dataset.files[i])
        rows.append(row)
    return dict(
        per_image=rows,
        mean={
            k: float(np.mean([row[k] for row in rows]))
            for k in ["PSNR", "SSIM", "SAM", "ERGAS", "SCC"]
        },
        factor=model.factor,
        prediction_clipping=clip,
        aggregation="macro mean across images",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data", required=True)
    p.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--clip", action="store_true")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = THAT(**ckpt["model_config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    test = MatDataset(args.data)
    overlap = test.scene_ids() & set(ckpt.get("development_scene_ids", []))
    if overlap:
        raise ValueError(
            "Test scenes overlap training/validation scenes stored in checkpoint"
        )
    report = evaluate(model, test, args.device, args.clip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    print(args.output)


if __name__ == "__main__":
    main()
