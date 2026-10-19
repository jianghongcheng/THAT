import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


class MatDataset(Dataset):
    """One HWC MAT pair per file. Independent scenes must be split by caller."""

    def __init__(self, path):
        path = Path(path)
        self.files = [path] if path.is_file() else sorted(path.glob("*.mat"))
        if not self.files:
            raise ValueError(f"No MAT files found: {path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        d = loadmat(self.files[index])
        ms, pan, gt = (
            np.asarray(d[k], dtype=np.float32) for k in ["LRMS", "PAN", "HRMS"]
        )
        if ms.ndim != 3 or gt.ndim != 3 or pan.ndim != 2:
            raise ValueError("Expected LRMS/HRMS HWC, PAN HW")
        if ms.shape[2] != gt.shape[2] or pan.shape != gt.shape[:2]:
            raise ValueError("MS/GT band counts or PAN/GT sizes differ")
        ratios = (gt.shape[0] / ms.shape[0], gt.shape[1] / ms.shape[1])
        if ratios[0] != ratios[1] or ratios[0] not in (2, 4, 8):
            raise ValueError("HR/LR scale must be equal in both axes and 2/4/8")
        for a in [ms, pan, gt]:
            if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                raise ValueError("Expected finite [0,1] data")
        return (
            torch.from_numpy(ms.transpose(2, 0, 1).copy()),
            torch.from_numpy(pan[None].copy()),
            torch.from_numpy(gt.transpose(2, 0, 1).copy()),
        )

    def scene_ids(self):
        result = set()
        for f in self.files:
            # Detect identical targets across separately named or serialized files.
            gt = np.asarray(loadmat(f)["HRMS"], dtype=np.float64)
            result.add("target:" + hashlib.sha256(gt.tobytes()).hexdigest())
            meta = f.with_suffix(".json")
            if meta.exists():
                source = json.loads(meta.read_text()).get("source_sha256")
                if source:
                    result.add("source:" + source)
        return result


def require_disjoint(*datasets):
    seen = set()
    for dataset in datasets:
        ids = dataset.scene_ids()
        if ids & seen:
            raise ValueError(
                "Data leakage: splits share a target or recorded source scene"
            )
        seen |= ids
