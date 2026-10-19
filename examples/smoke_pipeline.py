"""Run a complete synthetic experiment without downloading any dataset."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.io import savemat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/smoke"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(
            "Use an empty output directory to avoid overwriting an experiment."
        )
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")

    def run(module, *arguments):
        subprocess.run(
            [sys.executable, "-m", module, *map(str, arguments)], env=env, check=True
        )

    for seed, split in enumerate(["train", "train", "val", "test"]):
        raw = root / f"source_{seed}.mat"
        savemat(raw, {"cube": np.random.RandomState(seed).rand(32, 32, 5)})
        run(
            "that.cli.prepare_data",
            "--input",
            raw,
            "--key",
            "cube",
            "--output",
            root / split / f"{seed}.mat",
            "--factor",
            4,
            "--visible-bands",
            "0,1",
            "--sigma",
            1.5,
            "--crop-size",
            16,
        )

    checkpoint = root / "model.pt"
    run(
        "that.cli.train",
        "--data",
        root / "train",
        "--val-data",
        root / "val",
        "--bands",
        5,
        "--factor",
        4,
        "--epochs",
        1,
        "--device",
        "cpu",
        "--output",
        checkpoint,
    )
    result = root / "evaluation.json"
    run(
        "that.cli.evaluate",
        "--checkpoint",
        checkpoint,
        "--data",
        root / "test",
        "--device",
        "cpu",
        "--output",
        result,
    )
    print(json.dumps(json.loads(result.read_text())["mean"], indent=2))
    print(f"Synthetic experiment completed: {root}")


if __name__ == "__main__":
    main()
