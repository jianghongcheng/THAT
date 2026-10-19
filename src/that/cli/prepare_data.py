"""Generate Wald-style reduced-resolution pairs; require unspecified physics as input."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import convolve


def wald_pairs(cube, factor, visible_bands, sigma, crop_size=256, phase=0):
    cube = np.asarray(cube, dtype=np.float64)
    if (
        cube.ndim != 3
        or min(cube.shape[:2]) < crop_size
        or crop_size % factor
        or crop_size % 8
    ):
        raise ValueError(
            "Expected HWC cube large enough for crop; crop divisible by scale and 8"
        )
    if (
        factor not in (2, 4, 8)
        or sigma <= 0
        or not np.isfinite(sigma)
        or not 0 <= phase < factor
    ):
        raise ValueError("Invalid scale, sigma or decimation phase")
    if not np.isfinite(cube).all():
        raise ValueError("Nonfinite cube")
    bands = list(visible_bands)
    if (
        not bands
        or len(set(bands)) != len(bands)
        or min(bands) < 0
        or max(bands) >= cube.shape[2]
    ):
        raise ValueError(
            "Specify unique zero-based visible-band indices within the cube"
        )
    lo, hi = cube.min(), cube.max()
    if lo == hi:
        raise ValueError("Cannot normalize constant cube")
    cube = (cube - lo) / (hi - lo)
    top, left = ((n - crop_size) // 2 for n in cube.shape[:2])
    gt = cube[top : top + crop_size, left : left + crop_size, :]
    size = 4 if factor == 8 else 20
    axis = np.arange(size) - (size - 1) / 2
    kernel = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    blurred = convolve(gt, kernel[:, :, None], mode="reflect", origin=0)
    ms = blurred[phase::factor, phase::factor, :]
    pan = gt[:, :, bands].mean(-1)
    return dict(
        LRMS=ms.astype("float32"), PAN=pan.astype("float32"), HRMS=gt.astype("float32")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--factor", type=int, choices=[2, 4, 8], required=True)
    p.add_argument(
        "--visible-bands",
        required=True,
        help="Comma-separated zero-based indices, from sensor wavelengths",
    )
    p.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Gaussian sigma: paper does not specify",
    )
    p.add_argument("--phase", type=int, default=0)
    p.add_argument("--crop-size", type=int, default=256)
    args = p.parse_args()
    bands = [int(x) for x in args.visible_bands.split(",")]
    pairs = wald_pairs(
        loadmat(args.input)[args.key],
        args.factor,
        bands,
        args.sigma,
        args.crop_size,
        args.phase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output, pairs)
    provenance = dict(
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source=str(args.input.resolve()),
        key=args.key,
        factor=args.factor,
        visible_bands=bands,
        sigma=args.sigma,
        phase=args.phase,
        crop_size=args.crop_size,
        kernel_size=4 if args.factor == 8 else 20,
        boundary="reflect",
        normalization="global minmax before center crop",
        convolution_origin=0,
    )
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
