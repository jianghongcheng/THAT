"""Reduced-resolution hyperspectral image quality metrics."""

import numpy as np
from scipy.ndimage import convolve


def metrics(prediction, reference, factor):
    p, g = (np.asarray(x, dtype=np.float64) for x in (prediction, reference))
    if p.shape != g.shape or p.ndim != 3 or factor not in (2, 4, 8):
        raise ValueError("Expected equal HWC arrays and scale 2/4/8")
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("Nonfinite metric input")
    if min(g.shape[:2]) < 11:
        raise ValueError("SSIM requires H,W >= 11")
    mse = np.mean((p - g) ** 2)
    psnr = float("inf") if mse == 0 else -10 * np.log10(mse)
    # Bandwise SSIM, Gaussian 11x11 sigma1.5, valid centers, L=1.
    a = np.arange(-5, 6)
    w = np.exp(-a * a / (2 * 1.5**2))
    w /= w.sum()
    kernel = (w[:, None] * w[None, :])[:, :, None]
    smooth = lambda x: convolve(x, kernel, mode="reflect")[5:-5, 5:-5, :]
    mu_p, mu_g = smooth(p), smooth(g)
    vp, vg, cov = (
        smooth(p * p) - mu_p**2,
        smooth(g * g) - mu_g**2,
        smooth(p * g) - mu_p * mu_g,
    )
    ssim = np.mean(
        ((2 * mu_p * mu_g + 0.01**2) * (2 * cov + 0.03**2))
        / ((mu_p**2 + mu_g**2 + 0.01**2) * (vp + vg + 0.03**2))
    )
    # SAM: degrees, ignore pixels with a zero spectral norm (report valid count).
    norm = np.linalg.norm(p, axis=-1) * np.linalg.norm(g, axis=-1)
    valid = norm > 0
    angles = np.arccos(np.clip(np.sum(p * g, axis=-1)[valid] / norm[valid], -1, 1))
    sam = np.degrees(angles).mean() if valid.any() else float("nan")
    # ERGAS uses actual HR/LR scale and per-band reference means.
    means = g.mean((0, 1))
    band_mse = ((p - g) ** 2).mean((0, 1))
    ratios = np.divide(band_mse, means**2, out=np.zeros_like(means), where=means != 0)
    ratios[(means == 0) & (band_mse > 0)] = np.inf
    ergas = 100 / factor * np.sqrt(ratios.mean())
    # SCC: mean bandwise Pearson correlation.
    pc, gc = p - p.mean((0, 1)), g - g.mean((0, 1))
    den = np.sqrt((pc**2).sum((0, 1)) * (gc**2).sum((0, 1)))
    good = den > 0
    cc = np.sum(pc * gc, axis=(0, 1))[good] / den[good]
    scc = np.clip(cc, -1, 1).mean() if good.any() else float("nan")
    return dict(
        PSNR=float(psnr),
        SSIM=float(ssim),
        SAM=float(sam),
        ERGAS=float(ergas),
        SCC=float(scc),
        SAM_valid_pixels=int(valid.sum()),
        SCC_valid_bands=int(good.sum()),
    )
