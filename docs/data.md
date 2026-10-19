# Data format and preparation

## Source scenes

The preparation command accepts a MATLAB variable with shape `(height, width, spectral_bands)`. Complete sensor-specific band selection and removal of invalid bands before calling it. Visible-band indices refer to this processed cube, not the original sensor band numbering.

The default workflow normalizes the whole cube with one global min/max and extracts a central 256 × 256 target. It creates low-resolution HSI using a normalized 20 × 20 Gaussian kernel for ×2/×4 or a 4 × 4 kernel for ×8, then decimates the result. PAN is the mean of the explicitly selected visible bands.

Required experiment choices:

- `--sigma`: positive Gaussian standard deviation; no value is supplied by the paper.
- `--visible-bands`: comma-separated, zero-based indices derived from the sensor wavelengths and preprocessing.

Optional processing controls are `--phase` (decimation offset, default 0) and `--crop-size` (default 256). Kernel boundaries use reflection. Processing settings and the source-file SHA-256 hash are saved beside the MAT file as JSON.

## Prepared samples

Each MAT file contains exactly one sample:

| Variable | Shape | Meaning |
|---|---|---|
| `LRMS` | h × w × S | Low-resolution hyperspectral image |
| `PAN` | H × W | High-resolution panchromatic image |
| `HRMS` | H × W × S | Reference hyperspectral image |

All arrays must be finite and in [0, 1]. The band counts must match, H/h and W/w must equal the configured scale (2, 4, or 8), and target dimensions must be divisible by the window size (default 8). SSIM evaluation requires target dimensions of at least 11 pixels per axis.

## Splits and provenance

Use independent scenes for training, validation, and testing. Keep the generated JSON files next to their MAT files. Training and evaluation reject identical targets or shared source hashes across development/test boundaries. If data are prepared externally, identical targets are still detected, but source-scene provenance cannot be inferred from arbitrary crops; the experimenter must ensure scene independence.

The paper does not specify a complete split protocol. This repository's independent validation and held-out testing are an explicit evaluation protocol rather than a claim about how the published tables were produced.

Example local layout (data are ignored by Git):

```text
data/
├── train/
│   ├── scene_a.mat
│   └── scene_a.json
├── val/
│   ├── scene_b.mat
│   └── scene_b.json
└── test/
    ├── scene_c.mat
    └── scene_c.json
```
