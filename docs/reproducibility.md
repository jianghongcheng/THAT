# Implementation and reproducibility

The supplied paper's equations and architecture figures guide this implementation. The repository does not claim to reproduce the published tables. No pretrained checkpoints or real-data benchmark results are distributed.

## Architecture choices

| Topic | Implementation | Basis or limitation |
|---|---|---|
| Input fusion | Bicubic HSI interpolation, Conv/ReLU, PAN concatenation | Paper pipeline |
| PTSG | LayerNorm; parallel PTSA/WSA; additive fusion into MVFN; residual | Figure 3; branch merge operator is unspecified |
| Width / heads | 180 / 6 | Paper training settings |
| Groups / window | 3 / 8 | Explicit defaults; not numerically specified in the paper |
| PTSA token axis | Channel-feature tokens; 30 × 30 scores per head at width 180 | Inherited tensor formulation; paper does not define token dimensions |
| Clustering scope | Independent two-cluster fit per sample and head | Explicit choice to ensure batch-independent outputs |
| Masking | Higher-mean cluster retained; other logits set to −∞ | Implements prose stating softmax over pivotal entries; diagram instead shows zero logits |
| Empty masks | Retain each empty row's maximum; retain all constant scores | Numerical safeguard absent from the paper |
| MVFN | 3/5/7 depthwise paths, channel chunks, variance/pooling, concatenation, SiLU, projection, residual | Figure 4 |
| Reconstruction | Conv → PixelShuffle(1) → Conv; interpolated-HSI residual | HSI already has target spatial dimensions; paper's final upsampling scale is ambiguous |

The implementation removes the earlier inherited OCAB and nested HAB stack because these do not appear in the supplied architecture figure. The shallow projection remains 64 channels; transformer features use 180 channels. Local variance uses unbiased estimation with reflection padding.

K-means runs on CPU through scikit-learn. Cluster assignments are detached; selected attention scores remain differentiable. Dense attention is still computed. Sparse masking alone should not be interpreted as a demonstrated latency or FLOP reduction.

## Data and optimization

See [data preparation](data.md) for the implemented Wald-style protocol. Gaussian sigma, visible-band indices, boundary handling, and decimation alignment are explicit experiment settings; the paper does not determine all of them.

Training uses L1 loss, Adam (learning rate 5e-4, weight decay 0), 50 epochs, batch size 2, and a factor-0.5 learning-rate decay every 20 epochs. Independent validation PSNR selects the best checkpoint. The paper mentions “iterative alternating” training without specifying alternating variables or updates; this repository does not invent an additional optimization stage.

A fixed training seed and deterministic clustering initialization improve repeatability. Bitwise reproducibility across hardware, library versions, and CUDA kernels is not guaranteed. Record environment versions and processing provenance when reporting results.

## Metric conventions

Inputs use data range 1. Evaluation performs no implicit normalization or clipping.

- **PSNR:** computed from MSE over the complete HWC image.
- **SSIM:** bandwise 11 × 11 Gaussian window, sigma 1.5, valid centers; averaged across bands and spatial positions.
- **SAM:** mean spectral angle in degrees; zero-norm pixels are excluded and their valid count is reported.
- **ERGAS:** `100 / scale × sqrt(mean_band(MSE_band / mean_reference_band²))`. Zero-mean bands with nonzero error yield infinity.
- **SCC:** mean per-band Pearson correlation, following the original local metric code. This is not a high-pass-filtered correlation. Zero-variance bands are excluded and valid counts are reported.

The paper names the metrics without completely specifying these conventions. Dataset-level results are macro means across images. JSON preserves non-finite values as strings (`"inf"`, `"nan"`) instead of emitting invalid JSON.

## Validation scope

The local validation covers 14 numerical/regression tests and a complete CPU synthetic pipeline: pair generation, one training epoch, checkpoint selection/loading, and held-out metrics. It also checks that development scenes cannot be evaluated as held-out test scenes. These checks demonstrate executable code, not published reconstruction quality.
