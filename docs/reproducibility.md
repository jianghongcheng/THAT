# Implementation notes

## Model configuration

| Component | Configuration |
|---|---|
| Input projection | Bicubic interpolation, 1 × 1 Conv/ReLU, PAN concatenation |
| Shallow / transformer channels | 64 / 180 |
| Attention heads | 6 |
| PTSG groups / window size | 3 / 8 |
| PTSG fusion | LayerNorm → parallel PTSA/WSA → sum → MVFN → residual |
| MVFN | 3 × 3, 5 × 5, and 7 × 7 depthwise paths with direct, variance, and pooling features |
| Reconstruction | Conv → PixelShuffle(1) → Conv, plus interpolated-HSI residual |

PTSA computes channel-feature attention independently for each sample and head. With 180 channels and 6 heads, each attention matrix is 30 × 30. Two-cluster K-means retains the higher-mean score cluster; masked logits are set to −∞ before softmax. An empty row retains its maximum score. Constant-score inputs retain all entries.

K-means uses scikit-learn on CPU. Cluster assignments are detached, while retained scores receive gradients. Attention matrix products remain dense. MVFN computes unbiased local variance with reflection padding.

## Relationship to the paper

The paper specifies 180 channels, 6 attention heads, the PTSA formulation, and the MVFN branches. Group count, window size, branch merge operator, clustering scope, and token dimensions require implementation conventions; the table above records the configuration used here.

The paper's reconstruction description combines interpolation to the target resolution with a final PixelShuffle stage without a stated scale. This implementation keeps the target resolution throughout reconstruction and uses PixelShuffle(1). Masked logits use −∞ to implement softmax over selected entries, whereas the paper illustration shows zero logits. The PTSG contains PTSA, WSA, and MVFN; it does not include OCAB or an additional nested HAB stack.

Published figures in the README report the paper's experiments. Benchmark values have not been re-established with this implementation, and pretrained weights are not distributed.

## Optimization

Training uses L1 loss and Adam with learning rate 5e-4, weight decay 0, batch size 2, and 50 epochs. The learning rate is halved every 20 epochs. Validation PSNR selects the best checkpoint; the final checkpoint and training history are also saved.

The training loop uses a single L1 optimization objective. The paper's reference to alternating training does not define additional update rules. Dataset splits and processing settings must be recorded when comparing experiments. See [data preparation](data.md).

Use `--seed` to set the training seed. Clustering uses a fixed initialization seed. Hardware and library versions may affect numerical results.

## Evaluation metrics

Inputs use data range 1. Metrics are computed per image and averaged across images. Prediction clipping is controlled by `--clip` and recorded in the output. ERGAS uses the configured spatial scale (×2, ×4, or ×8). Evaluation rejects targets or recorded source scenes that overlap training or validation data.

| Metric | Definition |
|---|---|
| PSNR | MSE over all pixels and bands, with peak value 1 |
| SSIM | Bandwise 11 × 11 Gaussian window, sigma 1.5; valid-center average |
| SAM | Mean spectral angle in degrees |
| ERGAS | `100 / scale × sqrt(mean_band(MSE_band / mean_reference_band²))` |
| SCC | Mean bandwise Pearson correlation |

SAM excludes zero-norm pixels; SCC excludes zero-variance bands. Valid counts are included in each result. ERGAS returns infinity for a zero-mean reference band with nonzero error. JSON represents non-finite results as strings (`"inf"`, `"nan"`).

## Software validation

Regression tests cover attention masks, batch independence, branch routing, gradients, scale factors, analytical metric values, degradation, and split overlap. The synthetic example exercises data preparation, one training epoch, checkpoint loading, and held-out evaluation. GitHub Actions runs these checks on Python 3.10 and 3.11.
