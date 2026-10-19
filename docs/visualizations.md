# Figure gallery

Figures from [THAT, IEEE SMC 2025](https://doi.org/10.1109/SMC58881.2025.11343459). Source files and checksums are recorded in [`assets/figures/sources.json`](../assets/figures/sources.json).

## Network architecture

![Overall THAT architecture](../assets/figures/architecture.png)

Source: `PTST.pdf`. The paper diagram includes shallow fusion, PTSG stages, and reconstruction. See [implementation notes](reproducibility.md) for model configuration.

## Pivotal relations and adaptive features

![Paper illustration of pivotal relations and adaptive features](../assets/figures/feature-visualization.png)

Source: `comparison.pdf`. Pivotal relations and adaptive feature extraction.

## Token selection

![Query, relevant, and irrelevant token illustration](../assets/figures/token-selection.png)

Source: `example.pdf`. Query, relevant, and irrelevant token examples.

## MVFN

<p align="center">
  <img src="../assets/figures/mvfn.png" alt="Multi-level Variance-aware Feed-forward Network" width="540">
</p>

Source: `MVFN.pdf`. Three depthwise-convolution paths combine direct, local-variance, and pooled features with SiLU, concatenation, projection, and residual addition.

## Botswana ×2: qualitative comparison

![Paper Botswana times-two reconstruction comparison](../assets/figures/botswana-x2-comparison.png)

Source: `result.pdf`. Reconstruction views and enlarged regions for the compared methods, THAT (“Ours”), and ground truth. The dB labels are the values shown in the paper figure.

## Botswana ×2: band-wise PSNR

![Botswana times-two PSNR by spectral band](../assets/figures/botswana-x2-band-psnr.png)

Source: `Botswana_2_20_psnr.pdf`. Horizontal axis: spectral-band index. Vertical axis: PSNR in dB.

## PaviaC ×2: band-wise PSNR

![PaviaC times-two PSNR by spectral band](../assets/figures/paviac-x2-band-psnr.png)

Source: `PaviaC_2_20_psnr.pdf`. Horizontal axis: spectral-band index. Vertical axis: PSNR in dB.

## PaviaU ×2: band-wise PSNR

![PaviaU times-two PSNR by spectral band](../assets/figures/paviau-x2-band-psnr.png)

Source: `PaviaU_2_20_psnr.pdf`. Horizontal axis: spectral-band index. Vertical axis: PSNR in dB.
