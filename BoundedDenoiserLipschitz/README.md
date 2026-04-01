# Bounded Denoiser with Lipschitz Convergence and Jacobian Regularization



**Stabilizing Plug-and-Play Reconstruction with Bounded Denoiser with Lipschitz Convergence and Jacobian Regularization**
**Period:** March 2026
**Tech:** Python, jax, Unet, Diffusion, ViT

## Overview

The methodology of bounded denoiser will be experimented with Lipschitz Convergence and Jacobian Regularization.
(Jacobian Regularization → Lipschitz Continuity → Convergence)
This experiment studies that maintaining a Lipschitz constant L≤1 prevents the catastrophic error propagation typically observed in multi-iteration PnP frameworks.
the major goal of this experiment is to stabilize inverse problem pipeline in Vision models as effectively suppressing accumulation of artifacts when iterations increase at small -$\sigma$

**Core idea:**

> Use Lipschitz Convergence and Jacobian Regularization to ensure stabilization of diminishing-$\sigma$ and provide fixed-point convergence guarantees even in nonconvex and ill-posed inverse problems


## Key Contributions

1. **Investigation of bounded denoiser methodologies**

   * Work research and analysis of the specific methodologies regarding bounded denoiser's role and its ongoing experiments in Plug-and-Play optimization and in iterative inverse problems

2. **UNet, Diffusion, Vit bridging solutions**

   * Comparison among different models for examining the optimal x* and its applications with emphasis on stability and reconstruction quality.

3. **Continuation schedule integration**

   * A continuation-based update schedule into the experimental pipelines, enabling systematic control of the optimization trajectory and stability under diminishing-noise or adaptive-penalty settings.

4. **Unifying experimental framework**

   * A consistent experimental framework for evaluating denoiser regularization, model architecture, and continuation schedules.

---

## Repository Structure

```text
.
├── README.md
├── theorem.tex
├── pyproject.toml
├── src/
│   ├── models/
│   │   ├── unet2d.py                
│   │   ├── diffusion.py           
│   │   └── vit.py         
│   ├── denoiser/
│   │   └── SN_wrappers.py
│   ├── operators/
│   ├── pnp/
│   │   ├── ADMM_loop.py         
│   │   └── logging.py
│   ├── data/         
│   │   └── split_utils.py
│   └── utils/
├── notebooks/
│   ├── 00_env_check.ipynb
│   ├── 01_train_denoiser_realsn_jacreg.ipynb
│   └── 02_ablation_grid.ipynb
└── outputs/
    ├── checkpoints/
    └── results/
```

---


## Metrics & Evaluation

### Metrics

**Investigation of bounded denoiser methodologies**

   * PSNR / SSIM

**Investigation of bounded denoiser methodologies**

   * final primal residual $r^k$
   * Number of iterations to reach tolerance
   * Divergence / oscillation rate

**Stability Proxies**

   *  Mean Jacobian regularization term
   *  Per-layer spectral norm estimates
   *  Maximum / mean spectral norm across layers

---

## Results (To be filled)

* Improved convergence stability under diminishing-$\sigma$ schedules
* Accuracy degradation: **$\leq$ [N]\%**
* Improved residual consistency: $|x^k - z^k|_2 \rightarrow 0$
* 

---

## Safety Notes (Medical Use)

This repository is **research-only**. Outputs from generative models can be misleading even when visually plausible.
Always evaluate with diagnosis-aware metrics and boundedness checks before any clinical interpretation.

---

## Citation

If you build on this work, cite or contact:

* **[Author]**, jeon.isavelle@gmail.com

