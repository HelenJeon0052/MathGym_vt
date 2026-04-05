# Structure-Aware Split-Flow ODE Solver for ViTs

A structure-aware split-flow ODE formulation for Vision Transformers (ViTs), designed for efficient and stable computation in 3D medical image segmentation.

## Overview

Vision Transformers (ViTs) are effective for modeling global context. However, in 3D volumetric data, the number of tokens grows rapidly, which leads to substantial computational cost. This project explores a continuous-depth interpretation of transformer blocks and proposes a **structure-aware split-flow ODE solver** that reflects the internal structure of a transformer more explicitly than a black-box ODE formulation.

Instead of modeling the transformer dynamics with a single monolithic vector field, the proposed method decomposes the dynamics into:

- **attention flow**: token-mixing dynamics
- **MLP flow**: token-wise nonlinear reaction dynamics
- **friction / retention flow**: contraction-like stabilization dynamics

These subflows are recombined through operator splitting, with optional adaptive step-size control at inference time.

---

## Motivation

In 3D medical image segmentation, models must capture both:

- **long-range contextual information**
- **fine local spatial structure**

Standard transformer blocks are powerful, but their cost becomes heavy in volumetric settings. This project is motivated by the idea that computation should be **structured** and, eventually, **adaptive**:

- smoother latent dynamics should require less computation,
- more complex dynamics should receive more numerical refinement.

The goal is not merely to say “use an ODE,” but to design a solver that is consistent with the internal organization of the transformer block.

---

## Main Idea

Let the token state be

\[
u(t) \in \mathbb{R}^{T \times d},
\]

where \(T\) is the number of tokens and \(d\) is the embedding dimension.

Instead of defining a single vector field

\[
u' = f(u,t),
\]

we decompose the dynamics as

\[
\frac{du}{dt} = f_{\mathrm{attn}}(u,t) + f_{\mathrm{mlp}}(u,t) + f_{\mathrm{fric}}(u).
\]

### Interpretation of Each Component

- \(f_{\mathrm{attn}}\): attention-induced token mixing
- \(f_{\mathrm{mlp}}\): token-wise nonlinear transformation
- \(f_{\mathrm{fric}}\): retention / damping term for stabilizing the evolution

We use the retention–forcing form

\[
u' + P(u)u = Q(u),
\]

which can be rewritten as

\[
u' = -P(u)u + Q(u).
\]

Then the forcing term is further decomposed into

\[
Q(u) = Q_{\mathrm{attn}}(u) + Q_{\mathrm{mlp}}(u),
\]

so that

\[
f_{\mathrm{fric}}(u) = -P(u)u, \qquad
f_{\mathrm{attn}}(u,t) = Q_{\mathrm{attn}}(u,t), \qquad
f_{\mathrm{mlp}}(u,t) = Q_{\mathrm{mlp}}(u,t).
\]

---

## Split-Flow Solver

Each sub-vector field induces a flow map over step size \(h\):

\[
\phi_{\mathrm{attn}}^h,\qquad
\phi_{\mathrm{mlp}}^h,\qquad
\phi_{\mathrm{fric}}^h.
\]

### First-Order Lie Splitting

\[
\Phi_h^{\mathrm{Lie}}
\approx
\phi_{\mathrm{mlp}}^h
\circ
\phi_{\mathrm{attn}}^h
\circ
\phi_{\mathrm{fric}}^h.
\]

### Symmetric Second-Order Splitting

The main solver considered here is a symmetric second-order composition:

\[
\Phi_h^{\mathrm{Strang}}
\approx
\phi_{\mathrm{fric}}^{h/2}
\circ
\phi_{\mathrm{attn}}^{h/2}
\circ
\phi_{\mathrm{mlp}}^{h}
\circ
\phi_{\mathrm{attn}}^{h/2}
\circ
\phi_{\mathrm{fric}}^{h/2}.
\]

Equivalently, with

\[
A := f_{\mathrm{fric}},\qquad
B := f_{\mathrm{attn}},\qquad
C := f_{\mathrm{mlp}},
\]

we write

\[
\Phi_h^{\mathrm{Strang}}
\approx
e^{\frac{h}{2}A}
e^{\frac{h}{2}B}
e^{hC}
e^{\frac{h}{2}B}
e^{\frac{h}{2}A}.
\]

This is a **symmetric second-order splitting adapted to the three-way decomposition**.

---

## Friction / Retention Flow

The friction term is intended to stabilize the latent dynamics.

If \(P(u)\) is frozen over a substep, then

\[
u' = -\widehat{P}u
\]

admits the closed-form update

\[
u_{n+1} = \exp(-h\widehat{P})u_n.
\]

When \(\widehat{P}\) is diagonal or channel-wise, this update is cheap and explicitly damping.

To limit drift and uncontrolled growth, bounded gates are used for the coefficients in \(P\) and for the forcing terms.

---

## Adaptive Step-Size Control

To support adaptive computation, a split-pair local error surrogate is used:

\[
u_{n+1}^{(1)} = \Phi_h^{\mathrm{Lie}}(u_n), \qquad
u_{n+1}^{(2)} = \Phi_h^{\mathrm{Strang}}(u_n),
\]

\[
e_n =
\frac{\lVert u_{n+1}^{(2)} - u_{n+1}^{(1)} \rVert}
{\mathrm{atol} + \mathrm{rtol}\,\lVert u_{n+1}^{(2)} \rVert}.
\]

- If \(e_n \le 1\), the step is accepted.
- Otherwise, the step is rejected and recomputed with a smaller step size.

In the current formulation, this adaptivity is mainly expressed at the **global latent-state level**, rather than strictly local spatial regions.

---

## Experimental Design

### Exp-A: Discrete ViT Baseline
A standard residual transformer block is used as the baseline on a ViT-style 3D segmentation backbone.

- no ODE formulation
- no splitting
- no adaptive control

This serves as the reference point for segmentation quality and computational cost.

### Exp-B: Monolithic ODE Block
A continuous-depth block is inserted at the same position:
\[
u' = f_{\mathrm{mono}}(u,t), \qquad
f_{\mathrm{mono}}(u,t) = f_{\mathrm{attn}}(u,t) + f_{\mathrm{mlp}}(u,t).
\]

- attention and MLP are treated together
- fixed-step Euler or RK-based integration
- no splitting
- no adaptive control

This tests whether **continuous-depth modeling alone** is sufficient.

### Exp-C: Split-Flow Solver, Fixed-Step

The main experiment uses the structured decomposition
\[
u' = f_{\mathrm{attn}}(u,t) + f_{\mathrm{mlp}}(u,t) + f_{\mathrm{fric}}(u),
\]
with:

- Lie or symmetric second-order splitting
- no adaptive controller
- fixed step counts such as \(1,2,3\)

This isolates the effect of **structured splitting** from continuous-depth modeling itself.

### Exp-D: Split-Flow Solver, Adaptive
An optional extension of Exp-C introduces adaptive step-size control.

- structured decomposition
- split-pair error estimation
- adaptive step acceptance / rejection

This is intended to evaluate whether computation can be allocated more efficiently during inference.


---

## Repository Tree
```text
SplitFlowODESolver/
├─ README.md
├─ pyproject.toml
├─ .gitignore
│
├─ configs/
│  ├─ brats_train.yaml
│  ├─ ablation_mixmode_brats.yaml
│  └─ model_light3dvit.yaml
│
├─ julia/
│  ├─ main.jl
│  ├─ inference.jl
│  ├─ save_nifti.jl
│  └─ ...
│
└─ src/
   └─ AnomalyDetectionVit/
      ├─ data/
      │  ├─ preprocess_brats.py
      │  └─ ...
      ├─ models/
      │  ├─ attention.py
      │  ├─ encoder.py
      │  ├─ vit_3d.py
      │  ├─ odevit.py
      │  ├─ splitting.py
      │  ├─ hybrid_*.py
      │  ├─ patching/
      │  │  └─ ...
      │  └─ ...
      ├─ solvers/
      │  ├─ 
      │  ├─ 
      │  └─ ...
      ├─ tools/
      │  ├─ export_onnx.py
      │  ├─ check_onnx.py
      │  └─ ...
      ├─ train/
      │  ├─ train.py
      │  ├─ train.yaml
      │  └─ ...
      ├─ outputs/
      │  └─ ...
      └─ ...
```

This tree should be interpreted as a working experimental layout, not a finalized production structure.
And it will be reorganized according to the ongoing process.

---

## Metrics


Segmentation Quality

- Dice
- voxel-level sensitivity / specificity
- optional Hausdorff distance

Triage

- AUROC
- AUPRC
- sensitivity at fixed specificity


Efficiency / Systems

- inference latency
- FLOPs or approximate compute
- memory scaling

---

## Ablations


- replace only the bottlenecked transformer module
- vary input volume size / window size / patch size
- compare memory scaling
- compare discrete, monolithic ODE, and split-flow ODE variants

---

## Expected Advantages

- **Interpretability**  
  Separates token mixing, token-wise reaction, and retention into explicit components.

- **Stability control**  
  Friction acts as a contraction-like stabilizing mechanism.

- **Adaptive computation**  
  Step-size control provides a principled way to vary computational effort.

- **Modularity**  
  Each subflow can be assigned a solver suited to its structure.

---

## Limitations

---

## Future Work

- Add SwinUNETR as a secondary transfer architecture
- Explore alternative symmetric second-order orderings for the three-operator decomposition
- Compare fixed-step and adaptive split solvers empirically

---

## Status

This repository is currently focused on:

- solver formulation
- implementation skeletons for monolithic and split-flow ODE blocks
- fixed-step comparison experiments
- preparation for adaptive inference experiments