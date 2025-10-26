# Architecture & Design

This page provides an in-depth look at this repository's architecture and design principles.

## Overview

This repository is built on the [stochastic interpolant framework](https://arxiv.org/abs/2303.08797), a unified mathematical formulation that encompasses flow matching, consistency models, and various distillation methods.

## Stochastic Interpolant

The stochastic interpolant framework provides a unified view of generative modeling:

```
a_t = α(t) * a_0 + β(t) * a_1
```

Where:
- `a_t` interpolates between action at time `t=0` and action at time `t=1`
- `α(t)`, `β(t)` are interpolation schedules (linear or trigonometric)
- The goal is to learn a velocity field `v(a, t)` that transforms action at time `t=0` to action at time `t=1`
  - `v(a, t) = d/dt a(t)`

## Algorithms Overview

### **Flow Matching** (`flow`)

https://arxiv.org/abs/2210.02747

Standard flow matching loss.

**Train:**

```
b_θ ≈ argmin_θ 𝔼[‖b_t(I_t | o) - İ_t‖²],  where t ~ Unif([0,1]), z ~ N(0, I)
```

**Inference:**

```
d/dt a_t = b_t(a_t | o)  with initial condition  a_0 = z
```

### **Regression** (`regression`)

Direct regression loss, learns conditional action mean.

**Train:**

```
π_θ ≈ argmin_θ 𝔼[‖π_θ(o, I_0, t=0) - a‖²]
```

**Inference:**

```
â ← π_θ(o, z, t=0)
```

### **Two Step Denoising** (`tsd`)

Two step denoising, a simplified version of flow matching.

**Train:**

```
π_θ ≈ argmin_θ 𝔼[‖π_θ(o, I_0, t=0) - I_fix‖² + ‖π_θ(o, I_fix, t_fix) - a‖²]
```

**Inference:**

```
â_0 ← π_θ(o, z, 0)
â ← π_θ(o, t_fix * â_0 + (1 - t_fix) * z, t_fix)
```

### **Minimum Iterative Policy** (`mip`)

Minimum Iterative Policy, optimized for two-step sampling by removing redundant stochasticity in input.

**Train:**

```
π_mip^θ ≈ argmin_θ 𝔼[‖π_θ(o, I_0 = 0, t=0) - a‖² + ‖π_θ(o, I_t_fix, t_fix) - a‖²]
```

**Inference:**

```
â_mip^0 ← π_mip^θ(o, 0, t=0)
â_mip ← π_mip^θ(o, t_fix * â_mip^0, t_fix)
```

### **Consistency Trajectory Model** (`ctm`)

https://arxiv.org/abs/2310.02279

Consistency Trajectory Model, progressively distills multi-step flow into flow map/shortcut models.

**Train:**

```
Φ_{s,t}(I_s) ≈ Φ_{s+dt, t}(stopgrad(Φ_{s, s+dt}(I_s)))
```

**Inference:**

```
a = Φ_{0,1}(z),  where z ~ N(0, I)
```

### **Progressive Self-Distillation** (`psd`)

https://arxiv.org/pdf/2505.18825

Progressive Self-Distillation, a self-distillation framework which trains a flow map.

**Train:**

```
L_SD = L_b + L_D
L_b = ‖b_t(I_t) - İ_t‖²
L_D = ‖Φ_{s,t}(I_s) - Φ_{u,t}(Φ_{s,u}(I_s))‖²
```

**Inference:**

```
â = Φ_{0,1}(z),  where z ~ N(0, I)
```

### **Lagrangian Self-Distillation** (`lsd`)

https://arxiv.org/abs/2505.18825

Lagrangian Self-Distillation, a self-distillation framework which trains a flow map.

**Train:**

```
L_SD = L_b + L_D
L_b = ‖b_t(I_t) - İ_t‖²
L_D = ‖∂_t Φ_{s,t}(I_s) - b_t(Φ_{s,t}(I_s))‖²
```

**Inference:**

```
â = Φ_{0,1}(z),  where z ~ N(0, I)
```

### **Euler Self-Distillation** (`esd`)

https://arxiv.org/abs/2505.18825

Euler Self-Distillation, a self-distillation framework which trains a flow map.

**Train:**

```
L_SD = L_b + L_D
L_b = ‖b_t(I_t) - İ_t‖²
L_D = ‖∂_s Φ_{s,t}(I_s) + ∇ Φ_{s,t}(I_s) · b_s(I_s)‖²
```

**Inference:**

```
â = Φ_{0,1}(z),  where z ~ N(0, I)
```

### **Mean Flow** (`mf`)

https://arxiv.org/abs/2505.13447

Mean Flow, a self-distillation framework which trains a flow map.

**Define:**

```
Φ_{s,t}(I_s) = I_s + (t - s) * v̄_{s,t}(I_s)
```

**Train:**

```
L_SD = L_b + L_D
L_b = ‖b_t(I_t) - İ_t‖²
L_D = ‖∂_s Φ_{s,t}(I_s) + stopgrad(∇ Φ_{s,t}(I_s) · İ_s)‖²
```

**Inference:**

```
â = Φ_{0,1}(z),  where z ~ N(0, I)
```
