# Objective
Since 2022 an interest for learning generative models from a single image, as oppossed to from a large dataset has been grown. This service has the task of generating images when a __new defect__ has been detected, to simply to augment the training data for YOLO.

__Concurrent Single GAN__ or ConSinGAN is an architecture that instead of producing images produce feature maps at each stage, training multiple stages concurrently, they modify the pyramid of generators $\{G_0,\dots,G_N\}$ that SinGAN used adapting the rescaling to not be strictly geometric.  


Reference [Improved techniques for training simple-image GANS](https://www.tobiashinz.com/2020/03/24/improved-techniques-for-training-single-image-gans.html).

# ConSinGAN `functions.py` - Math Reference


## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Normalisation - `norm / denorm`](#2-normalisation--norm--denorm)
3. [Noise Generation - `generate_noise / sample_random_noise`](#3-noise-generation--generate_noise--sample_random_noise)
4. [Bilinear Upsampling - `upsampling`](#4-bilinear-upsampling--upsampling)
5. [WGAN-GP Gradient Penalty - `calc_gradient_penalty`](#5-wgan-gp-gradient-penalty--calc_gradient_penalty)
6. [Scale Pyramid - `adjust_scales2image / create_reals_pyramid`](#6-scale-pyramid--adjust_scales2image--create_reals_pyramid)
7. [Mask Dilation - `dilate_mask`](#7-mask-dilation--dilate_mask)
8. [GIF Animation - `generate_gif`](#8-gif-animation--generate_gif)
9. [Augmentation - `Augment / shuffle_grid`](#9-augmentation--augment--shuffle_grid)
10. [Checkpoint I/O](#10-checkpoint-io)
11. [Quick Reference](#11-quick-reference)


## 1  Module Overview

This module is the **utility backbone** of ConSinGAN - a single-image GAN that learns an internal patch distribution from one image and uses it to synthesise new samples, perform harmonisation, editing, and animation.

Every mathematical primitive lives here:

- Normalisation and denormalisation
- Noise sampling and pyramid construction
- WGAN gradient penalty
- Multi-scale image pyramid
- Noise-space animation
- Mask dilation and augmentation


## 2  Normalisation - `norm / denorm`

### 2.1  Forward: `norm`

Images loaded from disk have pixel values in **[0, 1]**. Neural network weights are initialised near zero, so inputs in [0, 1] create a systematic positive bias. Mapping to **[−1, 1]** centres the data, which:

- Matches the output range of `tanh` activations used in many generators
- Keeps gradients well-scaled throughout training
- Allows BatchNorm layers to operate correctly

$$\text{norm}(x) = (x - 0.5) \times 2$$

The constant `0.5` is the midpoint of [0, 1]. Subtracting it centres the range; multiplying by 2 stretches it to [−1, 1].

### 2.2  Inverse: `denorm`

When saving or displaying a tensor the transformation is reversed:

$$\text{denorm}(x) = \frac{x + 1}{2}$$

This maps [−1, 1] back to [0, 1]. A clamp ensures no out-of-range artefacts from network outputs.


## 3  Noise Generation - `generate_noise / sample_random_noise`

### 3.1  Why noise drives a GAN

In ConSinGAN the generator **G** maps a noise tensor **z** to an image patch. Unlike class-conditional GANs that use a single latent vector, ConSinGAN injects **spatially-structured noise** so that G can be applied convolutionally at each pyramid scale.

### 3.2  Gaussian noise with spatial downscaling

The default noise is sampled at a coarser resolution and then upsampled:

```
z ~ N(0, I),   shape (B, C, H/s, W/s)
z_up = Upsample(z, H, W)
```

This spatial downscaling (controlled by `scale`) introduces low-frequency correlations. Higher values of `scale` give smoother, more globally-coherent noise - helping the generator produce coherent large-scale structure at finer pyramid levels.

### 3.3  Noise pyramid - `sample_random_noise`

One noise tensor is produced per pyramid level:

| Level | Channels | Spatial size |
|-------|----------|--------------|
| 0 (coarsest) | `config.channels` C | `reals_shapes[0][2:4]` |
| 1 … D | `config.filters_per_conv` F | `reals_shapes[d][2:4] + num_layers × 2` |

The padding on levels 1…D compensates for the spatial reduction caused by valid convolutions inside the generator - ensuring the output of each level matches the target resolution.

---

## 4  Bilinear Upsampling - `upsampling`

Bilinear interpolation computes each output pixel as a **weighted average** of its four nearest input neighbours:

$$
z(x,y) = (1−\alpha)(1−\beta)·in(x₀,y₀) + \alpha(1−\beta)·in(x₁,y₀) \\ 
         + (1−\alpha)\beta·in(x₀,y₁)     + \alpha\beta·in(x₁,y₁)
$$

where $\alpha, \beta \in [0, 1]$ are the fractional distances to the nearest grid points.

**Why `align_corners=True`?**  
This option maps the corner pixels of input and output to exactly the same coordinates, preserving scale consistency across the pyramid. It matches the behaviour of the original `nn.Upsample` call in the 2021 codebase.


## 5  WGAN-GP Gradient Penalty - `calc_gradient_penalty`

### 5.1  Wasserstein distance

Standard GAN training minimises the Jensen–Shannon divergence (Prince et. al 2025) between real and generated distributions. This divergence **saturates** when the supports are disjoint (common early in training), causing vanishing gradients.

Wasserstein GAN instead minimises the **Earth-Mover distance**:

$$
W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \; \mathbb{E}[f(x)] - \mathbb{E}[f(G(z))]
$$

where the supremum is over 1-Lipschitz functions *f*. The discriminator D approximates this optimal *f*, so it must be constrained to be 1-Lipschitz.

### 5.2  Gradient penalty (Gulrajani et al., 2017)

Rather than weight clipping, WGAN-GP enforces the Lipschitz constraint by penalising deviations of the gradient norm from 1 along straight lines between real and fake samples:

$$
\hat{x}  =  \varepsilon \cdot x_{real} + (1 − \varepsilon) \cdot x_{fake}, \quad   \varepsilon \sim Unif[0, 1] \\[3mm] 
GP  =  \lambda · E[ ( \|\nabla_{\hat{x}} D(\hat{x})\|_2 − 1 )² ]
$$

The full discriminator loss is:

$$
L_D  =  E[D(G(z))] − E[D(x)] + GP
$$

A gradient norm greater than 1 means D is changing too fast (violating the Lipschitz condition). The penalty pushes it back toward 1. $\lambda$ is typically between 0.1 and 10.

### 5.3  Implementation detail

`torch.autograd.grad` computes the exact gradient through the discriminator forward pass. The norm is taken over the spatial + channel dimensions (`dim=1`), then squared and averaged over the batch.


## 6  Scale Pyramid - `adjust_scales2image / create_reals_pyramid`

### 6.1  Why a pyramid?

ConSinGAN trains a **hierarchy of generators**, each responsible for one frequency band. Coarser levels capture global structure (colour, large shapes); finer levels add texture detail. Training proceeds from coarsest to finest, with each level fine-tuning the output of the previous one.

### 6.2  Global scale - `adjust_scales2image`

(A bit confusing for me). The input is first rescaled so its longest dimension equals `max_size`:

$$
s_1  =  \min(\max_{\text{size}} / \max(H, W),  1)
$$

The per-level scale factor **r** is then chosen so the coarsest level has a shortest side of `min_size` pixels:

$$
r  =  (\min_{\text{size}} / \min(H_1, W_1)) ^ {(1 / \text{stop scale})}
$$

where H₁, W₁ are the dimensions after the global rescale.

### 6.3  Level spacing - `create_reals_pyramid`

For all modes except harmonisation, levels are spaced **log-linearly** so that fine scales are not over-represented:

$$
\text{scale}_i  =  r ^ {( ((S−1) / \ln S) \cdot \ln(S−i) + 1 )}
$$

where S = `stop_scale`. The exponent is a log-linear interpolation from 0 (i=0, coarsest) to S−1 (i=S−1, finest). The `+1` ensures the scale reaches exactly 1 at the finest level (full resolution).

For **harmonisation**, a simple geometric progression is used instead:

$$
\text{scale}_i  =  r ^ {(S − i)}
$$

## 7  Mask Dilation - `dilate_mask`

In harmonisation and editing tasks the user provides a binary mask indicating the modified region. Dilating the mask gives the generator a margin around the pasted area so it can blend edges smoothly.

### 7.1  Morphological dilation via max-pooling

A max-pool with kernel size `2r+1` and padding `r` computes the maximum value in a `(2r+1)×(2r+1)` neighbourhood at each position. For a binary mask this is equivalent to dilation with a square structuring element of radius r:

$$
\text{dilated}(x,y)  =  max_{|\Delta x|\le r, |\Delta y| \le r}  \text{mask}(x+\Delta x, y+ \Delta y)
$$

This approximates a disk kernel, is O(HW), and is hardware-accelerated via cuDNN.

| Mode | Radius |
|------|--------|
| harmonization | 7 px |
| editing | 20 px |

### 7.2  Gaussian feathering

A Gaussian blur (kernel=11, σ=5) replaces the hard binary boundary with a smooth soft mask, producing a feather zone of ~10 pixels so the generator can blend seamlessly.


## 8  GIF Animation (not actually needed for this project) - `generate_gif`

### 8.1  Goal

Rather than sampling **independent** noise frames (which produce flickering), the animation walks smoothly through noise space so consecutive frames are visually similar.

### 8.2  EMA momentum walk

$$
\text{diff}_t  =  \beta · (z_{t-1} − z_{t-2}) + (1−\beta) · \varepsilon_t \\[3mm]
z_t     =  \alpha · z_{fix} + (1−\alpha) · (z_{t-1} + \text{diff}_t)
$$


| Parameter | Role |
|-----------|------|
| $\beta$  | Momentum - higher → slower, smoother motion |
| $\alpha$  | Anchor strength - higher → less drift from reference |
| $start_scale$ | Locks coarse levels to $z_{fix}$; only fine-scale texture animates |

$\text{diff}_t$ is an exponential moving average of noise increments. It gives the walk inertia - the direction from the previous step is partially carried forward, preventing sharp jumps.

$z_t$ is attracted back to the fixed anchor $z_{fix}$ with strength $\alpha$, preventing unbounded drift and keeping the animation close to a reference image.

---

## 9  Augmentation - `Augment / shuffle_grid`

### 9.1  `Augment` class

A new random pipeline is built on **every call**, so augmentation strength and patch count vary per sample. The pipeline applies one colour/noise transform followed by random erasing:

| Transform | Effect |
|-----------|--------|
| `GaussianNoise` | Additive white noise - regularises the discriminator |
| `RandomInvert` | Flips pixel values - forces invariance to global brightness |
| `ColorJitter` | Random hue / saturation / brightness shifts |
| `RandomErasing` | Pastes random-coloured rectangles - simulates occlusion |

RandomErasing is applied with probability 0.9, using 1 or 2 patches scaled to 2–20% of the image area.

### 9.2  `shuffle_grid` - spatial self-supervision

Rectangular patches are cut and pasted at a slightly displaced position within the same image. This creates local inconsistencies (boundary artefacts, texture mismatches) that the discriminator learns to detect, encouraging the generator to produce globally coherent outputs.

Tile size and translation magnitude are **scaled down as tile count increases**, so the total disturbed area stays roughly constant across all configurations.


## 10  Checkpoint I/O

| Function | Purpose |
|----------|---------|
| `save_networks` | Saves `netG.pth`, `netD.pth` (or `netD_i.pth`), `z_opt.pth` |
| `load_trained_model` | Restores `Gs`, `Zs`, `reals`, `NoiseAmp` |
| `generate_dir2save` | Builds a unique path encoding all hyper-parameters |
| `post_config` | Computes derived fields: device, timestamp, seed |
| `load_config` | Reads `parameters.txt` and overrides config fields |

`generate_dir2save` encodes training mode, depth, learning rate scale, batch norm, activation, and timestamp so **different runs never overwrite each other**.


## 11  Quick Reference

| Function | Category | Key idea |
|----------|----------|----------|
| `norm` / `denorm` | Normalisation | `[0,1] <-> [−1,1]`  via  `(x−0.5)×2` |
| `generate_noise` | Noise | Gaussian z at reduced resolution, upsampled |
| `sample_random_noise` | Noise | Noise pyramid, one tensor per scale level |
| `upsampling` | Interpolation | Bilinear with `align_corners=True` |
| `calc_gradient_penalty` | WGAN-GP |  Enforces the Lipschitz constraint |
| `adjust_scales2image` | Pyramid | Computes global & per-level scale factors |
| `create_reals_pyramid` | Pyramid | Log-linear level spacing |
| `dilate_mask` | Mask | Max-pool dilation + Gaussian feathering |
| `Augment` | Augmentation | GaussianNoise / Invert / ColorJitter + Erasing |
| `shuffle_grid` | Augmentation | Patch displacement for spatial self-supervision |
| `save_networks` | Checkpoint | Saves G, D(s), z_opt to disk |
| `load_trained_model` | Checkpoint | Restores Gs, Zs, reals, NoiseAmp |
| `generate_gif` | Animation | EMA momentum walk in noise space |
