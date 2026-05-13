"""
Public API used by train.py:
  get_device(config)
  read_image(path, config) → Tensor (1,C,H,W) in [-1,1]
  adjust_scales2image(real, config) → Tensor (rescaled)
  create_reals_pyramid(real, config) → list[Tensor]
  generate_noise(shape, device) → Tensor
  calc_gradient_penalty(D, real, fake, device, lambda_gp) → Tensor
  draw_concat(noise_list, noise_amp, generator, reals, scale_num, config, mode)
  save_image(tensor, path)
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from generator.utils.config import GANConfig
from generator.utils.imresize import _imresize, imresize_to_shape, _np2torch, _torch2uint8



def get_device(config: GANConfig) -> torch.device:
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config.device == "cuda":
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")
    return device



def read_image(path: str, config: GANConfig) -> torch.Tensor:
    """
    Load an image from disk and convert to a (1, C, H, W) tensor in [-1, 1].
    Handles both RGB (channels=3) and greyscale (channels=1).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _np2torch(img, config)


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a (1, C, H, W) tensor in [-1, 1] to disk as a PNG."""
    arr = _torch2uint8(tensor)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def adjust_scales2image(real: torch.Tensor, config: GANConfig) -> torch.Tensor:
    """
    Rescale the input image so its longest side is config.max_size,
    compute the pyramid scale factor, and set config.stop_scale.

    Modifies config in-place.  Returns the globally rescaled real.
    """
    h, w = real.shape[2], real.shape[3]

    # Global rescale so the longest side ≤ max_size
    global_scale = min(config.max_size / max(h, w), 1.0)
    config.scale1 = global_scale

    if global_scale < 1.0:
        new_h = int(round(h * global_scale))
        new_w = int(round(w * global_scale))
        real = imresize_to_shape(real, (new_h, new_w), config)

    # Pyramid scale factor so we reach min_size in train_stages steps
    min_side = min(real.shape[2], real.shape[3])
    config.scale_factor = (config.min_size / min_side) ** (1.0 / config.train_stages)
    config.stop_scale   = config.train_stages - 1

    return real


def create_reals_pyramid(real: torch.Tensor, config: GANConfig) -> list[torch.Tensor]:
    """
    Build the image pyramid from coarsest (index 0) to finest (index stop_scale).
    reals[0] is the smallest image; reals[-1] is the input real itself.
    """
    reals = []
    for i in range(config.stop_scale + 1):
        # scale = r^(stop_scale - i), i.e. 1.0 at i==stop_scale
        scale = config.scale_factor ** (config.stop_scale - i)
        if abs(scale - 1.0) < 1e-6:
            reals.append(real)
        else:
            reals.append(_imresize(real, scale, config))
    return reals



def generate_noise(
    shape: list[int] | tuple[int, ...],
    device: torch.device,
    num_samp: int = 1,
) -> torch.Tensor:
    """Return a Gaussian noise tensor of shape (num_samp, *shape)."""
    return torch.randn(num_samp, *shape, device=device)


def generate_spatial_noise(
    ref_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convenience: generate noise matching (1, C, H, W) of ref_tensor."""
    return generate_noise(
        list(ref_tensor.shape[1:]),
        device=device,
        num_samp=1,
    )


def draw_concat(
    fixed_noise: list[torch.Tensor],
    noise_amp: list[float],
    generator: nn.Module,
    reals: list[torch.Tensor],
    scale_num: int,
    config: GANConfig,
    device: torch.device,
    mode: str = "rand",
) -> list[torch.Tensor]:
    """
    Build the noise list fed into GrowingGenerator.forward at scale `scale_num`.

    mode='rand'    : fixed noise for scales 0..n-1, fresh random for scale n
    mode='rec'     : fixed noise at all scales (reconstruction branch)
    """
    noise_list: list[torch.Tensor] = []

    for i in range(scale_num + 1):
        if i == 0:
            # Coarsest scale always uses fixed reconstruction noise
            z = fixed_noise[0].to(device)
        elif mode == "rec":
            z = fixed_noise[i].to(device)
        else:
            # Fresh random noise for the current scale
            if i < scale_num:
                z = torch.zeros_like(reals[i])   # zeros for previous scales
            else:
                z = generate_spatial_noise(reals[i], device)
        noise_list.append(z)

    return noise_list


def calc_gradient_penalty(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 0.1,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty.  Interpolates between real and fake,
    computes D's gradient at the midpoint, and penalises ||grad|| ≠ 1.
    """
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = D(interpolated)

    grads = autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gp = lambda_gp * ((grads.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
    return gp


def sample_from_generator(
    generator: nn.Module,
    fixed_noise: list[torch.Tensor],
    noise_amp: list[float],
    reals: list[torch.Tensor],
    config: GANConfig,
    device: torch.device,
    n_samples: int = 1,
) -> list[torch.Tensor]:
    """
    Draw `n_samples` random images from a fully trained generator pyramid.
    Returns a list of (1, C, H, W) tensors in [-1, 1].
    """
    generator.eval()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            noise = draw_concat(
                fixed_noise, noise_amp, generator, reals,
                len(reals) - 1, config, device, mode="rand",
            )
            fake = generator(noise, [r.shape for r in reals], noise_amp)
            samples.append(fake.cpu())
    generator.train()
    return samples


def generate_dir2save(config: GANConfig) -> str:
    return config.outf