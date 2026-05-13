"""
High-quality separable image resize.  Ported from the original
paper
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from math import pi
from scipy.ndimage import shift as ndimage_shift
from torchvision.transforms.functional import rgb_to_grayscale

from third_party.ConSinGAN.config import GANConfig



def _norm(t: torch.Tensor) -> torch.Tensor:
    """[0, 1] → [-1, 1]"""
    return t * 2.0 - 1.0


def _denorm(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]"""
    return (t + 1.0) / 2.0


def _torch2uint8(x: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) tensor in [-1,1] → (H, W, C) uint8 array."""
    x = x[0].permute(1, 2, 0)
    x = _denorm(x).mul(255).clamp(0, 255).cpu().numpy()
    return x.astype(np.uint8)


def _np2torch(x: np.ndarray, config: GANConfig) -> torch.Tensor:
    """(H, W, C) uint8 → (1, C, H, W) float tensor in [-1, 1]."""
    if config.channels == 3:
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    else:
        if x.ndim == 3:
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
            t = rgb_to_grayscale(t.squeeze(0)).unsqueeze(0)
        else:
            t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().div(255.0)

    device = getattr(config, "_torch_device", torch.device("cpu"))
    return _norm(t.to(device))


def _imresize(im: torch.Tensor, scale: float, config: GANConfig) -> torch.Tensor:
    arr = _torch2uint8(im)
    arr = imresize_in(arr, scale_factor=scale)
    return _np2torch(arr, config)


def imresize_to_shape(
    im: torch.Tensor,
    output_shape: tuple[int, int],
    config: GANConfig,
) -> torch.Tensor:
    arr = _torch2uint8(im)
    arr = imresize_in(arr, output_shape=output_shape)
    return _np2torch(arr, config)



def imresize_in(
    im: np.ndarray,
    scale_factor: float | list[float] | None = None,
    output_shape: tuple | list | None = None,
    kernel: str | np.ndarray | None = None,
    antialiasing: bool = True,
    kernel_shift_flag: bool = False,
) -> np.ndarray:
    scale_factor, output_shape = _fix_scale_and_size(im.shape, output_shape, scale_factor)

    if isinstance(kernel, np.ndarray) and scale_factor[0] <= 1:
        return _numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)

    _kernel_map = {
        "cubic":    (cubic,    4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box":      (box,      1.0),
        "linear":   (linear,   2.0),
        None:       (cubic,    4.0),
    }
    if kernel not in _kernel_map:
        raise ValueError(f"Unknown kernel: {kernel!r}. Choose from {list(_kernel_map)}")

    method, kernel_width = _kernel_map[kernel]
    antialiasing = antialiasing and (scale_factor[0] < 1)

    out_im = np.copy(im)
    for dim in np.argsort(scale_factor).tolist():
        if scale_factor[dim] == 1.0:
            continue
        weights, field_of_view = _contributions(
            im.shape[dim], output_shape[dim], scale_factor[dim],
            method, kernel_width, antialiasing,
        )
        out_im = _resize_along_dim(out_im, dim, weights, field_of_view)

    return out_im


def _fix_scale_and_size(
    input_shape: tuple,
    output_shape: tuple | None,
    scale_factor: float | list | None,
) -> tuple[list[float], list[int]]:
    if scale_factor is not None:
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]
        scale_factor = list(scale_factor)
        scale_factor += [1] * (len(input_shape) - len(scale_factor))

    if output_shape is not None:
        # FIX: np.uint removed — use int() conversion
        output_shape = [int(v) for v in output_shape] + list(input_shape[len(output_shape):])

    if scale_factor is None:
        scale_factor = list(1.0 * np.array(output_shape) / np.array(input_shape))

    if output_shape is None:
        output_shape = [int(v) for v in np.ceil(np.array(input_shape) * np.array(scale_factor))]

    return scale_factor, output_shape


def _contributions(
    in_length: int,
    out_length: int,
    scale: float,
    kernel,
    kernel_width: float,
    antialiasing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_kernel = (lambda x: scale * kernel(scale * x)) if antialiasing else kernel
    if antialiasing:
        kernel_width /= scale

    out_coords = np.arange(1, out_length + 1)
    match_coords = out_coords / scale + 0.5 * (1 - 1.0 / scale)
    left_boundary = np.floor(match_coords - kernel_width / 2)
    expanded_width = int(np.ceil(kernel_width)) + 2

    # FIX: np.uint(...) → .astype(np.intp)
    field_of_view = np.squeeze(
        (np.expand_dims(left_boundary, 1) + np.arange(expanded_width) - 1).astype(np.intp)
    )

    weights = fixed_kernel(
        np.expand_dims(match_coords, 1) - field_of_view.astype(float) - 1
    )

    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights /= row_sums

    # FIX: np.uint(...) → .astype(np.intp)
    mirror = np.concatenate(
        [np.arange(in_length), np.arange(in_length - 1, -1, -1)]
    ).astype(np.intp)
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    non_zero_cols = np.nonzero(np.any(weights, axis=0))[0]
    weights      = np.squeeze(weights[:, non_zero_cols])
    field_of_view = np.squeeze(field_of_view[:, non_zero_cols])

    return weights, field_of_view


def _resize_along_dim(
    im: np.ndarray,
    dim: int,
    weights: np.ndarray,
    field_of_view: np.ndarray,
) -> np.ndarray:
    tmp = np.swapaxes(im, dim, 0)
    w   = np.reshape(weights.T, list(weights.T.shape) + [1] * (im.ndim - 1))
    out = np.sum(tmp[field_of_view.T] * w, axis=0)
    return np.swapaxes(out, dim, 0)


def _numeric_kernel(
    im: np.ndarray,
    kernel: np.ndarray,
    scale_factor: list[float],
    output_shape: list[int],
    kernel_shift_flag: bool,
) -> np.ndarray:
    from scipy.ndimage import correlate as ndimage_correlate

    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    out = np.zeros_like(im)
    for c in range(im.shape[2]):
        out[:, :, c] = ndimage_correlate(im[:, :, c], kernel)

    row_idx = np.round(
        np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])
    ).astype(int)
    col_idx = np.round(
        np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])
    ).astype(int)

    return out[row_idx[:, None], col_idx, :]


def kernel_shift(kernel: np.ndarray, sf: float | list[float]) -> np.ndarray:
    if np.isscalar(sf):
        sf = [sf, sf]
    from scipy.ndimage import measurements
    current_com = measurements.center_of_mass(kernel)
    wanted_com  = np.array(kernel.shape) / 2 + 0.5 * (np.array(sf) - (kernel.shape[0] % 2))
    shift_vec   = wanted_com - current_com
    pad         = int(np.ceil(np.max(np.abs(shift_vec)))) + 1
    kernel      = np.pad(kernel, pad, mode="constant")
    return ndimage_shift(kernel, shift_vec)



def cubic(x: np.ndarray) -> np.ndarray:
    absx  = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1) + (
        -0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0
    ) * ((absx > 1) & (absx <= 2))


def lanczos2(x: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).eps
    return (
        (np.sin(pi * x) * np.sin(pi * x / 2) + eps) / ((pi ** 2 * x ** 2 / 2) + eps)
    ) * (np.abs(x) < 2)


def lanczos3(x: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).eps
    return (
        (np.sin(pi * x) * np.sin(pi * x / 3) + eps) / ((pi ** 2 * x ** 2 / 3) + eps)
    ) * (np.abs(x) < 3)


def box(x: np.ndarray) -> np.ndarray:
    return ((-0.5 <= x) & (x < 0.5)).astype(float)


def linear(x: np.ndarray) -> np.ndarray:
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))