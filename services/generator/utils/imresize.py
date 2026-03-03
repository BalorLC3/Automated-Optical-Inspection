from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from math import pi
from scipy.ndimage import shift as ndimage_shift   
from torchvision.transforms.functional import (
    rgb_to_grayscale,

)

from services.generator.utils.config import GANConfig
from services.generator.utils.functions import (
    _norm,
    _denorm,
)

def _torch2uint8(x: torch.Tensor) -> np.ndarray:
    """
    Convert a single image tensor to a uint8 NumPy array.

    Takes the first sample (index 0) and maps [-1, 1] → [0, 255].

    Parameters
    ----------
    x : torch.Tensor  - shape ``(1, C, H, W)``

    Returns
    -------
    np.ndarray  - shape ``(H, W, C)``, dtype uint8
    """
    x = x[0].permute(1, 2, 0)          # (C, H, W) → (H, W, C)
    x = _denorm(x).mul(255).cpu().numpy()
    return x.astype(np.uint8)

def _np2torch(x: np.ndarray, config: GANConfig) -> torch.Tensor:
    """
    Convert a uint8 NumPy image to a normalised float tensor.

    Parameters
    ----------
    x      : np.ndarray  — shape ``(H, W, C)`` or ``(H, W)``, dtype uint8
    config : GANConfig   — must have ``config.channels`` (1 or 3)
                           and ``config.device``

    Returns
    -------
    torch.Tensor  — shape ``(1, C, H, W)``, values in [−1, 1]
    """
    if config.channels == 3:
        # (H, W, C) uint8 → (1, C, H, W) float [0, 1]
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    else:
        # Greyscale: accept both (H,W,3) and (H,W,1) / (H,W)
        if x.ndim == 3:
            # Convert RGB → grey using torchvision (no skimage)
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
            t = rgb_to_grayscale(t.squeeze(0)).unsqueeze(0)   # (1,1,H,W)
        else:
            t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().div(255.0)

    device = getattr(config, "device", torch.device("cpu"))
    t = t.to(device)
    return _norm(t)


def imresize(im: torch.Tensor, scale: float, config: GANConfig) -> torch.Tensor:
    """
    Resize a tensor image by a scalar scale factor.

    Converts to uint8 NumPy, applies the high-quality kernel-based resize,
    then converts back to a normalised tensor.

    Parameters
    ----------
    im     : torch.Tensor  — shape ``(1, C, H, W)``, values in [−1, 1]
    scale  : float         — scale factor (e.g. 0.5 for half-size)
    config : GANConfig

    Returns
    -------
    torch.Tensor  — resized image, same normalisation as input
    """
    arr = _torch2uint8(im)
    arr = imresize_in(arr, scale_factor=scale)
    return _np2torch(arr, config)


def imresize_to_shape(
    im: torch.Tensor,
    output_shape: tuple[int, int],
    config: GANConfig,
) -> torch.Tensor:
    """
    Resize a tensor image to an exact ``(H, W)`` output shape.

    Parameters
    ----------
    im           : torch.Tensor       — shape ``(1, C, H, W)``
    output_shape : (int, int)         — target ``(height, width)``
    config       : GANConfig

    Returns
    -------
    torch.Tensor  — resized image ``(1, C, H_out, W_out)``
    """
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
    """
    High-quality image resize using separable 1-D kernel filtering.

    The algorithm processes each spatial dimension independently
    (separable filtering), which is both faster and numerically cleaner
    than a 2-D convolution.  For each output position it computes a
    weighted sum of the input pixels whose *field of view* overlaps with
    the kernel support, then normalises the weights to 1.

    Parameters
    ----------
    im              : np.ndarray        — shape ``(H, W, C)``, dtype uint8
    scale_factor    : float or list     — uniform or per-dim scale
    output_shape    : tuple or list     — target ``(H, W)`` (alternative to scale)
    kernel          : str or np.ndarray — interpolation kernel name or custom array.
                      Supported names: "cubic" (default), "lanczos2", "lanczos3",
                      "box", "linear".
    antialiasing    : bool              — apply low-pass pre-filter when downscaling
    kernel_shift_flag : bool            — shift kernel centre of mass (see kernel_shift)

    Returns
    -------
    np.ndarray  — resized image, same dtype as input
    """
    scale_factor, output_shape = _fix_scale_and_size(im.shape, output_shape, scale_factor)

    # Numeric kernel path: direct correlation + subsampling (downscale only)
    if isinstance(kernel, np.ndarray) and scale_factor[0] <= 1:
        return _numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)

    # Select interpolation method
    _kernel_map = {
        "cubic":    (cubic,    4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box":      (box,      1.0),
        "linear":   (linear,   2.0),
        None:       (cubic,    4.0),   # default
    }
    if kernel not in _kernel_map:
        raise ValueError(f"Unknown kernel: {kernel!r}. Choose from {list(_kernel_map)}")

    method, kernel_width = _kernel_map[kernel]

    # Anti-aliasing stretches the kernel for downscaling to act as a low-pass filter
    antialiasing = antialiasing and (scale_factor[0] < 1)

    # Process dimensions from smallest scale to largest — most efficient order
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
    """
    Standardise scale_factor and output_shape so both are fully specified.

    Rules
    -----
    - Scalar scale_factor → broadcast to all spatial dims.
    - scale_factor shorter than input_shape → pad with 1.0.
    - Missing scale_factor → derived from output_shape / input_shape.
    - Missing output_shape → ceil(input_shape × scale_factor).

    Parameters
    ----------
    input_shape   : tuple  — shape of the input array
    output_shape  : tuple or None
    scale_factor  : float, list, or None

    Returns
    -------
    (scale_factor, output_shape)  — both as lists of the same length as input_shape
    """
    if scale_factor is not None:
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]
        scale_factor = list(scale_factor)
        # Pad remaining dims (e.g. channel) with 1
        scale_factor += [1] * (len(input_shape) - len(scale_factor))

    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    if scale_factor is None:
        scale_factor = list(1.0 * np.array(output_shape) / np.array(input_shape))

    if output_shape is None:
        output_shape = list(np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor))))

    return scale_factor, output_shape


def _contributions(
    in_length: int,
    out_length: int,
    scale: float,
    kernel,
    kernel_width: float,
    antialiasing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute interpolation weights and field-of-view for one spatial dimension.

    For each output pixel we find all input pixels within the kernel support
    and compute a normalised weight for each.

    Theory
    ------
    Output coordinate p_out maps to input coordinate:

        p_in  =  p_out / scale + 0.5 · (1 − 1/scale)

    This formula accounts for the fact that pixel centres (not pixel indices)
    must be aligned at the boundaries.  For scale < 1 (downscaling) the
    kernel is stretched by 1/scale and the weights are renormalised, acting
    as a low-pass anti-aliasing filter.

    Boundary handling uses **mirror padding**: indices outside [0, N−1] are
    reflected, which is equivalent to symmetric boundary conditions and
    avoids dark/bright border artefacts.

    Parameters
    ----------
    in_length    : int    — input dimension size
    out_length   : int    — output dimension size
    scale        : float  — resize scale for this dimension
    kernel       : callable — interpolation kernel function
    kernel_width : float  — support radius of the kernel
    antialiasing : bool   — stretch kernel for downscaling

    Returns
    -------
    weights      : np.ndarray  — shape ``(out_length, kernel_size)``
    field_of_view: np.ndarray  — shape ``(out_length, kernel_size)``, input indices
    """
    # Stretch kernel when antialiasing (downscale low-pass filter)
    fixed_kernel = (lambda x: scale * kernel(scale * x)) if antialiasing else kernel
    if antialiasing:
        kernel_width /= scale

    out_coords = np.arange(1, out_length + 1)

    # Map output pixel centres to input pixel centres
    match_coords = out_coords / scale + 0.5 * (1 - 1.0 / scale)

    # Left boundary of the kernel support window
    left_boundary = np.floor(match_coords - kernel_width / 2)

    # +2 ensures we never miss a partially-covered pixel at either edge
    expanded_width = int(np.ceil(kernel_width)) + 2

    field_of_view = np.squeeze(
        np.uint(np.expand_dims(left_boundary, 1) + np.arange(expanded_width) - 1)
    )

    weights = fixed_kernel(
        np.expand_dims(match_coords, 1) - field_of_view.astype(float) - 1
    )

    # Normalise so weights sum to 1 (preserves image brightness)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights /= row_sums

    # Mirror padding for boundary handling (reflection at both ends)
    mirror = np.uint(
        np.concatenate([np.arange(in_length), np.arange(in_length - 1, -1, -1)])
    )
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # Drop zero-weight columns to save computation
    non_zero_cols = np.nonzero(np.any(weights, axis=0))[0]
    weights = np.squeeze(weights[:, non_zero_cols])
    field_of_view = np.squeeze(field_of_view[:, non_zero_cols])

    return weights, field_of_view


def _resize_along_dim(
    im: np.ndarray,
    dim: int,
    weights: np.ndarray,
    field_of_view: np.ndarray,
) -> np.ndarray:
    """
    Apply pre-computed 1-D interpolation weights along one dimension.

    The operation is a gather-and-weight-sum:

        out[i] = Σ_j  weights[i, j] · in[field_of_view[i, j]]

    Broadcasting is used so the same weight matrix applies to all
    non-resized dimensions simultaneously (bsxfun style).

    Parameters
    ----------
    im           : np.ndarray  — current image array
    dim          : int         — dimension to resize along
    weights      : np.ndarray  — ``(out_length, kernel_size)``
    field_of_view: np.ndarray  — ``(out_length, kernel_size)`` input indices

    Returns
    -------
    np.ndarray  — image with dimension ``dim`` resized
    """
    # Swap target dim to front so indexing is always along axis 0
    tmp = np.swapaxes(im, dim, 0)

    # Reshape weights for broadcasting against the gathered tensor
    w = np.reshape(weights.T, list(weights.T.shape) + [1] * (im.ndim - 1))

    # Gather input values and compute weighted sum
    out = np.sum(tmp[field_of_view.T] * w, axis=0)

    return np.swapaxes(out, dim, 0)


def _numeric_kernel(
    im: np.ndarray,
    kernel: np.ndarray,
    scale_factor: list[float],
    output_shape: list[int],
    kernel_shift_flag: bool,
) -> np.ndarray:
    """
    Resize using a custom numeric kernel: correlate then subsample.

    Only valid for downscaling (scale_factor ≤ 1).  Correlation is performed
    per channel using ``scipy.ndimage.correlate`` (kept from the original
    since there is no lightweight pure-NumPy equivalent for arbitrary kernels).

    Parameters
    ----------
    im               : np.ndarray    — ``(H, W, C)``
    kernel           : np.ndarray    — 2-D custom kernel
    scale_factor     : list[float]
    output_shape     : list[int]
    kernel_shift_flag: bool          — align kernel centre of mass first

    Returns
    -------
    np.ndarray  — correlated and subsampled image
    """
    from scipy.ndimage import correlate as ndimage_correlate   # scoped import

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
    """
    Shift a 2-D kernel so its centre of mass aligns with the pixel grid.

    Two corrections are applied:

    1. **Centre-of-mass alignment** — removes any asymmetry introduced during
       kernel estimation, so the degradation model has no implicit translation.

    2. **Sub-pixel grid alignment** — shifts the centre so that the top-left
       output pixel corresponds to the centre of the first sf×sf input block.
       The exact shift differs between odd and even kernel sizes.

    Parameters
    ----------
    kernel : np.ndarray        — 2-D kernel array
    sf     : float or list     — scale factor (scalar or [sy, sx])

    Returns
    -------
    np.ndarray  — shifted kernel (slightly larger due to padding)
    """
    if np.isscalar(sf):
        sf = [sf, sf]

    from scipy.ndimage import measurements
    current_com = measurements.center_of_mass(kernel)

    # Target: kernel centre + half-pixel offset for even-sized kernels
    wanted_com = (
        np.array(kernel.shape) / 2
        + 0.5 * (np.array(sf) - (kernel.shape[0] % 2))
    )

    shift_vec = wanted_com - current_com

    # Pad so nothing is lost after shifting
    pad = int(np.ceil(np.max(np.abs(shift_vec)))) + 1
    kernel = np.pad(kernel, pad, mode="constant")

    return ndimage_shift(kernel, shift_vec)


# ---------------------------------------------------------------------------
# Interpolation kernels
# ---------------------------------------------------------------------------
# All kernels take a distance x from the left pixel centre and return
# the interpolation weight.  They are vectorised over NumPy arrays.

def cubic(x: np.ndarray) -> np.ndarray:
    """
    Keys' bicubic kernel  (a = −0.5).

    Piecewise polynomial with C¹ continuity:

        |x| ≤ 1:  1.5|x|³ − 2.5|x|² + 1
        1 < |x| ≤ 2:  −0.5|x|³ + 2.5|x|² − 4|x| + 2
        |x| > 2:  0

    Support radius: 2.  Reproduces polynomials up to degree 3.
    Default kernel for `imresize_in`.
    """
    absx  = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (
        (1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1)
        + (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0) * ((absx > 1) & (absx <= 2))
    )


def lanczos2(x: np.ndarray) -> np.ndarray:
    """
    Lanczos kernel with window size a=2.

    Defined as sinc(x) · sinc(x/a) for |x| < a, zero otherwise:

        L(x)  =  [sin(πx)·sin(πx/2) + ε] / [(π²x²/2) + ε]   for |x| < 2

    The ε term avoids division by zero at x=0 (where the value should be 1).
    Higher quality than cubic for downscaling but slightly more ringing.
    """
    eps = np.finfo(np.float32).eps
    return (
        ((np.sin(pi * x) * np.sin(pi * x / 2) + eps) / ((pi ** 2 * x ** 2 / 2) + eps))
        * (np.abs(x) < 2)
    )


def lanczos3(x: np.ndarray) -> np.ndarray:
    """
    Lanczos kernel with window size a=3.

    Same construction as lanczos2 but with a wider support (radius 3):

        L(x)  =  [sin(πx)·sin(πx/3) + ε] / [(π²x²/3) + ε]   for |x| < 3

    Sharper than lanczos2 with more ringing; best for high-quality upscaling.
    """
    eps = np.finfo(np.float32).eps
    return (
        ((np.sin(pi * x) * np.sin(pi * x / 3) + eps) / ((pi ** 2 * x ** 2 / 3) + eps))
        * (np.abs(x) < 3)
    )


def box(x: np.ndarray) -> np.ndarray:
    """
    Box (nearest-neighbour) kernel.

    Returns 1.0 for x ∈ [−0.5, 0.5), 0 otherwise.
    Equivalent to nearest-neighbour interpolation; no smoothing.
    Support radius: 0.5.
    """
    return ((-0.5 <= x) & (x < 0.5)).astype(float)


def linear(x: np.ndarray) -> np.ndarray:
    """
    Linear (bilinear / hat) kernel.

    Piecewise linear, C⁰ continuous:

        −1 ≤ x < 0:   x + 1
         0 ≤ x ≤ 1:   1 − x
        otherwise:    0

    Support radius: 1.  Fast and smooth but blurrier than cubic.
    """
    return (
        (x + 1) * ((-1 <= x) & (x < 0))
        + (1 - x) * ((0 <= x) & (x <= 1))
    )