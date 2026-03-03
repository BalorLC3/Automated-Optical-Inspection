import os
import math
import copy
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from torchvision.io import decode_image
from torchvision.transforms.v2 import (
    Compose,
    RandomChoice,
    RandomApply,
    RandomInvert,
    ColorJitter,
    GaussianNoise,
    RandomErasing,
)
from torchvision.transforms.functional import (
    rgb_to_grayscale,
    gaussian_blur,
)
from typing import Callable

from services.generator.utils.config import GANConfig
from services.generator.utils.imresize import imresize


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(device: torch.device | str = "auto") -> torch.device:
    """
    Resolve a torch.device.

    Parameters
    ----------
    device : "auto" | str | torch.device
        "auto" picks CUDA when available, otherwise CPU.

    Returns
    -------
    torch.device
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

# Reproducibility
def set_seed(seed: int = 17) -> None:
    """
    Fix all random seeds for reproducible runs.

    Parameters
    ----------
    seed : int
        Seed value applied to Python random, NumPy, and PyTorch (CPU + GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Normalisation / denormalisation
def _norm(x: torch.Tensor) -> torch.Tensor:
    """
    Map pixel values from [0, 1] → [-1, 1].

    Formula:  out = (x - 0.5) × 2,  clamped to [-1, 1].

    This centres the data so the generator and discriminator see
    zero-mean inputs, which stabilises GAN training.

    Parameters
    ----------
    x : torch.Tensor
        Float tensor with values in [0, 1].

    Returns
    -------
    torch.Tensor  - values in [-1, 1]
    """
    return (x - 0.5).mul_(2).clamp_(-1, 1)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of :func:`norm`: map [-1, 1] → [0, 1].

    Formula:  out = (x + 1) / 2,  clamped to [0, 1].

    Parameters
    ----------
    x : torch.Tensor
        Float tensor with values in [-1, 1].

    Returns
    -------
    torch.Tensor  - values in [0, 1]
    """
    return x.add(1).div_(2).clamp_(0, 1)


# Tensor <-> NumPy conversion
def image_to_numpy(inp: torch.Tensor) -> np.ndarray:
    """
    Convert a batched image tensor to a displayable NumPy array.

    Expects shape ``(B, C, H, W)`` where C is 1 (greyscale) or 3 (RGB).
    Takes the *last* sample in the batch (index -1).

    Parameters
    ----------
    inp : torch.Tensor
        Normalised image tensor (values in [-1, 1]).

    Returns
    -------
    np.ndarray
        - RGB:       shape ``(H, W, 3)``,  dtype float32, range [0, 1]
        - Greyscale: shape ``(H, W)``,     dtype float32, range [0, 1]

    Raises
    ------
    ValueError
        If the channel dimension is neither 1 nor 3.
    """
    inp = _denorm(inp).detach().cpu()

    if inp.shape[1] == 3:
        return inp[-1].numpy().transpose(1, 2, 0).clip(0, 1)
    elif inp.shape[1] == 1:
        return inp[-1, 0].numpy().clip(0, 1)
    else:
        raise ValueError(f"Expected 1 or 3 channels, got {inp.shape[1]}")

# Image I/O
def read_image(path: str, config: GANConfig) -> torch.Tensor:
    """
    Load an image from disk into a normalised float tensor.

    Uses torchvision's ``decode_image`` (replaces skimage ``imread``).

    Parameters
    ----------
    path   : str        - file path (JPEG, PNG, …)
    config : GANConfig  - must contain ``config.channels`` (1 or 3)

    Returns
    -------
    torch.Tensor  - shape ``(1, C, H, W)``, values in [-1, 1]

    Raises
    ------
    ValueError
        If ``config.channels`` is neither 1 nor 3.
    """
    x = decode_image(path)          # uint8, (C, H, W), RGB

    if config.channels == 3:
        x = x[:3]                   # drop alpha channel if present
    elif config.channels == 1:
        x = rgb_to_grayscale(x)
    else:
        raise ValueError("channels must be 1 or 3")

    x = x.unsqueeze(0).float().div(255.0)   # (1, C, H, W) in [0, 1]
    return _norm(x)


def read_image_dir(path: str, config: GANConfig) -> torch.Tensor:
    """
    Alias of :func:`read_image` - load an image given a directory path.

    Provided for API compatibility with the 2021 codebase.
    """
    return read_image(path, config)


def read_image2np(path: str) -> np.ndarray:
    """
    Load an image directly into a NumPy array (uint8, RGB, no normalisation).

    Parameters
    ----------
    path : str  - file path

    Returns
    -------
    np.ndarray  - shape ``(H, W, 3)``, dtype uint8
    """
    x = decode_image(path)              # uint8 tensor (C, H, W)
    return x[:3].permute(1, 2, 0).numpy()   # (H, W, 3)


def save_image(name: str, img: torch.Tensor) -> None:
    """
    Save a normalised image tensor to disk as a PNG/JPEG.

    Parameters
    ----------
    name : str           - output file path
    img  : torch.Tensor  - normalised tensor ``(B, C, H, W)``
    """
    plt.imsave(fname=name, arr=image_to_numpy(img), vmin=0, vmax=1)


def upsampling(img: torch.Tensor, sx: int, sy: int) -> torch.Tensor:
    """
    Bilinear upsample a tensor to spatial size ``(sx, sy)``.

    Wraps ``F.interpolate`` with ``align_corners=True`` to match the
    behaviour of the original ``nn.Upsample`` call in the 2021 codebase.

    Parameters
    ----------
    img : torch.Tensor  - shape ``(B, C, H, W)``
    sx  : int           - target height
    sy  : int           - target width

    Returns
    -------
    torch.Tensor  - shape ``(B, C, sx, sy)``
    """
    return F.interpolate(
        img,
        size=(round(sx), round(sy)),
        mode="bilinear",
        align_corners=True,
    )

def generate_noise(
    size: tuple[int, int, int],
    config: GANConfig,
    n_samples: int = 1,
    noise_type: str = "gaussian",
    scale: int = 1,
) -> torch.Tensor:
    """
    Sample a noise tensor used as generator input.

    Gaussian noise is the default - it is first generated at a coarser
    spatial resolution ``(H/scale, W/scale)`` and then upsampled to
    ``(H, W)``.  This introduces low-frequency structure, giving the
    generator a useful inductive bias at each pyramid scale.

    Parameters
    ----------
    size       : (C, H, W)     - desired output spatial dimensions
    config     : GANConfig     - holds ``config.device``
    n_samples  : int           - batch size
    noise_type : "gaussian" | "uniform"
    scale      : int           - spatial down-factor before upsampling

    Returns
    -------
    torch.Tensor  - shape ``(n_samples, C, H, W)``

    Raises
    ------
    NotImplementedError  - for unknown noise_type
    """
    dev = get_device(config.device)

    if noise_type == "gaussian":
        noise = torch.randn(
            n_samples,
            size[0],
            round(size[1] / scale),
            round(size[2] / scale),
            device=dev,
        )
        noise = upsampling(noise, size[1], size[2])
    elif noise_type == "uniform":
        noise = torch.rand(
            n_samples, size[0], size[1], size[2], device=dev
        )
    else:
        raise NotImplementedError(f"Unknown noise type: {noise_type!r}")

    return noise


def sample_random_noise(
    depth: int,
    reals_shapes: list[tuple],
    config: GANConfig,
) -> list[torch.Tensor]:
    """
    Build the noise pyramid fed to the generator at inference time.

    One noise tensor is produced per pyramid level (0 … depth inclusive).

    - Level 0 uses ``config.channels`` channels at the coarsest resolution.
    - Higher levels use ``config.filters_per_conv`` channels padded by
      ``config.num_layers`` pixels on each side (generation mode only).

    Parameters
    ----------
    depth        : int              - highest scale index
    reals_shapes : list of tuples   - spatial shapes of each pyramid level
    config       : GANConfig

    Returns
    -------
    list[torch.Tensor]  - length ``depth + 1``
    """
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(
                generate_noise(
                    [config.channels, reals_shapes[d][2], reals_shapes[d][3]],
                    config=config,
                ).detach()
            )
        else:
            noise.append(
                generate_noise(
                    [
                        config.filters_per_conv,
                        reals_shapes[d][2] + config.num_layers * 2,
                        reals_shapes[d][3] + config.num_layers * 2,
                    ],
                    config=config,
                ).detach()
            )
    return noise

# WGAN-GP gradient penalty
def calc_gradient_penalty(
    discriminator: Callable[[torch.Tensor], torch.Tensor],
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    lambda_: float,
    config: GANConfig,
) -> torch.Tensor:
    """
    Compute the WGAN-GP gradient penalty.

    Theory
    ------
    Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017) enforces
    the 1-Lipschitz constraint on the discriminator D by penalising deviations of ‖∇D(x̂)‖₂ from 1, where x̂ is a random convex interpolation between a real and a fake sample:

        x̂ = ε · x_real + (1 - ε) · x_fake,   ε ~ Uniform[0, 1]

    The penalty added to the discriminator loss is:

        GP = λ · E[ (‖∇_{x̂} D(x̂)‖₂ - 1)² ]

    A large gradient norm means the discriminator is changing too fast -
    the penalty pushes it back toward 1-Lipschitz behaviour.

    Parameters
    ----------
    discriminator : callable  - D(x) → scalar score
    real_data     : Tensor    - shape ``(B, C, H, W)``
    fake_data     : Tensor    - same shape as real_data
    lambda_       : float     - GP weight (typically 0.1 – 10)
    config        : GANConfig

    Returns
    -------
    torch.Tensor  - scalar gradient penalty (differentiable)
    """
    dev = get_device(config.device)

    # Random mixing coefficient, broadcast to data shape
    alpha = torch.rand(1, 1, device=dev).expand_as(real_data)

    # Interpolated sample on the straight line between real and fake
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    disc_out = discriminator(interpolates)

    # Compute ∂D/∂x̂ via autograd
    gradients = torch.autograd.grad(
        outputs=disc_out,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_out, device=dev),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty

# Scale pyramid helpers
def adjust_scales2image(
    real_: torch.Tensor,
    config: GANConfig,
) -> torch.Tensor:
    """
    Compute the global rescale factor and per-level scale factor.

    Multi-scale training requires a *pyramid* of the input image.
    This function determines:

    1. ``config.scale1`` - shrinks the image so its longest side equals
       ``config.max_size``.
    2. ``config.scale_factor`` - the geometric ratio between consecutive
       pyramid levels, chosen so the coarsest level has a shortest side
       of ``config.min_size`` pixels.

    The scale factor satisfies:

        min_dim · r^(stop_scale) = min_size
        ⟹  r = (min_size / min_dim)^(1 / stop_scale)

    Parameters
    ----------
    real_  : torch.Tensor  - raw input image ``(1, C, H, W)``
    config : GANConfig

    Returns
    -------
    torch.Tensor  - rescaled image at scale1
    """
    config.scale1 = min(
        config.max_size / max(real_.shape[2], real_.shape[3]), 1.0
    )
    real = imresize(real_, config.scale1, config)

    config.stop_scale = config.train_stages - 1
    config.scale_factor = math.pow(
        config.min_size / min(real.shape[2], real.shape[3]),
        1 / config.stop_scale,
    )
    return real


def create_reals_pyramid(
    real: torch.Tensor,
    config: GANConfig,
) -> list[torch.Tensor]:
    """
    Build the multi-resolution image pyramid used for training.

    Each level is a differently scaled version of the full-resolution image.
    The scaling schedule depends on the training mode:

    - **Harmonisation** uses a simple geometric progression:
        scale_i = r^(stop_scale - i)

    - **All other modes** use a log-linear schedule that spaces levels
      more evenly in log-scale:
        exponent = ((S-1) / log S) · log(S - i) + 1

    where S = ``config.stop_scale``.  The ``+1`` ensures the finest level
    is always at full resolution (scale = 1).

    Parameters
    ----------
    real   : torch.Tensor  - full-resolution input ``(1, C, H, W)``
    config : GANConfig

    Returns
    -------
    list[torch.Tensor]
        Images from coarsest (index 0) to finest (index -1, full-res).
    """
    reals = []
    S = config.stop_scale

    for i in range(S):
        if config.train_mode == "harmonization":
            scale = math.pow(config.scale_factor, S - i)
        else:
            # Log-linear spacing - avoids bunching levels near full resolution
            scale = math.pow(
                config.scale_factor,
                ((S - 1) / math.log(S)) * math.log(S - i) + 1,
            )
        reals.append(imresize(real, scale, config))

    reals.append(real)
    return reals

# Checkpoint I/O
def save_networks(
    generator: nn.Module,
    discriminators: nn.Module | list[nn.Module],
    z: torch.Tensor,
    config: GANConfig,
) -> None:
    """
    Persist generator, discriminator(s), and fixed noise to disk.

    Saves:
    - ``<outf>/netG.pth``
    - ``<outf>/netD.pth``  (single) or ``<outf>/netD_<i>.pth``  (list)
    - ``<outf>/z_opt.pth``

    Parameters
    ----------
    generator      : nn.Module
    discriminators : nn.Module or list[nn.Module]
    z              : torch.Tensor  - fixed latent/noise used during training
    config         : GANConfig     - must have ``config.outf`` (output dir)
    """
    torch.save(generator.state_dict(), os.path.join(config.outf, "netG.pth"))

    if isinstance(discriminators, list):
        for i, netD in enumerate(discriminators):
            torch.save(
                netD.state_dict(),
                os.path.join(config.outf, f"netD_{i}.pth"),
            )
    else:
        torch.save(
            discriminators.state_dict(),
            os.path.join(config.outf, "netD.pth"),
        )

    torch.save(z, os.path.join(config.outf, "z_opt.pth"))

def load_trained_model(config: GANConfig):
    """
    Load a fully trained ConSinGAN model from disk.

    Expects the checkpoint directory produced by :func:`generate_dir2save`
    to contain ``Gs.pth``, ``Zs.pth``, ``reals.pth``, and ``NoiseAmp.pth``.

    Parameters
    ----------
    config : GANConfig

    Returns
    -------
    tuple  - (Gs, Zs, reals, NoiseAmp)

    Raises
    ------
    FileNotFoundError  - if the model directory does not exist
    """
    dir_ = generate_dir2save(config)

    if not os.path.exists(dir_):
        raise FileNotFoundError(f"No trained model found at: {dir_}")

    map_loc = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    Gs       = torch.load(os.path.join(dir_, "Gs.pth"),       map_location=map_loc)
    Zs       = torch.load(os.path.join(dir_, "Zs.pth"),       map_location=map_loc)
    reals    = torch.load(os.path.join(dir_, "reals.pth"),    map_location=map_loc)
    NoiseAmp = torch.load(os.path.join(dir_, "NoiseAmp.pth"), map_location=map_loc)

    return Gs, Zs, reals, NoiseAmp

def generate_dir2save(config: GANConfig) -> str:
    """
    Construct a deterministic output directory path for a training run.

    The path encodes all hyper-parameters that affect the model so that
    different runs never overwrite each other:

        TrainedModels/<image_name>/<timestamp>_<mode>_train_depth_<d>_lr_scale_<lr>[_BN][_act_<act>[_<alpha>]]

    Parameters
    ----------
    config : GANConfig

    Returns
    -------
    str  - relative directory path
    """
    image_name = config.input_name[:-4].split("/")[-1]
    dir2save = f"TrainedModels/{image_name}/{config.timestamp}_{config.train_mode}"

    if config.train_mode in ("harmonization", "editing") and config.fine_tune:
        dir2save += "_fine-tune"

    dir2save += f"_train_depth_{config.train_depth}_lr_scale_{config.lr_scale}"

    if config.batch_norm:
        dir2save += "_BN"

    dir2save += f"_act_{config.activation}"
    if config.activation == "lrelu":
        dir2save += f"_{config.lrelu_alpha}"

    return dir2save

def post_config(config: GANConfig) -> GANConfig:
    """
    Finalise configuration after CLI / file parsing.

    Sets derived fields that are computed rather than directly specified:
    - ``config.device``          - torch.device resolved from ``config.gpu``
    - ``config.noise_amp_init``  - copy of initial noise amplitude
    - ``config.timestamp``       - ISO-style timestamp for the run
    - ``config.manualSeed``      - random seed (generated if None)

    Parameters
    ----------
    config : GANConfig

    Returns
    -------
    GANConfig  - same object, mutated in-place and returned
    """
    config.device = get_device(config.device)
    config.noise_amp_init = config.noise_amp
    config.timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10_000)

    set_seed(config.manualSeed)

    if torch.cuda.is_available() and config.not_cuda:
        print("WARNING: CUDA is available but --not_cuda is set.")

    return config


def load_config(config: GANConfig) -> GANConfig:
    """
    Override ``config`` fields from a saved ``parameters.txt`` file.

    Each line in the file has the format ``key - value``.
    Values are cast to int or float where possible; otherwise kept as str.

    Parameters
    ----------
    config : GANConfig  - must have ``config.model_dir``

    Returns
    -------
    GANConfig  - same object with fields overwritten from file

    Raises
    ------
    SystemExit  - if ``config.model_dir`` does not exist
    """
    if not os.path.exists(config.model_dir):
        raise FileNotFoundError(f"Model directory not found: {config.model_dir}")

    params_path = os.path.join(config.model_dir, "parameters.txt")
    with open(params_path) as f:
        for line in f:
            key, _, value = line.partition("-")
            key, value = key.strip(), value.strip()
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass
            setattr(config, key, value)

    return config


# Mask dilation
def _binary_dilation_torch(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Morphological binary dilation using max-pooling (PyTorch, no skimage).

    Dilation replaces each pixel with the maximum value in its neighbourhood,
    effectively expanding bright regions by ``radius`` pixels.

    A disk-shaped structuring element is approximated by a square kernel of
    size ``(2·radius + 1)``, which is fast via ``F.max_pool2d``.

    Parameters
    ----------
    mask   : torch.Tensor  - binary mask ``(1, 1, H, W)``
    radius : int

    Returns
    -------
    torch.Tensor  - dilated mask, same shape
    """
    k = 2 * radius + 1
    return F.max_pool2d(mask, k, stride=1, padding=radius)


def dilate_mask(mask: torch.Tensor, config: GANConfig) -> torch.Tensor:
    """
    Dilate and feather a segmentation mask for harmonisation / editing.

    Steps:

    1. **Dilation** - expand the mask boundary using max-pooling.
       Radius is 7 px (harmonisation) or 20 px (editing) to give the
       generator a comfortable margin around the pasted region.

    2. **Gaussian blur** - smooth the hard boundary into a soft transition
       zone so the generator can blend seamlessly.

    3. **Normalise** - remap to [0, 1] so the mask can be used as a soft
       spatial weight.

    Parameters
    ----------
    mask   : torch.Tensor  - ``(1, 3, H, W)`` or ``(1, 1, H, W)``
    config : GANConfig

    Returns
    -------
    torch.Tensor  - soft mask ``(1, 3, H, W)``
    """
    radius = 7 if config.train_mode == "harmonization" else 20

    mask = mask[:, :1]                              # keep only first channel → (1,1,H,W)
    mask = _binary_dilation_torch(mask, radius)
    mask = gaussian_blur(mask, kernel_size=11, sigma=5.0)
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


# Augmentation
class Augment:
    """
    On-the-fly stochastic augmentation pipeline.

    A new random pipeline is built on every call so that the strength and
    number of erasing patches varies per-sample.

    The pipeline applies **one** of {GaussianNoise, RandomInvert, ColorJitter}
    and then, with probability 0.9, pastes 1 or 2 randomly-coloured patches
    (``RandomErasing``) to simulate partial occlusion.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a freshly sampled augmentation to ``x``.

        Parameters
        ----------
        x : torch.Tensor  - image tensor ``(C, H, W)`` in [0, 1]

        Returns
        -------
        torch.Tensor  - augmented image, same shape
        """
        return self._build_pipeline()(x)

    def _build_pipeline(self) -> Compose:
        num_holes = random.randint(1, 2)
        scale = (0.02, 0.08) if num_holes == 2 else (0.08, 0.2)

        return Compose([
            RandomChoice([
                GaussianNoise(),
                RandomInvert(p=1.0),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            ]),
            RandomApply(
                [RandomErasing(p=1.0, scale=scale, ratio=(0.5, 2.0), value="random")
                 for _ in range(num_holes)],
                p=0.9,
            ),
        ])


# Grid shuffle (spatial self-supervision)
def shuffle_grid(image: np.ndarray, max_tiles: int = 5) -> np.ndarray:
    """
    Randomly translate rectangular patches within an image (in-place copy).

    This is used as a self-supervised signal: the discriminator must learn
    to detect unnatural patch boundaries, encouraging the generator to
    produce globally coherent textures.

    Parameters
    ----------
    image     : np.ndarray  - shape ``(H, W, C)``
    max_tiles : int         - maximum number of patches to displace

    Returns
    -------
    np.ndarray  - modified copy of ``image``
    """
    img_w, img_h = image.shape[0], image.shape[1]
    n = random.randint(1, max_tiles)

    # Tile and translation bounds scale with number of tiles
    bounds = {
        1: dict(w=(0.2, 0.5),  h=(0.2, 0.5),  tx=(0.05, 0.15), ty=(0.05, 0.15)),
        2: dict(w=(0.15, 0.3), h=(0.15, 0.3), tx=(0.05, 0.10), ty=(0.05, 0.10)),
        3: dict(w=(0.1, 0.2),  h=(0.1, 0.2),  tx=(0.05, 0.10), ty=(0.05, 0.10)),
    }.get(n, dict(w=(0.1, 0.15), h=(0.1, 0.15), tx=(0.05, 0.10), ty=(0.05, 0.10)))

    w_min, w_max = int(img_w * bounds["w"][0]), int(img_w * bounds["w"][1])
    h_min, h_max = int(img_h * bounds["h"][0]), int(img_h * bounds["h"][1])
    tx_min, tx_max = int(img_w * bounds["tx"][0]), int(img_w * bounds["tx"][1])
    ty_min, ty_max = int(img_h * bounds["ty"][0]), int(img_h * bounds["ty"][1])

    tiles = []
    for _ in range(n):
        x, y = random.randint(0, img_w - 1), random.randint(0, img_h - 1)
        w = min(random.randint(w_min, w_max), img_w - x)
        h = min(random.randint(h_min, h_max), img_h - y)
        tx = random.randint(tx_min, tx_max) * (1 if random.random() > 0.5 else -1)
        ty = random.randint(ty_min, ty_max) * (1 if random.random() > 0.5 else -1)

        # Clamp destination so the tile stays inside the image
        x_new = max(0, min(x + tx, img_w - w))
        y_new = max(0, min(y + ty, img_h - h))
        tiles.append((x, y, w, h, x_new, y_new))

    new_image = copy.deepcopy(image)
    for x, y, w, h, xn, yn in tiles:
        new_image[xn:xn + w, yn:yn + h, :] = image[x:x + w, y:y + h, :]

    return new_image


def save_video(out_path: str, frames: list[np.ndarray], fps: int):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    writer.release()

def generate_gif(
    dir2save: str,
    netG: nn.Module,
    fixed_noise: list[torch.Tensor],
    reals: list[torch.Tensor],
    noise_amp: list[float],
    config: GANConfig,
    alpha: float = 0.1,
    beta: float = 0.9,
    start_scale: int = 1,
    num_images: int = 100,
    fps: int = 10,
) -> None:
    """
    Render a smooth animation GIF by interpolating in noise space.

    Interpolation scheme
    --------------------
    The noise walk uses an exponential moving average (EMA) with momentum
    ``beta`` to create smooth, temporally coherent transitions:

        diff_t  = b·(z_{t-1} - z_{t-2}) + (1-B)·e_t     [momentum step]
        z_t     = a·z_fixed + (1-a)·(z_{t-1} + diff_t)   [anchor to fixed noise]

    - ``beta``  controls smoothness: higher → slower, smoother motion.
    - ``alpha`` controls attraction to the fixed noise: higher → less drift.
    - ``start_scale`` locks the coarser levels to fixed noise, animating
      only fine-scale details.

    Parameters
    ----------
    dir2save    : str              - output directory
    netG        : nn.Module        - trained generator
    fixed_noise : list[Tensor]     - anchor noise pyramid
    reals       : list[Tensor]     - real image pyramid (for shapes)
    noise_amp   : list[float]      - per-scale noise amplitude
    config      : GANConfig
    alpha       : float            - anchor strength (0 = free, 1 = static)
    beta        : float            - momentum for smooth walk
    start_scale : int              - first scale to animate (0 = all scales)
    num_images  : int              - number of frames
    fps         : int              - frames per second

    Returns
    -------
    None  - writes a .gif file to ``dir2save``
    """
    def _to_frame(t: torch.Tensor) -> np.ndarray:
        t = _denorm(t).detach()[0].cpu().numpy().transpose(1, 2, 0)
        return (t * 255).astype(np.uint8)

    reals_shapes = [r.shape for r in reals]
    all_frames: list[np.ndarray] = []

    with torch.no_grad():
        z_prev1 = [0.99 * fixed_noise[i] + 0.01 * sample_random_noise(len(fixed_noise) - 1, reals_shapes, config)[i]
                   for i in range(len(fixed_noise))]
        z_prev2 = fixed_noise

        for _ in range(num_images):
            eps = sample_random_noise(len(fixed_noise) - 1, reals_shapes, config)
            diff = [beta * (z_prev1[i] - z_prev2[i]) + (1 - beta) * eps[i]
                    for i in range(len(fixed_noise))]
            z_curr = [alpha * fixed_noise[i] + (1 - alpha) * (z_prev1[i] + diff[i])
                      for i in range(len(fixed_noise))]

            # Lock coarse scales to fixed noise
            if start_scale > 0:
                z_curr = [fixed_noise[i] for i in range(start_scale)] + \
                         [z_curr[i] for i in range(start_scale, len(fixed_noise))]

            z_prev2, z_prev1 = z_prev1, z_curr
            all_frames.append(_to_frame(netG(z_curr, reals_shapes, noise_amp)))

    out_path = os.path.join(
        dir2save, f"start_scale={start_scale}_alpha={alpha}_beta={beta}.gif"
    )
    save_video(out_path, all_frames, fps=fps)


move_to_cpu: Callable[[torch.Tensor], torch.Tensor] = lambda t: t.to("cpu")
move_to_gpu: Callable[[torch.Tensor], torch.Tensor] = (
    lambda t: t.to("cuda") if torch.cuda.is_available() else t.to("cpu")
)