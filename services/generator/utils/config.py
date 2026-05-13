from __future__ import annotations

import logging
from dataclasses import dataclass, field


def create_logger(name: str = "consingan") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    return logging.getLogger(name)


@dataclass
class GANConfig:
    # Hardware
    device: str = "auto"       # "auto" | "cpu" | "cuda"
    gpu: int = 0
    not_cuda: bool = False

    # Architecture
    channels: int = 3
    padding: int = 0
    kernel: int = 3
    num_layers: int = 3
    filters_per_conv: int = 64
    activation: str = "lrelu"  # "relu" | "lrelu" | "elu" | "selu"
    lrelu_alpha: float = 0.05
    elu_alpha: float = 1.0
    batch_norm: bool = False

    # Training mode
    train_mode: str = "generation"   # "generation" | "harmonization" | "animation"
    fine_tune: bool = False

    # Pyramid
    min_size: int = 25
    max_size: int = 250
    train_stages: int = 6
    train_depth: int = 3
    start_scale: int = 0

    # Derived at runtime — do not set manually
    stop_scale: int = 0
    scale_factor: float = 0.75
    scale1: float = 1.0

    # Noise
    noise_amp_init: float = 0.1
    additive_noise: float = 0.1

    # Optimisation
    max_epochs: int = 1500
    gamma: float = 0.1
    lr_g: float = 5e-4
    lr_d: float = 5e-4
    lr_scale: float = 0.1
    beta1: float = 0.5
    generator_steps: int = 3
    discriminator_steps: int = 3
    lambda_grad: float = 0.1
    alpha: float = 10.0           # Reconstruction loss weight

    # I/O
    input_name: str = ""
    outf: str = "output"
    model_dir: str = ""

    # Runtime
    timestamp: str = ""
    manualSeed: int | None = None