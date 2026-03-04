from __future__ import annotations
from dataclasses import dataclass, field
import logging

# NOTE: Consider using YAML files and a script to automatically generate
#       runs next time 

def create_logger(name: str = 'generation-service'):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger(name)
    return logger

@dataclass
class GANConfig:
    # Hardware
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    gpu: int = 0                  # GPU index when device="cuda"
    not_cuda: bool = False        # Force CPU even if CUDA is available

    # Architecture
    channels: int = 3             # Input image channels (1 or 3)
    padding: int = 0
    kernel: int = 3
    num_layers: int = 3           # Conv layers per stage
    filters_per_conv: int = 64
    activation: str = "relu"      # "relu" | "lrelu"
    lrelu_alpha: float = 0.05     # Slope for leaky ReLU (ignored if activation != "lrelu")
    elu_alpha: float = 1.0
    batch_norm: bool = False

    # Training mode
    train_mode: str = "harmonization"  # "generation" | "harmonization" | "editing" | "animation"
    fine_tune: bool = False            # Fine-tune on harmonization/editing tasks

    # Pyramid
    min_size: int = 25            # Shortest side at coarsest pyramid level (px)
    max_size: int = 250           # Longest side after global rescale (px)
    train_stages: int = 6         # Total number of pyramid levels
    train_depth: int = 3          # Number of scales trained simultaneously (ConSinGAN)
    start_scale: int = 0          # First scale to animate in generate_gif

    # Derived at runtime by adjust_scales2image — do not set manually
    stop_scale: int = 0
    scale_factor: float = 0.0
    scale1: float = 1.0

    # Noise
    additive_noise: float = 0.1
    noise_amp: float = 0.1        # Initial noise amplitude
    noise_amp_init: float = 0.0   # Copy of noise_amp set by post_config

    # Optimisation
    max_epochs: int = 2000
    gamma: float = 0.1
    lr_g: float = 1e-5
    lr_d: float = 1e-5
    lr_scale: float = 0.1         # LR decay per stage
    beta: float = 0.5             # Adam beta1
    generator_steps: int = 3
    discriminator_steps: int = 3
    lambda_grad: float = 0.1      # WGAN-GP lambda
    alpha: float = 10             # Reconstruction loss weight

    # I/O
    input_name: str = ""          # Path to input image
    outf: str = "output"          # Output directory for checkpoints
    model_dir: str = ""           # Directory of a saved model to load

    # Derived runtime fields (set by post_config)
    timestamp: str = ""
    manualSeed: int | None = None