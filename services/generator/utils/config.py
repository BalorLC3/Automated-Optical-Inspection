from dataclasses import dataclass

@dataclass
class GANConfiguration:
    # Hypeparameters
    padding: int = 0
    kernel: int = 3
    num_layers: int = 3
    filters_per_conv: int = 64    

    # Pyramid parameters
    channels: int = 3
    additive_noise: float = 0.1
    min_size: int = 25
    max_size: int = 250
    train_depth: int = 3
    start_scale: int = 0

    # Optimization hyperparameters
    max_epochs: int = 2000
    gamma: float = 0.1
    lr_g: float = 1e-5
    lr_d: float = 1e-5
    beta: float = 0.5
    generator_steps: int = 3
    discriminator_steps: int = 3
    lambda_grad: float = 0.1
    alpha: float = 10
    activation: str = 'relu' # Original paper used leaky RELU so alphaLRelu can be added
    batch_norm: int = 0