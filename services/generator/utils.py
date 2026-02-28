import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import os
import random
import datetime
from torchvision.transforms import v2

from dataclasses import dataclass


def get_device(device: torch.device | str = "auto") -> torch.device:
    '''
    Automatically detects device
    '''
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def set_seed(seed: int = 17):
    '''
    Global seed helper
    '''
    random.seed(seed)
    np.random.seed(17)
    torch.manual_seed(17)
    device = get_device()
    if device == "cuda":
        torch.cuda.manual_seed(17)
        torch.cuda.manual_seed_all(17)

@dataclass
class Configuration:
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
    

def multiplicative_noise():
    # v2.GaussianNoise()
    ...
    

class Augment:
    def __init__(self):
        pass