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
import torch.nn.functional as F
from pathlib import Path

from services.generator.utils.config import GANConfig

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
    
def norm(x: torch.Tensor):
    '''Applies standard normalization'''
    out = (x - 0.5) * 2
    return out.clamp(-1, 1) # Consider using the torch.tanh

def denorm(x: torch.Tensor):
    '''Reverts normalization'''
    out = (x + 1) / 2
    return out.clamp(0, 1)

def _image_to_numpy(inp: torch.Tensor) -> np.ndarray:
    '''
    Assuming image is [batch, channels, width, height]
    '''
    inp = denorm(inp)
    inp = move_to_cpu(inp).detach()

    if inp.shape[1] == 3:
        img = inp[-1].numpy().transpose(1, 2, 0)
    elif inp.shape[1] == 1:
        img = inp[-1, 0].numpy()
    else: 
        raise ValueError("Unsupported number of channels must be 1 or 3")
    
    return img.clip(0, 1)
    
def generate_noise(
        size: tuple, 
        n_samples: int = 1,
        device: torch.device | 'str' = 'auto',
        noise_type: str = 'gaussian',
        scale: int = 1
    ):
    
    dev = get_device(device)
    if noise_type == 'gaussian':
        noise = torch.randn(
            n_samples,
            size[0],
            round(size[1]/scale),
            round(size[2]/scale),
            device=dev
        ) 
    elif noise_type == 'uniform':
        noise = torch.randn(
            n_samples, 
            size[0],
            size[1],
            size[2],
            device=dev
        )
    else: 
        raise NotImplementedError
    
    return noise

def upsampling(img: torch.Tensor, sx: int, sy: int) -> torch.tensor:
    m = F.interpolate(
        img, 
        size=(round(sx), round(sy)),
        mode='bilinear',
        align_corners=True # This is not recommended by documentation but the original paper uses it
    )
    return m

def save_image(name: str, img: torch.Tensor):
    plt.imsave(name, _image_to_numpy(img), vmin = 0, vmax= 1)

def sample_random_noise(depth: int, reals_shapes: tuple[tuple], opt: GANConfig):
    noise = []
    for d in range(depth - 1):
        if d == 0:
            noise.append(generate_noise(
                size = [opt.channels, reals_shapes[d][2], reals_shapes[d][3]],
                device = opt.device
            ).detach()
        )
        else:
            # In generation mode
            noise.append(generate_noise(
                size = [opt.filters_per_conv, reals_shapes[d][2] + opt.num_layers * 2, reals_shapes[d][3] + opt.num_layers * 2],
                device=opt.device
            ).detach()
        )

class Augment:
    def __init__(self):
        pass


move_to_cpu = lambda _Tensor : _Tensor.to('cpu')
move_to_gpu = lambda _Tensor : (_Tensor.to('cuda') 
                                if torch.cuda.is_available() 
                                else _Tensor.to('cpu')
                            )
