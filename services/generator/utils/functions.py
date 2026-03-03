import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import os
import random
import datetime
from torchvision.transforms.v2 import (
    Compose,
    RandomChoice,
    RandomApply,
    RandomInvert,
    ColorJitter,
    GaussianNoise,
    RandomErasing,
)
import torch.nn.functional as F
from pathlib import Path
from typing import Callable
from torchvision.io import decode_image
from torchvision.transforms.functional import (
    rgb_to_grayscale,
    gaussian_blur,
)
import copy

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
        raise ValueError("Unsupported number of channels, must be 1 or 3")
    
    return img.clip(0, 1)
    
def generate_noise(
        size: tuple[int, int], 
        config: GANConfig,
        n_samples: int = 1,
        noise_type: str = 'gaussian',
        scale: int = 1,
    ):
    
    dev = get_device(config.device)

    if noise_type == 'gaussian':
        noise = torch.randn(
            n_samples,
            size[0],
            round(size[1]/scale),
            round(size[2]/scale),
            device=dev
        ) 
        noise = upsampling(noise, size[1], size[2])
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
        input = img, 
        size=(round(sx), round(sy)),
        mode='bilinear',
        align_corners=True # This is not recommended by documentation but the original paper uses it
    )
    return m

def save_image(name: str, img: torch.Tensor):
    plt.imsave(
        fname = name, 
        arr = _image_to_numpy(img), 
        vmin = 0, 
        vmax = 1
    )

def sample_random_noise(
        depth: int, 
        reals_shapes: tuple[tuple], 
        config: GANConfig
    ):
    dev = get_device(config.device)
    noise = []
    for d in range(depth - 1):
        if d == 0:
            noise.append(generate_noise(
                    size = [config.channels, reals_shapes[d][2], reals_shapes[d][3]],
                    device = dev
                ).detach()
            )
        else:
            # In generation mode
            noise.append(generate_noise(
                    size = [config.filters_per_conv, reals_shapes[d][2] + config.num_layers * 2, reals_shapes[d][3] + config.num_layers * 2],
                    device = dev
                ).detach()
            )

def calc_gradient_penalty(
        discriminator: Callable[[torch.Tensor], torch.Tensor], 
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        lambda_: float, # Constant Lambda
        config: GANConfig
    ):
    # Get device
    dev = get_device(config.device)
    
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(dev)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(dev)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    discriminator_intp = discriminator(interpolates)

    gradients =  torch.autograd.grad(
        outputs = discriminator_intp,
        inputs = interpolates,
        grad_outputs = torch.ones(discriminator_intp.size()).to(dev),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_

    return gradient_penalty

def read_image(
        path: str, 
        config: GANConfig
    ):
    # uint8 tensor, shape (C, H, W), RGB
    x = decode_image(path)

    if config.channels == 3:
        x = x[:3]      # Drop alpha if exists
    else:
        x = rgb_to_grayscale(x)

    x = x.unsqueeze(0).float() / 255.0  # (1, C, H, W)
    x = norm(x)
    return x

def binary_dilation_torch(mask, radius: int):
    kernel_size = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size, stride=1, padding=radius)

def dilate_mask(mask, config: GANConfig):
    radius = 7 if config.train_mode == "harmonization" else 20

    # mask: (1,3,H,W) -> (1,1,H,W)
    mask = mask[:, :1]

    mask = binary_dilation_torch(mask, radius)
    mask = gaussian_blur(mask, kernel_size=11, sigma=5)

    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    return mask

def shuffle_grid(image, max_tiles=5):
    tiles = []
    img_w, img_h = image.shape[0], image.shape[1]
    _max_tiles = random.randint(1, max_tiles)
    # _max_tiles = random.randint(3,3)
    if _max_tiles == 1:
        w_min, h_min = int(img_w*0.2), int(img_h*0.2)
        w_max, h_max = int(img_w*0.5), int(img_h*0.5)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.15), int(img_h*0.15)
    elif _max_tiles == 2:
        w_min, h_min = int(img_w*0.15), int(img_h*0.15)
        w_max, h_max = int(img_w*0.3), int(img_h*0.3)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)
    elif _max_tiles == 3:
        w_min, h_min = int(img_w*0.1), int(img_h*0.1)
        w_max, h_max = int(img_w*0.2), int(img_h*0.2)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)
    else:
        w_min, h_min = int(img_w*0.1), int(img_h*0.1)
        w_max, h_max = int(img_w*0.15), int(img_h*0.15)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)

    for _ in range(_max_tiles):
        x, y = random.randint(0, img_w), random.randint(0, img_h)
        w, h = random.randint(w_min, w_max), random.randint(h_min, h_max)
        if x + w >= img_w:
            w = img_w - x
        if y + h >= img_h:
            h = img_h - y
        x_t, y_t = random.randint(x_translation_min, x_translation_max), random.randint(y_translation_min, y_translation_max)
        if random.random() < 0.5:
            x_t, y_t = -x_t, -y_t
            if x + x_t < 0:
                x_t = -x
            if y + y_t < 0:
                y_t = -y
        else:
            if x + x_t + w >= img_w:
                x_t = img_w - w - x
            if y + y_t + h >= img_h:
                y_t = img_h - h - y
        tiles.append([x, y, w, h, x+x_t, y+y_t])

    new_image = copy.deepcopy(image)
    for tile in tiles:
        x, y, w, h, x_new, y_new = tile
        new_image[x_new:x_new+w, y_new:y_new+h, :] = image[x:x+w, y:y+h, :]

    return new_image

class Augment:
    def __init__(self):
        super().__init__()

    def strong(self):

        num_holes = random.randint(1, 2)
        if num_holes == 2:
            scale = (0.02, 0.08)
        else:
            scale = (0.08, 0.2)
        
        return Compose([
            RandomChoice([
                GaussianNoise(),
                RandomInvert(p=1.0),
                ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.05
                ),
            ]),
            RandomApply([
                RandomErasing(
                    p=1.0,
                    scale=scale,
                    ratio=(0.5, 2.0),
                    value="random"
                )
                for _ in range(num_holes)
            ], p=0.9)
        ])
    
    def __call__(self, x: torch.Tensor):
        return self.strong()(x)


# Auxiliar lambda functions
move_to_cpu = lambda _Tensor : _Tensor.to('cpu')
move_to_gpu = lambda _Tensor : (_Tensor.to('cuda') 
                                if torch.cuda.is_available() 
                                else _Tensor.to('cpu')
                            )
