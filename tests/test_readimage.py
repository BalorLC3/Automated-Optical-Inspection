from services.generator.utils.functions import (
    decode_image, 
    rgb_to_grayscale,
    norm,
)
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from services.generator.utils.config import GANConfig

from pathlib import Path

def read_image(path: str, config: GANConfig):
    x = decode_image(path)

    if config.channels == 3:
        x = x[:3]
    else:
        x = rgb_to_grayscale(x)

    x = x.unsqueeze(0).float() / 255.0
    x = norm(x)
    return x


def binary_dilation_torch(mask, radius: int):
    kernel_size = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size, stride=1, padding=radius)


def dilate_mask(mask, config: GANConfig):
    radius = 7 if config.train_mode == "harmonization" else 20

    mask = mask[:, :1]  # (1,1,H,W)

    mask = binary_dilation_torch(mask, radius)
    mask = gaussian_blur(mask, kernel_size=11, sigma=5)

    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask


ROOT = Path(__file__).resolve().parents[1]
config = GANConfig(channels=3)

path = ROOT / "tests" / "Chin_posing.jpg"
x = read_image(path, config)

print("read_image:")
print(" shape:", x.shape)
print(" dtype:", x.dtype)
print(" min/max:", x.min().item(), x.max().item())