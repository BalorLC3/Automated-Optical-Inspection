from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from generator.utils.config import GANConfig



def weights_init(m: nn.Module) -> None:
    class_name = m.__class__.__name__
    if class_name.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find("Norm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def get_activation(config: GANConfig) -> nn.Module:
    activations = {
        "relu":  nn.ReLU(inplace=True),
        "lrelu": nn.LeakyReLU(config.lrelu_alpha, inplace=True),  # was nn.ReLU
        "elu":   nn.ELU(config.elu_alpha, inplace=True),
        "selu":  nn.SELU(inplace=True),
    }
    key = config.activation
    if key not in activations:
        raise ValueError(f"Unknown activation '{key}'. Choose from {list(activations)}")
    return activations[key]


def upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bicubic", align_corners=True)



class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        ker_size: int,
        padding: int,
        config: GANConfig,
        is_generator: bool = False,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=ker_size,
                padding=padding,
            ),
        )
        if is_generator and config.batch_norm:
            self.add_module(
                "norm",
                nn.BatchNorm2d(out_channel),        
            )
        self.add_module(
            config.activation,
            get_activation(config),                 
        )



class Discriminator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config
        _ch = config.filters_per_conv

        self.head = ConvBlock(
            in_channel=config.channels,             
            out_channel=_ch,
            ker_size=config.kernel,
            padding=config.padding,
            config=config,
        )

        self.body = nn.Sequential()
        for i in range(config.num_layers - 1):
            block = ConvBlock(
                in_channel=_ch,
                out_channel=_ch,
                ker_size=config.kernel,
                padding=config.padding,
                config=config,
            )
            self.body.add_module(f"block{i + 1}", block)

        self.tail = nn.Conv2d(
            in_channels=_ch,
            out_channels=1,
            kernel_size=config.kernel,
            padding=config.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        return self.tail(x)



class GrowingGenerator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config
        _ch = config.filters_per_conv

        self._pad = nn.ZeroPad2d(1)

        pad_size = (
            config.num_layers - 1
            if config.train_mode in ("generation", "animation")
            else config.num_layers
        )
        self._pad_block = nn.ZeroPad2d(pad_size)

        self.head = ConvBlock(
            in_channel=config.channels,
            out_channel=_ch,
            ker_size=config.kernel,
            padding=config.padding,
            config=config,
            is_generator=True,
        )

        self.body = nn.ModuleList()
        first_stage = nn.Sequential()
        for i in range(config.num_layers - 1):
            block = ConvBlock(
                in_channel=_ch,
                out_channel=_ch,
                ker_size=config.kernel,
                padding=config.padding,
                config=config,
                is_generator=True,
            )
            first_stage.add_module(f"block{i + 1}", block)
        self.body.append(first_stage)

        self.tail = nn.Sequential(
            nn.Conv2d(
                in_channels=_ch,
                out_channels=config.channels,        
                kernel_size=config.kernel,
                padding=config.padding,
            ),
            nn.Tanh(),
        )

    def init_next_stage(self) -> None:
        """Grow the generator by one scale: clone the last body stage."""
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(
        self,
        noise: list[torch.Tensor],
        real_shapes: list[torch.Size],
        noise_amp: list[float],
    ) -> torch.Tensor:
        config = self.config
        x = self.head(self._pad(noise[0]))

        if config.train_mode in ("generation", "animation"):
            x = upsample(x, size=(x.shape[2] + 2, x.shape[3] + 2))

        x = self._pad_block(x)
        x_prev_out = self.body[0](x)

        for idx, block in enumerate(self.body[1:], 1):   
            if config.train_mode in ("generation", "animation"):
                x_prev_out_1 = upsample(
                    x_prev_out,
                    size=(real_shapes[idx][2], real_shapes[idx][3]),
                )
                x_prev_out_2 = upsample(
                    x_prev_out,
                    size=(
                        real_shapes[idx][2] + config.num_layers * 2,
                        real_shapes[idx][3] + config.num_layers * 2,
                    ),
                )
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
            else:
                x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(
                    self._pad_block(x_prev_out + noise[idx] * noise_amp[idx])
                )
            x_prev_out = x_prev + x_prev_out_1

        return self.tail(self._pad(x_prev_out))