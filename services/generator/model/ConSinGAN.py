import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from typing import Callable

from services.generator.utils.config import GANConfig


def weights_init(m: Callable[[torch.Tensor], torch.Tensor]) -> None:
    '''
    Weight initialization
    
    Parameters
    ----------
    m : callable - Neural network to initialize weight with normal data 
    '''
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_activation(config: GANConfig):
    '''
    Function mapping the activation of a neural network

    Parameters
    ----------
    config         : GANConfig     - must have ``config.activation`` (activation function)
    '''
    activations = {
        "lrelu": nn.ReLU(config.lrelu_alpha, inplace=True),
        "elu": nn.ELU(config.elu_alpha, inplace=True),
        "selu": nn.SELU(inplace=True)
    }
    return activations[config.activation]


def upsample(x: torch.Tensor, size: tuple[int, int]):
    return F.interpolate(
        x, 
        size=size,
        mode="bicubic",
        align_corners=True
    )

class ConvBlock(nn.Sequential):
    def __init__(
            self, 
            in_channel: int,
            out_channel: int,
            ker_size: int, 
            padding: int,
            config: GANConfig, 
            is_generator: bool = False
        ):
        super(ConvBlock, self).__init__()
        self.add_module(
            'conv',
            module=nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=ker_size,
                padding=padding
            )
        )
        if is_generator and config.batch_norm:
            self.add_module(
                'norm',
                module=nn.BatchNorm2d,
            )
        self.add_module(
            name=config.activation,
            module=get_activation
            )
        

class Discriminator(nn.Module):
    def __init__(self, config: GANConfig):
        super(Discriminator, self).__init__()
        self.config = config
        _channel = int(config.filters_per_conv)
        
        self.head = ConvBlock(
            input_channel=config.channels,
            out_channel=_channel,
            ker_size=config.kernel,
            padding=config.padding, 
            config=config
        )

        self.body = nn.Sequential()
        for i in range(config.num_layers):
            block = ConvBlock(
                in_channel=_channel,
                out_channel=_channel, 
                ker_size=config.kernel,
                padding=config.padding,
                config=config
            )
            self.body.add_module(
                f'block{i+1}', 
                block
            )
        
        self.tail = nn.Conv2d(
            in_channels=_channel,
            out_channels=1,
            kernel_size=config.kernel,
            padding=config.padding,
            config=config
        )

    def forward(self, x: torch.Tensor):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class GrowingGenerator(nn.Module):
    def __init__(self, config: GANConfig):
        super(GrowingGenerator, self).__init__()

        self.config = config
        _channels = int(config.filters_per_conv)

        self._pad = nn.ZeroPad2d(1)
        self._pad_block = nn.ZeroPad2d(
            config.num_layers - 1 if config.train_mode == 'generation' or config.train_mode == 'animation' else nn.ZeroPad2d(config.num_layers)
        )

        self.head = ConvBlock(
            in_channel=config.channels, 
            out_channel=_channels,
            ker_size=config.kernel,
            padding=config.padding,
            config=config,
            is_generator=True
        )

        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(config.num_layers):
            block = ConvBlock(
                in_channel=_channels,
                out_channel=_channels,
                ker_size=config.kernel,
                padding=config.padding,
                config=config,
                is_generator=True
            )
            _first_stage.add_module(f'block{i+1}', block)
        self.body.append(_first_stage)
        
        self.tail = nn.Sequential(
            nn.Conv2d(
                in_channels=_channels,
                kernel_size=config.kernel,
                padding=config.padding,
            ),
            nn.Tanh()
        )

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(
            self, 
            noise,
            real_shapes,
            noise_amp 
        ):
        config = self.config
        x = self.head(self._pad(noise[0]))
        # Do some upsampling for training models for diversity

        if config.train_mode == "generation" or config.train_mode == "animation":
            x = upsample(
                x, 
                size=[x.shape[2] + 2, x.shape[3] + 2]
            )
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)

        for idx, block in enumerate(self.boddy[1:], 1):
            if config.train_mode == "generation" or config.train_mode == "animation":
                x_prev_out_1 = upsample(
                    x_prev_out, 
                    size=[real_shapes[idx][2], real_shapes[idx][3]]
                )
                x_prev_out_2 = upsample(
                    x_prev_out, 
                    size=[real_shapes[idx][2] + config.num_layers*2, 
                          real_shapes[idx][3] + config.num_layers*2]
                )
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
            else: 
                x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(
                    self._pad_block(x_prev_out + noise[idx] * noise_amp[idx])
                )
            x_prev_out = x_prev + x_prev_out_1
        
        out = self.tail(self._pad(x_prev_out))
        return out

        