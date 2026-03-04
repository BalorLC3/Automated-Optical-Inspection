import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from torchvision.utils import make_grid

from services.generator.utils.config import GANConfig, create_logger
from services.generator.model import ConSinGAN
from services.generator.utils import functions


class TrainGenerator:
    def __init__(self, _input: torch.Tensor | str, config: GANConfig):
        self.config = config
        self._input = _input 
        
        real = functions.read_image(_input, config=config)
        self.real = functions.adjust_scales2image(real, config)

        self.reals = functions.create_reals_pyramid(self.real, config)

        # Logging for faster performance
        self.logger = create_logger()
        self.device = functions.get_device(config.device)
        self.logger.info(f"Training on image pyramid: {[r.shape for r in self.reals]}")

        # We only initialize the generator first, the discriminator is at the training loop
        self.generator = self.init_generator()

        # Empty lists
        fixed_noise = []
        noise_amp = []
    
    def init_generator(self):
        return ConSinGAN.GrowingGenerator(self.config).to(self.device).apply(ConSinGAN.weights_init)
    
    def init_discriminator(self):
        return ConSinGAN.Discriminator(self.config).to(self.device).apply(ConSinGAN.weights_init)

    def train(self):
        config = self.config
        for scale_num in range(config.stop_scale + 1):
            out_ = functions.generate_dir2save(config)
            config.outf = '%s/%d' % (out_, scale_num)
            try: 
                os.makedirs(config.outf)
            except Exception as e:
                print(f"Could not make output directory")
                pass

            functions.save_image()


class TrainHarmonization:
    def __init__(self):
        ...