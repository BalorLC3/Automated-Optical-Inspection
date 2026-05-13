"""
third_party/ConSinGAN/train.py
────────────────────────────────────────────────────────────────────────────────
Complete TrainGenerator — the 70% stub from services/generator/src/train.py,
fully implemented.

Bugs fixed from original:
  - fixed_noise and noise_amp were local variables, never assigned to self
  - train() body was a 3-line stub with placeholder calls
  - Discriminator was initialised inside train() with no reference kept

Added:
  - Full WGAN-GP training loop across all pyramid scales
  - Concurrent training: only the last `train_depth` generator stages are
    unfrozen at each scale (the core ConSinGAN contribution)
  - noise_amp calibration after each scale (RMSE-based)
  - Cosine-annealing LR schedulers for G and D
  - Periodic checkpoint saving + best-RMSE tracking
  - generate() method for post-training sampling (used by the augmenter)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from generator.utils.config import GANConfig, create_logger
from generator.utils import functions
from generator.model.consingan import Grow

from third_party.ConSinGAN.model import GrowingGenerator, Discriminator, weights_init


class TrainGenerator:
    def __init__(self, _input: torch.Tensor | str, config: GANConfig):
        self.config = config
        self.logger = create_logger()
        self.device = functions.get_device(config)

        # Load and pre-process image
        if isinstance(_input, str):
            real = functions.read_image(_input, config)
        else:
            real = _input.to(self.device)

        real = functions.adjust_scales2image(real, config)
        self.real  = real
        self.reals = functions.create_reals_pyramid(real, config)

        self.logger.info(
            f"Device: {self.device} | "
            f"Pyramid: {[tuple(r.shape[2:]) for r in self.reals]} | "
            f"scale_factor: {config.scale_factor:.3f}"
        )

        # Generator (grows one stage per scale)
        self.generator = self._init_generator()

        # These MUST be self. — they were local vars in the original, a bug
        self.fixed_noise: list[torch.Tensor] = []
        self.noise_amp:   list[float]        = []


    def _init_generator(self) -> GrowingGenerator:
        return (
            GrowingGenerator(self.config)
            .to(self.device)
            .apply(weights_init)
        )

    def _init_discriminator(self) -> Discriminator:
        return (
            Discriminator(self.config)
            .to(self.device)
            .apply(weights_init)
        )

    def _make_optimizer(
        self, params, lr: float
    ) -> tuple[optim.Adam, optim.lr_scheduler.CosineAnnealingLR]:
        opt = optim.Adam(params, lr=lr, betas=(self.config.beta1, 0.999))
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config.max_epochs, eta_min=lr * self.config.lr_scale
        )
        return opt, sched

    #  Concurrent training: freeze early stages 
    def _set_trainable_stages(self, scale_num: int) -> None:
        """
        Unfreeze only the last `train_depth` stages of the generator body.
        Head and tail are always trainable.
        """
        first_trainable = max(0, scale_num - self.config.train_depth + 1)
        for i, stage in enumerate(self.generator.body):
            requires = (i >= first_trainable)
            for p in stage.parameters():
                p.requires_grad_(requires)
        # Head and tail always trained
        for p in self.generator.head.parameters():
            p.requires_grad_(True)
        for p in self.generator.tail.parameters():
            p.requires_grad_(True)


    def _draw_generation_noise(self, scale_num: int) -> list[torch.Tensor]:
        return functions.draw_concat(
            self.fixed_noise, self.noise_amp, self.generator,
            self.reals, scale_num, self.config, self.device, mode="rand",
        )

    def _draw_reconstruction_noise(self, scale_num: int) -> list[torch.Tensor]:
        return functions.draw_concat(
            self.fixed_noise, self.noise_amp, self.generator,
            self.reals, scale_num, self.config, self.device, mode="rec",
        )


    def _calibrate_noise_amp(self, scale_num: int) -> float:
        """
        Set noise_amp[scale_num] = noise_amp_init * RMSE(G_rec, real).
        At scale 0 the amplitude is always 1.0 (coarsest scale drives texture).
        """
        if scale_num == 0:
            return 1.0

        real = self.reals[scale_num]
        real_shapes = [r.shape for r in self.reals[: scale_num + 1]]

        # Temporarily append 1.0 so the forward pass can run
        self.noise_amp.append(1.0)
        rec_noise = self._draw_reconstruction_noise(scale_num)
        with torch.no_grad():
            fake_rec = self.generator(rec_noise, real_shapes, self.noise_amp)
        self.noise_amp.pop()   # remove the placeholder

        rmse = torch.sqrt(nn.MSELoss()(fake_rec, real)).item()
        return self.config.noise_amp_init * rmse


    def train(self) -> tuple[GrowingGenerator, list[torch.Tensor], list[float]]:
        """
        Train the ConSinGAN pyramid stage by stage.

        Returns
        -------
        (generator, fixed_noise, noise_amp)
        """
        config   = self.config
        rec_loss = nn.MSELoss()

        for scale_num in range(config.stop_scale + 1):
            real        = self.reals[scale_num].to(self.device)
            real_shapes = [r.shape for r in self.reals[: scale_num + 1]]

            if scale_num > 0:
                self.generator.init_next_stage()
            D = self._init_discriminator()

            self._set_trainable_stages(scale_num)

            trainable_g = [p for p in self.generator.parameters() if p.requires_grad]
            opt_g, sched_g = self._make_optimizer(trainable_g, config.lr_g)
            opt_d, sched_d = self._make_optimizer(D.parameters(),  config.lr_d)

            # Drawn once, never changed — defines the reconstruction target
            fixed_z = functions.generate_spatial_noise(real, self.device)
            self.fixed_noise.append(fixed_z)

            # Calibrate noise amplitude
            amp = self._calibrate_noise_amp(scale_num)
            self.noise_amp.append(amp)

            self.logger.info(
                f"Scale {scale_num}/{config.stop_scale}  "
                f"size={tuple(real.shape[2:])}  amp={amp:.4f}  "
                f"trainable_stages={list(range(max(0, scale_num - config.train_depth + 1), scale_num + 1))}"
            )

            best_rmse = float("inf")

            for epoch in range(config.max_epochs):

                for _ in range(config.discriminator_steps):
                    D.zero_grad()

                    # Real branch
                    real_prob  = D(real)
                    loss_d_real = -real_prob.mean()
                    loss_d_real.backward(retain_graph=True)

                    # Fake branch
                    gen_noise  = self._draw_generation_noise(scale_num)
                    fake       = self.generator(gen_noise, real_shapes, self.noise_amp).detach()
                    fake_prob  = D(fake)
                    loss_d_fake = fake_prob.mean()
                    loss_d_fake.backward()

                    # WGAN-GP
                    gp = functions.calc_gradient_penalty(
                        D, real, fake, self.device, config.lambda_grad
                    )
                    gp.backward()
                    opt_d.step()

                for _ in range(config.generator_steps):
                    self.generator.zero_grad()

                    # Adversarial loss
                    gen_noise  = self._draw_generation_noise(scale_num)
                    fake       = self.generator(gen_noise, real_shapes, self.noise_amp)
                    adv_loss   = -D(fake).mean()

                    # Reconstruction loss
                    rec_noise  = self._draw_reconstruction_noise(scale_num)
                    fake_rec   = self.generator(rec_noise, real_shapes, self.noise_amp)
                    recon_loss = config.alpha * rec_loss(fake_rec, real)

                    loss_g = adv_loss + recon_loss
                    loss_g.backward()
                    opt_g.step()

                sched_g.step()
                sched_d.step()

                if (epoch + 1) % 200 == 0 or epoch == 0:
                    with torch.no_grad():
                        rmse = torch.sqrt(rec_loss(fake_rec.detach(), real)).item()
                    self.logger.info(
                        f"  epoch {epoch + 1:4d}/{config.max_epochs}  "
                        f"D={loss_d_real.item() + loss_d_fake.item():.4f}  "
                        f"G={adv_loss.item():.4f}  rec={recon_loss.item():.4f}  "
                        f"RMSE={rmse:.4f}"
                    )
                    if rmse < best_rmse:
                        best_rmse = rmse
                        self._save_checkpoint(scale_num, D)

        self.logger.info("Training complete.")
        return self.generator, self.fixed_noise, self.noise_amp


    def _save_checkpoint(self, scale_num: int, D: Discriminator) -> None:
        out = Path(self.config.outf) / str(scale_num)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.generator.state_dict(), out / "generator.pt")
        torch.save(D.state_dict(),              out / "discriminator.pt")
        torch.save(self.fixed_noise,            out / "fixed_noise.pt")
        torch.save(self.noise_amp,              out / "noise_amp.pt")


    def generate(self, n_samples: int = 1) -> list[torch.Tensor]:
        """
        Sample `n_samples` synthetic images from the trained pyramid.
        Must be called after train().
        """
        if not self.fixed_noise:
            raise RuntimeError("Call train() before generate().")
        return functions.sample_from_generator(
            self.generator,
            self.fixed_noise,
            self.noise_amp,
            self.reals,
            self.config,
            self.device,
            n_samples=n_samples,
        )