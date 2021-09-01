import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from .diff_augment import DiffAugment
from utils.utils import instantiate_from_config


def hinge_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = torch.mean(F.relu(1.0 + fake_pred))
        d_loss_real = torch.mean(F.relu(1.0 - real_pred))
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = -torch.mean(fake_pred)
    return d_loss


def logistic_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = torch.mean(F.softplus(fake_pred))
        d_loss_real = torch.mean(F.softplus(-real_pred))
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = torch.mean(F.softplus(-fake_pred))
    return d_loss


def r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


class GSNLoss(nn.Module):
    def __init__(
        self,
        discriminator_config,
        disc_loss='logistic',
        concat_depth=True,
        recon_weight=1000,
        r1_weight=0.01,
        d_reg_every=16,
        aug_policy='',
    ):
        super().__init__()

        discriminator_config.params.in_channel = 4 if concat_depth else 3

        self.discriminator = instantiate_from_config(discriminator_config)

        if disc_loss == 'hinge':
            self.disc_loss = hinge_loss
        elif disc_loss == 'logistic':
            self.disc_loss = logistic_loss

        self.concat_depth = concat_depth
        self.recon_weight = recon_weight
        self.r1_weight = r1_weight
        self.d_reg_every = d_reg_every
        self.aug_policy = aug_policy

    def forward(self, rgb_real, rgb_fake, depth_real, depth_fake, global_step, optimizer_idx):
        rgb_real.requires_grad = True  # for R1 gradient penalty

        if self.concat_depth:
            if depth_fake.shape[-1] != depth_real.shape[-1]:
                # downscale real depth so it doesn't have more details than fake depth
                depth_real = F.interpolate(depth_real, size=depth_fake.shape[-1], mode='bilinear', align_corners=False)
                # then resize both depth back up to match RGB res
                depth_real = F.interpolate(depth_real, size=rgb_real.shape[-1], mode='bilinear', align_corners=False)
                depth_fake = F.interpolate(depth_fake, size=rgb_real.shape[-1], mode='bilinear', align_corners=False)

            disc_in_real = torch.cat([rgb_real, depth_real], dim=1)
            disc_in_fake = torch.cat([rgb_fake, depth_fake], dim=1)
        else:
            disc_in_real = rgb_real
            disc_in_fake = rgb_fake

        if self.aug_policy:
            disc_in_real = DiffAugment(disc_in_real, normalize=True, policy=self.aug_policy)
            disc_in_fake = DiffAugment(disc_in_fake, normalize=True, policy=self.aug_policy)

        if optimizer_idx == 0:  # optimize generator
            logits_fake, _ = self.discriminator(disc_in_fake)
            g_loss = self.disc_loss(logits_fake, None, mode='g')

            log = {"loss_train/g_loss": g_loss.detach()}

            return g_loss, log

        if optimizer_idx == 1:  # optimize discriminator
            logits_real, recon_real = self.discriminator(disc_in_real)
            logits_fake, _ = self.discriminator(disc_in_fake.detach())

            disc_loss = self.disc_loss(fake_pred=logits_fake, real_pred=logits_real, mode='d')

            disc_recon_loss = F.mse_loss(disc_in_real, recon_real) * self.recon_weight

            # lazy regularization so we don't need to compute grad penalty every iteration
            if (global_step % self.d_reg_every == 0) and self.r1_weight > 0:
                grad_penalty = r1_loss(logits_real, rgb_real)

                # the 0 * logits_real is to trigger DDP allgather
                # https://github.com/rosinality/stylegan2-pytorch/issues/76
                grad_penalty = self.r1_weight / 2 * grad_penalty * self.d_reg_every + (0 * logits_real.sum())
            else:
                grad_penalty = torch.tensor(0.0)

            d_loss = disc_loss + disc_recon_loss + grad_penalty

            log = {
                "loss_train/disc_loss": disc_loss.detach(),
                "loss_train/disc_recon_loss": disc_recon_loss.detach(),
                "loss_train/r1_loss": grad_penalty.detach(),
                "loss_train/d_loss": d_loss.detach(),
                "loss_train/logits_real": logits_real.mean().detach(),
                "loss_train/logits_fake": logits_fake.mean().detach(),
            }

            return d_loss, log
