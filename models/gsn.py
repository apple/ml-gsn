import copy
import torch
from torchmetrics import FID
import pytorch_lightning as pl
from einops import rearrange, repeat
from torchmetrics.utilities.data import dim_zero_cat

from utils.fid import calculate_fid
from utils.utils import instantiate_from_config
from .model_utils import RenderParams, ema_accumulate

import torch.distributed as dist


class GSN(pl.LightningModule):
    def __init__(
        self,
        loss_config,
        decoder_config,
        generator_config,
        texture_net_config,
        img_res=64,
        patch_size=None,
        lr=0.002,
        ttur_ratio=1.0,
        voxel_res=32,
        voxel_size=0.6,
        **kwargs,
    ):
        super().__init__()

        self.img_res = img_res
        self.patch_size = patch_size
        self.lr = lr
        self.ttur_ratio = ttur_ratio
        self.coordinate_scale = voxel_res * voxel_size
        self.z_dim = decoder_config.params.z_dim
        self.voxel_res = voxel_res

        decoder_config.params.out_res = voxel_res

        generator_config.params.img_res = img_res
        generator_config.params.global_feat_res = voxel_res
        generator_config.params.coordinate_scale = self.coordinate_scale
        generator_config.params.nerf_mlp_config.params.z_dim = decoder_config.params.out_ch
        self.generator_config = generator_config

        texture_net_config.params.in_res = generator_config.params.nerf_out_res
        texture_net_config.params.out_res = img_res

        loss_config.params.discriminator_config.params.in_channel = 4 if loss_config.params.concat_depth else 3
        loss_config.params.discriminator_config.params.in_res = img_res

        self.decoder = instantiate_from_config(decoder_config)
        self.generator = instantiate_from_config(generator_config)
        self.texture_net = instantiate_from_config(texture_net_config)
        self.loss = instantiate_from_config(loss_config)

        self.decoder_ema = copy.deepcopy(self.decoder)
        self.generator_ema = copy.deepcopy(self.generator)
        self.texture_net_ema = copy.deepcopy(self.texture_net)

    def set_trajectory_sampler(self, trajectory_sampler):
        self.trajectory_sampler = trajectory_sampler

    def generate(self, z, camera_params):
        # camera_params should be a dict with Rt and K (if Rt is not present it will be sampled)

        nerf_out_res = self.generator_config.params.nerf_out_res
        samples_per_ray = self.generator_config.params.samples_per_ray

        # use EMA weights if in eval mode
        decoder = self.decoder if self.training else self.decoder_ema
        generator = self.generator if self.training else self.generator_ema
        texture_net = self.texture_net if self.training else self.texture_net_ema

        # map 1D latent code z to 2D latent code w
        w = decoder(z=z)

        if 'Rt' not in camera_params.keys():
            Rt = self.trajectory_sampler.sample_trajectories(self.generator, w)
            camera_params['Rt'] = Rt

        # duplicate latent codes along the trajectory dimension
        T = camera_params['Rt'].shape[1]  # trajectory length
        w = repeat(w, 'b c h w -> b t c h w', t=T)
        w = rearrange(w, 'b t c h w -> (b t) c h w')

        if self.patch_size is None:
            # compute full image in one pass
            indices_chunks = [None]
        elif nerf_out_res <= self.patch_size:
            indices_chunks = [None]
        elif nerf_out_res > self.patch_size:
            # break the whole image into manageable pieces, then compute each of those separately
            indices = torch.arange(nerf_out_res ** 2, device=z.device)
            indices_chunks = torch.chunk(indices, chunks=int(nerf_out_res ** 2 / self.patch_size ** 2))

        rgb, depth = [], []
        for indices in indices_chunks:
            render_params = RenderParams(
                Rt=rearrange(camera_params['Rt'], 'b t h w -> (b t) h w').clone(),
                K=rearrange(camera_params['K'], 'b t h w -> (b t) h w').clone(),
                samples_per_ray=samples_per_ray,
                near=self.generator_config.params.near,
                far=self.generator_config.params.far,
                alpha_noise_std=self.generator_config.params.alpha_noise_std,
                nerf_out_res=nerf_out_res,
                mask=indices,
            )

            y_hat = generator(local_latents=w, render_params=render_params)
            rgb.append(y_hat['rgb'])  # shape [BT, HW, C]
            depth.append(y_hat['depth'])

        # combine image patches back into full images
        rgb = torch.cat(rgb, dim=1)
        depth = torch.cat(depth, dim=1)

        rgb = rearrange(rgb, 'b (h w) c -> b c h w', h=nerf_out_res, w=nerf_out_res)
        rgb = texture_net(rgb)
        rgb = rearrange(rgb, '(b t) c h w -> b t c h w', t=T)

        depth = rearrange(depth, '(b t) (h w) -> b t 1 h w', t=T, h=nerf_out_res, w=nerf_out_res)

        Rt = rearrange(y_hat['Rt'], '(b t) h w -> b t h w', t=T)
        K = rearrange(y_hat['K'], '(b t) h w -> b t h w', t=T)

        return rgb, depth, Rt, K

    def on_train_batch_end(self, *args, **kwargs):
        self.update_ema()

    def update_ema(self, decay=0.999):
        ema_accumulate(self.decoder_ema, self.decoder, decay)
        ema_accumulate(self.generator_ema, self.generator, decay)
        ema_accumulate(self.texture_net_ema, self.texture_net, decay)

    def forward(self, z, camera_params):
        rgb, depth, Rt, K = self.generate(z, camera_params)
        return rgb, depth, Rt, K

    def training_step(self, x, batch_idx, optimizer_idx):
        B = x['rgb'].shape[0]

        # redraw latent codes until each rank has a unique one (otherwise each rank samples the exact same codes)
        rank = dist.get_rank()
        for i in range(rank + 1):
            z = torch.randn(B, self.z_dim, device=x['rgb'].device)

        y_rgb = rearrange(x['rgb'].clone(), 'b t c h w -> (b t) c h w')
        y_depth = rearrange(x['depth'].clone(), 'b t c h w -> (b t) c h w')

        if hasattr(self, 'trajectory_sampler'):
            del x['Rt']  # delete the given trajectory so we can sample another, otherwise reuse this one

        y_hat_rgb, y_hat_depth, _, _ = self.generate(z, camera_params=x)
        y_hat_depth = y_hat_depth / self.coordinate_scale  # scale depth so that it is roughly in [0, 1]
        y_depth = y_depth / self.coordinate_scale

        y_hat_rgb = rearrange(y_hat_rgb, 'b t c h w -> (b t) c h w')
        y_hat_depth = rearrange(y_hat_depth, 'b t c h w -> (b t) c h w')

        loss, log_dict = self.loss(y_rgb, y_hat_rgb, y_depth, y_hat_depth, self.global_step, optimizer_idx)

        for key, value in log_dict.items():
            self.log(key, value, rank_zero_only=True, prog_bar=True)
        return loss

    def validation_step(self, x, batch_idx):
        # redraw latent codes until each rank has a unique one (otherwise each rank samples the exact same codes)
        rank = dist.get_rank()
        for i in range(rank + 1):
            z = torch.randn(x['K'].shape[0], self.z_dim, device=x['K'].device)

        if hasattr(self, 'trajectory_sampler'):
            del x['Rt']  # delete the given trajectory so we can sample another, otherwise reuse this one

        rgb_fake, _, _, _ = self(z, x)
        rgb_fake = rearrange(rgb_fake, 'b t c h w -> (b t) c h w')
        rgb_real = rearrange(x['rgb'].clone(), 'b t c h w -> (b t) c h w')

        # pytorch lighting doesn't play well with FID if it doesn't have anything in the state buffers
        # so we only initialize it when we need it
        if not hasattr(self, 'fid'):
            self.fid = FID(feature=2048).cuda()
        elif batch_idx == 0:
            self.fid.reset()

        self.fid.update((rgb_real * 255).type(torch.uint8), real=True)
        self.fid.update((rgb_fake * 255).type(torch.uint8), real=False)
        return

    def validation_epoch_end(self, outputs):
        # each process stores features separately, so gather them together to calculate FID over the full distribution
        real_features = dim_zero_cat(self.fid.real_features)
        real_features_list = [torch.empty_like(real_features) for _ in range(dist.get_world_size())]
        dist.all_gather(real_features_list, real_features)
        real_features = dim_zero_cat(real_features_list)

        fake_features = dim_zero_cat(self.fid.fake_features)
        fake_features_list = [torch.empty_like(fake_features) for _ in range(dist.get_world_size())]
        dist.all_gather(fake_features_list, fake_features)
        fake_features = dim_zero_cat(fake_features_list)

        rank = dist.get_rank()
        if rank == 0:
            fid = calculate_fid(real_features, fake_features)  # returned as numpy array
            fid = torch.tensor([float(fid)], device=self.device)  # but we need a torch tensor for DDP
            print('')
            print('FID with {} samples: {}'.format(len(real_features), fid))
            print('')
        else:
            fid = torch.tensor([0.0], device=self.device)

        # share the result with all GPUs so that the checkpointing function doesn't crash
        dist.broadcast(tensor=fid, src=0)
        self.log('metrics/fid', fid, rank_zero_only=True)

    def configure_optimizers(self):
        opt_ae = torch.optim.RMSprop(
            list(self.decoder.parameters()) + list(self.generator.parameters()) + list(self.texture_net.parameters()),
            lr=self.lr,
            alpha=0.99,
            eps=1e-8,
        )
        opt_disc = torch.optim.RMSprop(
            self.loss.discriminator.parameters(), lr=self.lr * self.ttur_ratio, alpha=0.99, eps=1e-8
        )
        return [opt_ae, opt_disc]

    def on_save_checkpoint(self, checkpoint):
        # save the config if its available
        try:
            checkpoint['opt'] = self.opt
        except Exception:
            pass
