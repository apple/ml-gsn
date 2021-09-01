import math
import numpy as np
from einops import repeat

import torch
from torch import nn

from .layers import *
from utils.utils import instantiate_from_config
from .nerf_utils import get_sample_points, volume_render_radiance_field, sample_pdf_2


class StyleGenerator2D(nn.Module):
    def __init__(self, out_res, out_ch, z_dim, ch_mul=1, ch_max=512, skip_conn=True):
        super().__init__()

        self.skip_conn = skip_conn

        # dict key is the resolution, value is the number of channels
        # a trend in both StyleGAN and BigGAN is to use a constant number of channels until 32x32
        self.channels = {
            4: ch_max,
            8: ch_max,
            16: ch_max,
            32: ch_max,
            64: (ch_max // 2 ** 1) * ch_mul,
            128: (ch_max // 2 ** 2) * ch_mul,
            256: (ch_max // 2 ** 3) * ch_mul,
            512: (ch_max // 2 ** 4) * ch_mul,
            1024: (ch_max // 2 ** 5) * ch_mul,
        }

        self.latent_normalization = PixelNorm()
        self.mapping_network = []
        for i in range(3):
            self.mapping_network.append(EqualLinear(in_channel=z_dim, out_channel=z_dim, lr_mul=0.01, activate=True))
        self.mapping_network = nn.Sequential(*self.mapping_network)

        log_size_in = int(math.log(4, 2))  # 4x4
        log_size_out = int(math.log(out_res, 2))

        self.input = ConstantInput(channel=self.channels[4])

        self.conv1 = ModulatedConv2d(
            in_channel=self.channels[4],
            out_channel=self.channels[4],
            kernel_size=3,
            z_dim=z_dim,
            upsample=False,
            activate=True,
        )

        if self.skip_conn:
            self.to_rgb1 = ToRGB(in_channel=self.channels[4], out_channel=out_ch, z_dim=z_dim, upsample=False)
            self.to_rgbs = nn.ModuleList()

        self.convs = nn.ModuleList()

        in_channel = self.channels[4]
        for i in range(log_size_in + 1, log_size_out + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                ModulatedConv2d(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=3,
                    z_dim=z_dim,
                    upsample=True,
                    activate=True,
                )
            )

            self.convs.append(
                ModulatedConv2d(
                    in_channel=out_channel,
                    out_channel=out_channel,
                    kernel_size=3,
                    z_dim=z_dim,
                    upsample=False,
                    activate=True,
                )
            )

            if self.skip_conn:
                self.to_rgbs.append(ToRGB(in_channel=out_channel, out_channel=out_ch, z_dim=z_dim, upsample=True))

            in_channel = out_channel

        # if not accumulating with skip connections we need final layer to map to out_ch channels
        if not self.skip_conn:
            self.out_rgb = ToRGB(in_channel=out_channel, out_channel=out_ch, z_dim=z_dim, upsample=False)
            self.to_rgbs = [None] * (log_size_out - log_size_in)  # dummy for easier control flow

        if self.skip_conn:
            self.n_layers = len(self.convs) + len(self.to_rgbs) + 2
        else:
            self.n_layers = len(self.convs) + 2

    def process_latents(self, z):
        # output should be list with separate latent code for each conditional layer in the model

        if isinstance(z, list):  # latents already in proper format
            pass
        elif z.ndim == 2:  # standard training, shape [B, ch]
            z = self.latent_normalization(z)
            z = self.mapping_network(z)
            z = [z] * self.n_layers
        elif z.ndim == 3:  # latent optimization, shape [B, n_latent_layers, ch]
            n_latents = z.shape[1]
            z = [self.latent_normalization(self.mapping_network(z[:, i])) for i in range(n_latents)]
        return z

    def forward(self, z):
        z = self.process_latents(z)

        out = self.input(z[0])
        B = out.shape[0]
        out = out.view(B, -1, 4, 4)

        out = self.conv1(out, z[0])

        if self.skip_conn:
            skip = self.to_rgb1(out, z[1])
            i = 2
        else:
            i = 1

        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            out = conv1(out, z[i])
            out = conv2(out, z[i + 1])

            if self.skip_conn:
                skip = to_rgb(out, z[i + 2], skip)
                i += 3
            else:
                i += 2

        if not self.skip_conn:
            skip = self.out_rgb(out, z[i])
        return skip


class NerfStyleGenerator(nn.Module):
    """NeRF MLP with style modulation.

    This module maps input latent codes, coordinates, and viewing directions to alpha values and an output
    feature vector. Conventionally the output is a 3 dimensional RGB colour vector, but more general features can
    be output to be used for downstream upsampling and refinement.

    Note that skip connections are important for training any model with more than 4 layers, as shown by DeepSDF.

    Args:
    ----
    n_layers: int
        Number of layers in the MLP (excluding those for predicting alpha and the feature output.)
    channels: int
        Channels per layer.
    out_channel: int
        Output channels.
    z_dim: int
        Dimension of latent code.
    omega_coord: int
        Number of frequency bands to use for coordinate positional encoding.
    omega_dir: int
        Number of frequency bands to use for view direction positional encoding.
    skips: list
        Layers at which to apply skip connections. Coordinates will be concatenated to feature inputs at these
        layers.

    """

    def __init__(self, n_layers=8, channels=256, out_channel=3, z_dim=128, omega_coord=10, omega_dir=4, skips=[4]):
        super().__init__()

        self.skips = skips

        self.from_coords = PositionalEncoding(in_dim=3, frequency_bands=omega_coord)
        self.from_dirs = PositionalEncoding(in_dim=3, frequency_bands=omega_dir)
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [ModulationLinear(in_channel=self.from_coords.out_dim, out_channel=channels, z_dim=z_dim)]
        )

        for i in range(1, n_layers):
            if i in skips:
                in_channels = channels + self.from_coords.out_dim
            else:
                in_channels = channels
            self.layers.append(ModulationLinear(in_channel=in_channels, out_channel=channels, z_dim=z_dim))

        self.fc_alpha = ModulationLinear(
            in_channel=channels, out_channel=1, z_dim=z_dim, demodulate=False, activate=False
        )
        self.fc_feat = ModulationLinear(in_channel=channels, out_channel=channels, z_dim=z_dim)
        self.fc_viewdir = ModulationLinear(
            in_channel=channels + self.from_dirs.out_dim, out_channel=channels, z_dim=z_dim
        )
        self.fc_out = ModulationLinear(
            in_channel=channels, out_channel=out_channel, z_dim=z_dim, demodulate=False, activate=False
        )

    def process_latents(self, z):
        # output should be list with separate latent code for each conditional layer in the model
        # should be a list

        if isinstance(z, list):  # latents already in proper format
            pass
        elif z.ndim == 2:  # standard training, shape [B, ch]
            z = [z] * (self.n_layers + 4)
        elif z.ndim == 3:  # latent optimization, shape [B, n_latent_layers, ch]
            n_latents = z.shape[1]
            z = [z[:, i] for i in range(n_latents)]
        return z

    def forward(self, z, coords, viewdirs=None):
        """Forward pass.

        Input:
        -----
        z: torch.Tensor
            Latent codes of shape [B, z_dim].
        coords: torch.Tensor
            Spatial coordinates of shape [B, 3].
        viewdirs: torch.Tensor
            View directions of shape [B, 3].

        Return:
        ------
        out: torch.Tensor
            RGB pixels or feature vectors of shape [B, out_channel].
        alpha: torch.Tensor
            Occupancy values of shape [B, 1].

        """
        coords = self.from_coords(coords)
        z = self.process_latents(z)

        h = coords
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, coords], dim=-1)

            h = layer(h, z[i])

        alpha = self.fc_alpha(h, z[i + 1])

        if viewdirs is None:
            return None, alpha

        h = self.fc_feat(h, z[i + 2])

        viewdirs = self.from_dirs(viewdirs)
        h = torch.cat([h, viewdirs], dim=-1)

        h = self.fc_viewdir(h, z[i + 3])
        out = self.fc_out(h, z[i + 4])

        return out, alpha


class NerfSimpleGenerator(nn.Module):
    """NeRF MLP with with standard latent concatenation.

    This is essentially the generator architecture used in GRAF: https://arxiv.org/abs/2007.02442

    This module maps input latent codes, coordinates, and viewing directions to alpha values and an output
    feature vector. Conventionally the output is a 3 dimensional RGB colour vector, but more general features can
    be output to be used for downstream upsampling and refinement.

    Note that skip connections are important for training any model with more than 4 layers, as shown by DeepSDF.

    Args:
    ----
    n_layers: int
        Number of layers in the MLP (excluding those for predicting alpha and the feature output.)
    channels: int
        Channels per layer.
    out_channel: int
        Output channels.
    z_dim: int
        Dimension of latent code.
    omega_coord: int
        Number of frequency bands to use for coordinate positional encoding.
    omega_dir: int
        Number of frequency bands to use for view direction positional encoding.
    skips: list
        Layers at which to apply skip connections. Coordinates will be concatenated to feature inputs at these
        layers.

    """

    def __init__(self, n_layers=8, channels=256, out_channel=3, z_dim=128, omega_coord=10, omega_dir=4, skips=[4]):
        super().__init__()

        self.skips = skips

        self.from_coords = PositionalEncoding(in_dim=3, frequency_bands=omega_coord)
        self.from_dirs = PositionalEncoding(in_dim=3, frequency_bands=omega_dir)

        self.layers = nn.ModuleList(
            [EqualLinear(in_channel=self.from_coords.out_dim + z_dim, out_channel=channels, activate=True)]
        )

        for i in range(1, n_layers):
            if i in skips:
                in_channels = channels + self.from_coords.out_dim
            else:
                in_channels = channels
            self.layers.append(EqualLinear(in_channel=in_channels, out_channel=channels, activate=True))

        self.fc_alpha = EqualLinear(in_channel=channels, out_channel=1, activate=False)
        self.fc_feat = EqualLinear(in_channel=channels, out_channel=channels, activate=True)
        self.fc_viewdir = EqualLinear(in_channel=channels + self.from_dirs.out_dim, out_channel=channels, activate=True)
        self.fc_out = EqualLinear(in_channel=channels, out_channel=out_channel, activate=False)

    def forward(self, z, coords, viewdirs=None):
        """Forward pass.

        Input:
        -----
        z: torch.Tensor
            Latent codes of shape [B, z_dim].
        coords: torch.Tensor
            Spatial coordinates of shape [B, 3].
        viewdirs: torch.Tensor
            View directions of shape [B, 3].

        Return:
        ------
        out: torch.Tensor
            RGB pixels or feature vectors of shape [B, out_channel].
        alpha: torch.Tensor
            Occupancy values of shape [B, 1].

        """
        coords = self.from_coords(coords)
        h = torch.cat([z, coords], dim=1)

        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, coords], dim=-1)

            h = layer(h)

        alpha = self.fc_alpha(h)

        if viewdirs is None:
            return None, alpha

        h = self.fc_feat(h)

        viewdirs = self.from_dirs(viewdirs)
        h = torch.cat([h, viewdirs], dim=-1)

        h = self.fc_viewdir(h)
        out = self.fc_out(h)

        return out, alpha


class RenderNet2d(nn.Module):
    """2D rendering refinement module.

    Inspired by GIRAFFE: https://arxiv.org/abs/2011.12100

    This module takes as input a set of feature maps and upsamples them to higher resolution RGB outputs. Skips
    connections are used to aggregate RGB outputs at each layer for more stable training.

    Args:
    ----
    in_channel: int
        Input channels.
    in_res: int
        Input resolution.
    out_res: int
        Output resolution.
    mode: str
        Which mode to use for the render block. Options are 'original' and 'blur'. Original mode is implemented
        as described in the GIRAFFE paper using nearest neighbour upsampling + conv. Blur mode is closer to the
        skip generator in StyleGAN2, and uses transposed convolution with stride for upsampling.
    deep: bool
        Each block in the rendering network uses two convolutional layers. Otherwise, a single 3x3 conv is used
        per resolution.

    """

    def __init__(self, in_channel, in_res, out_res, mode='blur', deep=False, **kwargs):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))

        self.render_blocks = nn.ModuleList()
        for i in range(log_size_out - log_size_in):
            self.render_blocks.append(
                ConvRenderBlock2d(in_channel=in_channel, out_channel=in_channel // 2, mode=mode, deep=deep)
            )
            in_channel = in_channel // 2

    def forward(self, x):
        """Forward pass.

        Input:
        -----
        x: torch.Tensor
            Input feature maps of shape [B, in_channel, in_res, in_res].

        Return:
        ------
        rgb: torch.Tensor
            RGB images of shape [B, 3, out_res, out_res].

        """
        rgb = None
        for block in self.render_blocks:
            x, rgb = block(x, rgb)

        rgb = torch.sigmoid(rgb)
        return rgb


class SceneGenerator(nn.Module):
    """NeRF scene generator.

    Args:
    ----
    nerf_mlp_config: OmegaConf config, or dict
        Config for the radiance field MLP.
    img_res: int
        Image resolution of the final generated image.
    feature_nerf: bool
        If True, NeRF outputs features which are refined with a render network. If False, NeRF outputs standard
        RGB.
    global_feat_res: int
        Resolution of the local latent code grid produced by the global generator. If global_feat_res=0 then
        the global generator is not used.
    coordinate_scale: float
        The length, in real world units, of the maximum distance that can be covered by a single trajectory. Used
        to normalize the coordinate frame to [-1, 1].
    alpha_activation: str
        Activation function to use to rectify alpha predictions. Options are 'relu' and 'softplus'. ReLU
        activation is used in the original NeRF formulation. Softplus was introduced in Deformable NeRF, and
        could potentially allow for better gradients in unoccupied regions (which otherwise receive no gradient).
        This could be an alternative to adding noise to the alpha values at the beginning of training.
    local_coordinates: bool
        Convert global coordinates to local coordinates.
    hierarchical_sampling: bool
        Perform two passes through the NeRF MLP, where the results of the first pass are used to inform ray
        sampling in the second pass.
    density_bias: float
        Adds a value to all occupancy predictions as in Mip-NeRF (https://arxiv.org/abs/2103.13415). Can speed up
        training by starting the model at a better initialization.

    """

    def __init__(
        self,
        nerf_mlp_config,
        img_res=None,
        feature_nerf=False,
        global_feat_res=16,
        coordinate_scale=None,
        alpha_activation='softplus',
        local_coordinates=True,
        hierarchical_sampling=False,
        density_bias=0,
        **kwargs
    ):
        super().__init__()

        self.img_res = img_res
        self.feature_nerf = feature_nerf
        self.global_feat_res = global_feat_res
        self.coordinate_scale = coordinate_scale
        self.alpha_activation = alpha_activation
        self.local_coordinates = local_coordinates
        self.hierarchical_sampling = hierarchical_sampling
        self.density_bias = density_bias
        self.out_dim = nerf_mlp_config.params.out_channel

        self.local_generator = instantiate_from_config(nerf_mlp_config)

    def get_local_coordinates(self, global_coords, local_grid_length, preserve_y=True):
        local_coords = global_coords.clone()
        # it is assumed that the global coordinates are scaled to [-1, 1]
        # convert to [0, 1] scale
        local_coords = (local_coords + 1) / 2
        # scale so that each grid cell in the local_latent grid is 1x1 in size
        local_coords = local_coords * local_grid_length
        # subtract integer from each coordinate so that they are all in range [0, 1]
        local_coords = local_coords - (local_coords - 0.5).round()
        # return to [-1, 1] scale
        local_coords = (local_coords * 2) - 1

        if preserve_y:
            # preserve the y dimension in the global coordinate frame, since it doesn't have a local latent code
            coords = torch.cat([local_coords[..., 0:1], global_coords[..., 1:2], local_coords[..., 2:3]], dim=-1)
        else:
            coords = torch.cat([local_coords[..., 0:1], local_coords[..., 1:2], local_coords[..., 2:3]], dim=-1)
        return coords

    def sample_local_latents(self, local_latents, xyz):
        if local_latents.ndim == 4:
            B, local_z_dim, H, W = local_latents.shape
            # take only x and z coordinates, since our latent codes are in a 2D grid (no y dimension)
            # for the purposes of grid_sample we treat H*W as the H dimension and samples_per_ray as the W dimension
            xyz = xyz[:, :, :, [0, 2]]  # [B, H * W, samples_per_ray, 2]
        elif local_latents.ndim == 5:
            B, local_z_dim, D, H, W = local_latents.shape

            B, HW, samples_per_ray, _ = xyz.shape
            H = int(np.sqrt(HW))
            xyz = xyz.view(B, H, H, samples_per_ray, 3)

        samples_per_ray = xyz.shape[2]

        # all samples get the most detailed latent codes
        sampled_local_latents = nn.functional.grid_sample(
            input=local_latents,
            grid=xyz,
            mode='bilinear',  # bilinear mode will use trilinear interpolation if input is 5D
            align_corners=False,
            padding_mode="zeros",
        )
        # output is shape [B, local_z_dim, H * W, samples_per_ray]

        if local_latents.ndim == 4:
            # put channel dimension at end: [B, H * W, samples_per_ray, local_z_dim]
            sampled_local_latents = sampled_local_latents.permute(0, 2, 3, 1)
        elif local_latents.ndim == 5:
            sampled_local_latents = sampled_local_latents.permute(0, 2, 3, 4, 1)

        # merge everything else into batch dim: [B * H * W * samples_per_ray, local_z_dim]
        sampled_local_latents = sampled_local_latents.reshape(-1, local_z_dim)

        return sampled_local_latents, local_latents

    def query_network(self, xyz, local_latents, viewdirs):
        if self.coordinate_scale is not None:
            # this tries to get all input coordinates to lie within [-1, 1]
            xyz = xyz / (self.coordinate_scale / 2)

        B, n_samples, samples_per_ray, _ = xyz.shape  # n_samples = H * W
        sampled_local_latents, local_latents = self.sample_local_latents(local_latents, xyz=xyz)

        if self.local_coordinates:
            # map global coordinate space into local coordinate space (i.e. each grid cell has a [-1, 1] range)
            preserve_y = local_latents.ndim == 4  # if latents are 2D, then keep the y coordinate global
            xyz = self.get_local_coordinates(
                global_coords=xyz, local_grid_length=self.global_feat_res, preserve_y=preserve_y
            )

        xyz = xyz.reshape(-1, 3)
        viewdirs = viewdirs.reshape(-1, 3) if viewdirs is not None else None

        rgb, alpha = self.local_generator(z=sampled_local_latents, coords=xyz, viewdirs=viewdirs)

        # shape is [B, H*W, samples_per_ray, 3] if not masks, otherwise [B, n_rays, samples_per_ray, 3]
        rgb = rgb.view(B, -1, samples_per_ray, self.out_dim) if rgb is not None else None
        alpha = alpha.view(B, -1, samples_per_ray)

        return rgb, alpha

    def importance_sampling(self, z_vals, weights, samples_per_ray):
        with torch.no_grad():
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

            B, n_rays, _ = z_vals_mid.shape
            z_vals_mid = z_vals_mid.view(-1, z_vals_mid.shape[-1])
            weights = weights.view(-1, weights.shape[-1]) + 1e-5

            z_samples = sample_pdf_2(
                bins=z_vals_mid,
                weights=weights[..., 1:-1],
                num_samples=samples_per_ray,
                det=False,
            )
            z_samples = z_samples.detach()

            z_vals = z_vals.view(B, n_rays, -1)
            z_samples = z_samples.view(B, n_rays, -1)
        return z_samples

    def forward(self, local_latents, render_params=None, xyz=None):
        assert (xyz is not None) ^ (render_params is not None), 'Use either xyz or render_params, not both.'
        return_alpha_only = True if xyz is not None else False

        if render_params is not None:
            H, W = render_params.nerf_out_res, render_params.nerf_out_res

            # if using feature-NeRF, need to adjust camera intrinsics to account for lower sampling resolution
            if self.img_res is not None:
                downsampling_ratio = render_params.nerf_out_res / self.img_res
            else:
                downsampling_ratio = 1
            fx, fy = render_params.K[0, 0, 0] * downsampling_ratio, render_params.K[0, 1, 1] * downsampling_ratio
            xyz, viewdirs, z_vals, rd, ro = get_sample_points(
                tform_cam2world=render_params.Rt.inverse(),
                F=(fx, fy),
                H=H,
                W=W,
                samples_per_ray=render_params.samples_per_ray,
                near=render_params.near,
                far=render_params.far,
                perturb=self.training,
                mask=render_params.mask,
            )
        else:
            xyz = xyz.unsqueeze(1)  # expand to make shape [B, 1, n_query_points, 3]
            viewdirs = None

        # coarse prediction
        rgb_coarse, alpha_coarse = self.query_network(xyz, local_latents, viewdirs)

        if return_alpha_only:
            return alpha_coarse

        if self.hierarchical_sampling:
            _, _, _, weights, _, occupancy_prior = volume_render_radiance_field(
                rgb=rgb_coarse,
                occupancy=alpha_coarse,
                depth_values=z_vals,
                ray_directions=rd,
                radiance_field_noise_std=render_params.alpha_noise_std,
                alpha_activation=self.alpha_activation,
                activate_rgb=not self.feature_nerf,
                density_bias=self.density_bias,
            )

            z_vals_fine = self.importance_sampling(z_vals, weights, render_params.samples_per_ray)

            xyz = ro[..., None, :] + rd[..., None, :] * z_vals_fine[..., :, None]
            viewdirs = viewdirs[:, :, 0:1].expand_as(xyz)
            rgb_fine, alpha_fine = self.query_network(xyz, local_latents, viewdirs)

            rgb = torch.cat([rgb_coarse, rgb_fine], dim=-2)
            alpha = torch.cat([alpha_coarse, alpha_fine], dim=-1)
            z_vals = torch.cat([z_vals, z_vals_fine], dim=-1)

            _, indices = torch.sort(z_vals, dim=-1)
            z_vals = torch.gather(z_vals, -1, indices)
            rgb_indices = repeat(indices, 'b n_rays n_samples -> b n_rays n_samples d', d=rgb.shape[-1])
            rgb = torch.gather(rgb, -2, rgb_indices)
            alpha = torch.gather(alpha, -1, indices)
        else:
            rgb, alpha = rgb_coarse, alpha_coarse
            z_vals = z_vals

        rgb, _, _, _, depth, occupancy_prior = volume_render_radiance_field(
            rgb=rgb,
            occupancy=alpha,
            depth_values=z_vals,
            ray_directions=rd,
            radiance_field_noise_std=render_params.alpha_noise_std,
            alpha_activation=self.alpha_activation,
            activate_rgb=not self.feature_nerf,
            density_bias=self.density_bias,
        )

        out = {
            'rgb': rgb,
            'depth': depth,
            'Rt': render_params.Rt,
            'K': render_params.K,
            'local_latents': local_latents,
            'occupancy_prior': occupancy_prior,
        }
        return out
