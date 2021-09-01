# Adapted from https://github.com/krrish94/nerf-pytorch

import torch
from einops import repeat


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


def get_ray_bundle_batch(height: int, width: int, focal_length, tform_cam2world: torch.Tensor):
    r"""Compute the bundle of rays passing through all pixels of a batch of image (one ray per pixel).

    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(B, 4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(B, width, height, 3)` denoting the centers of
      each ray. `ray_origins[B][i][j]` denotes the origin of the ray passing through pixel at batch index
      `B`, row index `j`, and column index `i`.
    ray_directions (torch.Tensor): A tensor of shape :math:`(B, width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[B][i][j]` denotes the direction of the ray
      passing through the pixel at batch index `B`, row index `j`, and column index `i`.
    """

    x = torch.arange(width, dtype=tform_cam2world.dtype, device=tform_cam2world.device).to(tform_cam2world)
    y = torch.arange(height, dtype=tform_cam2world.dtype, device=tform_cam2world.device)
    ii, jj = meshgrid_xy(x, y)

    if type(focal_length) in [tuple, list]:
        # if given two values, assume they are fx and fy
        fx, fy = focal_length
    else:
        # otherwise assume fx and fy share the same magnitude, but opposing polarity
        fx, fy = focal_length, -focal_length

    # construct unit direction vectors
    # shape [height, width, 3]
    directions = torch.stack([(ii - width * 0.5) / fx, (jj - height * 0.5) / fy, -torch.ones_like(ii)], dim=-1)

    B = tform_cam2world.shape[0]

    # shape [B x height x width, 1, 3]
    directions = directions.view(1, -1, 1, 3).repeat(B, 1, 1, 1).view(-1, 1, 3)
    # shape [B x height x width, 4, 4]
    tform_cam2world = tform_cam2world.unsqueeze(1).repeat(1, height * width, 1, 1).view(-1, 4, 4)

    ray_directions = torch.sum(directions * tform_cam2world[:, :3, :3], dim=-1).view(B, height, width, 3)
    ray_origins = tform_cam2world[:, :3, -1].view(B, height, width, 3)
    return ray_origins, ray_directions


def get_sample_points(
    tform_cam2world, F, H, W, samples_per_ray=32, near=0, far=1, use_viewdirs=True, perturb=False, mask=None
):
    B = tform_cam2world.shape[0]
    ray_origins, ray_directions = get_ray_bundle_batch(H, W, F, tform_cam2world)  # [B, H, W, 3]

    ro = ray_origins.view((B, -1, 3))
    rd = ray_directions.view((B, -1, 3))

    if mask is not None:
        if len(mask.shape) == 1:
            # same mask for each image in batch, mask is shape [n_patch_pixels]
            ro = ro[:, mask, :]
            rd = rd[:, mask, :]
        elif len(mask.shape) == 2:
            # different mask for each image in batch, mask is shape [B, n_patch_pixels]
            mask = repeat(mask, 'b n_patch_pixels -> b n_patch_pixels 3')
            # ro is shape [B, n_pixels, 3], gather along pixel dimension
            ro = torch.gather(ro, dim=1, index=mask)
            rd = torch.gather(rd, dim=1, index=mask)

    near = near * torch.ones_like(rd[..., :1])
    far = far * torch.ones_like(rd[..., :1])

    num_rays = ro.shape[1]

    t_vals = torch.linspace(0.0, 1.0, samples_per_ray, dtype=ro.dtype, device=ro.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals

    if perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand

    # pts -> (B, H*W, N_samples, 3)
    # pts are in world coordinates
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    if use_viewdirs:
        viewdirs = rd
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((B, -1, 1, 3))

        # input_dirs -> (B, H*W, N_samples, 3)
        viewdirs = viewdirs.expand(pts.shape)
    else:
        viewdirs = None

    return pts, viewdirs, z_vals, rd, ro


def volume_render_radiance_field(
    rgb,
    occupancy,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    alpha_activation='relu',
    activate_rgb=True,
    density_bias=0,
):

    one_e_10 = torch.tensor([1e10], dtype=ray_directions.dtype, device=ray_directions.device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                occupancy.shape,
                dtype=occupancy.dtype,
                device=occupancy.device,
            )
            * radiance_field_noise_std
        )

    if alpha_activation == 'relu':
        sigma_a = torch.nn.functional.relu(occupancy + noise)
    elif alpha_activation == 'softplus':
        # Deformable NeRF uses softplus instead of ReLU https://arxiv.org/pdf/2011.12948.pdf
        sigma_a = torch.nn.functional.softplus(occupancy + noise + density_bias)

    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    if activate_rgb:
        rgb = torch.sigmoid(rgb)

        # widened sigmoid from https://github.com/google/mipnerf/blob/main/internal/models.py#L123
        rgb_padding = 0.001
        rgb = rgb * (1 + 2 * rgb_padding) - rgb_padding

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)

    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)

    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    # occupancy prior from Neural Volumes
    # https://github.com/facebookresearch/neuralvolumes/blob/master/models/neurvol1.py#L130
    occupancy_prior = torch.mean(
        torch.log(0.1 + alpha.view(alpha.size(0), -1)) + torch.log(0.1 + 1.0 - alpha.view(alpha.size(0), -1)) - -2.20727
    )

    return rgb_map, disp_map, acc_map, weights, depth_map, occupancy_prior


def sample_pdf_2(bins, weights, num_samples, det=False):
    """sample_pdf function from another concurrent pytorch implementation
    by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
    """

    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=num_samples, dtype=weights.dtype, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [num_samples],
            dtype=weights.dtype,
            device=weights.device,
        )

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
