import torch
from .nerf_utils import get_ray_bundle_batch


def backproject(voxel_dim, voxel_size, world_center, Rt, K, features, depth=None, resize=True, return_pointcloud=False):
    """
    Take 2d features and fills them along rays in a 3d volume.
    Inspired by https://github.com/magicleap/Atlas/blob/master/atlas/model.py#L35

    Args:
        voxel_dim (tuple, int): tuple indicating the number of voxels along each dimension
            of the voxel grid (nx,ny,nz)
        voxel_size (float): the dimensions of a voxel in real world units (i.e. whatever units
            were used to measure the camera intrinsic and extrinsic matrices). E.g. if voxel_size
            is (100, 100, 100) and represents a 4mx4mx4m cube in reality, then voxel_size would
            be equal to 0.04 (4m per 100 voxels = 4/100 = 0.04m per voxel).
        world_center (torch.Tensor): xyz coordinate indicating the center of the world coordinate frame
        Rt (torch.Tensor): Bx4x4 extrinsic camera matrices
        K (float): Bx4x4 intrinsic camera matrices
        features (torch.Tensor): BxCxHxW features to be backprojected into 3D
        depth (torch.Tensor): Bx1xHxW depth values to use for backprojection (optional)
        resize (bool): indicates whether to upsample the input features to ensure there are no holes in
            the ray projection

    Returns:
        volume (torch.Tensor): volume containing backprojected features. Of shape [nx, ny, nz, C].

    """
    tform_cam2world = Rt.inverse()
    fx, fy = K[0, 0, 0], K[0, 1, 1]  # grab the first value and assume all others are similar

    if resize:
        # upscale the feature map so that we don't get empty holes in our ray projections
        B, C, H, W = features.shape
        features = torch.nn.functional.interpolate(features, size=max(voxel_dim), mode="bilinear", align_corners=False)
        downsample_ratio = max(voxel_dim) / max(H, W)  # new / old
        fx, fy = fx * downsample_ratio, fy * downsample_ratio
        samples_per_ray = max(voxel_dim)

    nx, ny, nz = voxel_dim
    B, C, H, W = features.shape
    device = features.device

    if depth is not None:
        samples_per_ray = 1
        if resize:
            depth = torch.nn.functional.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

    voxels_per_unit_dimension = 1 / voxel_size

    # get ray origins and ray directions based on focal length and extrinsic matrix
    ro, rd = get_ray_bundle_batch(H, W, (fx, fy), tform_cam2world)
    ro = ro.view((-1, 3))
    rd = rd.view((-1, 3))
    num_rays = ro.shape[0]

    if depth is None:
        # project points along each ray at uniform intervals
        t_vals = torch.linspace(0.0, 1.0, samples_per_ray, dtype=ro.dtype, device=ro.device)
        z_vals = 0 * (1.0 - t_vals) + (voxel_size * samples_per_ray) * t_vals
        z_vals = z_vals.expand([num_rays, samples_per_ray])
    else:
        z_vals = depth.view(num_rays, 1)

    # pts -> (num_rays, N_samples, 3)
    # pts are in world coordinates
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    # now we want to convert from world coordinates to voxel coordinates
    pts = pts * voxels_per_unit_dimension  # scale to match voxel grid
    world_center = torch.tensor(world_center, dtype=torch.float, device=device)
    world_center = world_center * voxels_per_unit_dimension
    # one corner of voxel grid will always be at (0, 0, 0), and the oppostie corner at (voxel_dim)
    #     so we can easily find center by dividing by 2
    voxel_center = torch.tensor(voxel_dim, dtype=torch.float, device=device) / 2
    offset = voxel_center - world_center
    pts_aligned = pts + offset  # pts should now be aligned with voxel grid
    pts_grid = pts_aligned.round().long()  # snap to grid

    pts_flat = pts_grid.view(-1, 3)
    px = pts_flat[:, 0]
    py = pts_flat[:, 1]
    pz = pts_flat[:, 2]

    # find out which points along the backprojected rays lie within the volume
    valid = (px >= 0) & (px < voxel_dim[0]) & (py >= 0) & (py < voxel_dim[1]) & (pz >= 0) & (pz < voxel_dim[2])

    volume = torch.zeros(B, nx, ny, nz, C, dtype=features.dtype, device=device)

    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    batch_idx = torch.repeat_interleave(batch_idx, repeats=H * W * samples_per_ray, dim=0)

    # put channel dimension at the back
    features = features.permute(0, 2, 3, 1).contiguous()
    # replicate features for each sample along the rays
    features = features.view(-1, C).unsqueeze(1).expand(-1, samples_per_ray, -1).reshape(-1, C)

    if return_pointcloud:
        batch_coords = torch.cat([batch_idx.unsqueeze(1), pts_aligned.view(-1, 3)], dim=-1)
        return batch_coords[valid], features[valid]
    else:
        volume[batch_idx[valid], pts_flat[valid][:, 0], pts_flat[valid][:, 1], pts_flat[valid][:, 2], :] = features[
            valid
        ]
        volume = volume.permute(0, 4, 1, 2, 3).contiguous()
        return volume
