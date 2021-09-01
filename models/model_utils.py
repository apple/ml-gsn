import torch
from torch import nn
import numpy as np


def flatten_trajectories(data):
    # merge batch and trajectory dimensions in data dictionary
    for key in data.keys():
        if torch.is_tensor(data[key]):
            if data[key].ndim > 2:
                shape = [*data[key].shape]
                data[key] = data[key].reshape([shape[0] * shape[1]] + shape[2:])
    return data


def unflatten_trajectories(data, trajectory_length):
    # unmerge batch and trajectory dimensions in data dictionary
    for key in data.keys():
        if torch.is_tensor(data[key]):
            if data[key].ndim > 1:
                shape = [*data[key].shape]
                data[key] = data[key].reshape([-1, trajectory_length] + shape[1:])
    return data


def collapse_trajectory_dim(x):
    B, T = x.shape[:2]
    other_dims = x.shape[2:]
    return x.view(B * T, *other_dims)


def expand_trajectory_dim(x, T):
    other_dims = x.shape[1:]
    return x.view(-1, T, *other_dims)


def resize_trajectory(x, size):
    # Interpolation for image size, but for tensors with a trajectory dimension
    T = x.shape[1]
    x = collapse_trajectory_dim(x)
    x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
    x = expand_trajectory_dim(x, T)
    return x


def ema_accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


class RenderParams:
    """Render parameters.

    A simple container for the variables required for rendering.
    """

    def __init__(self, Rt, K, samples_per_ray, near, far, alpha_noise_std=0, nerf_out_res=None, mask=None):
        self.samples_per_ray = samples_per_ray
        self.near = near
        self.far = far
        self.alpha_noise_std = alpha_noise_std
        self.Rt = Rt
        self.K = K
        self.mask = mask
        if nerf_out_res:
            self.nerf_out_res = nerf_out_res


class TrajectorySampler(nn.Module):
    """Trajectory sampler.

    Randomly select a trajectory to traverse the scene corresponding to a given latent code. To do so we evalaute
    the occupancy of the scene at each point along each trajectory in a sample set of real trajectories, and then
    choose one that is the most unoccupied. If multiple trajectories are valid, randomly choose from among them.

    Args:
    ----
    real_Rts: torch.Tensor
        Tensor of shape [N, seq_len, 4, 4] containing N real trajectories of camera extrinsic matrices Rt.
    mode: str
        Mode to use when sampling. Options are "bin", "sample", amd "random". "bin" splits trajectories into bins
        based on trajectory length, then takes the trajectory with the most open space. "sample" weighs each
        trajectory according to their occupancy, and then takes a weighted sample. "random" randomly selects
        trajectories without checking for collisions.
    num_bins: int
        Number of bins to uniformly sample trajectories from. Bins are formed based on trajectory distance to
        prevent the sampler from always selecting short trajectories and avoiding long ones.
    alpha_activation: str
        Activation function to use to rectify alpha predictions. Options are 'relu' and 'softplus'.
    jitter_range: int
        Add a random offset to trajectory coordinates during each sample step so that they don't all start at the
        exact same point.

    """

    def __init__(self, real_Rts, mode='sample', num_bins=10, alpha_activation='relu', jitter_range=0):
        super().__init__()

        self.real_Rts = nn.Parameter(real_Rts, requires_grad=False)  # shape [n_trajectories, seq_len, 4, 4]
        self.mode = mode
        self.num_bins = num_bins
        self.alpha_activation = alpha_activation
        self.jitter_range = jitter_range

        # convert Rt matrices to camera pose matrices, then extract translation component
        # make sure Rts are float, since inverse doesn't work with FP16
        self.real_trajectories = real_Rts.float().inverse()[:, :, :3, 3].contiguous()
        # shape [n_trajectories, seq_len, 3]
        self.real_trajectories = nn.Parameter(self.real_trajectories, requires_grad=False)
        self.seq_len = self.real_trajectories.shape[1]

        if mode == 'bin':
            self.bin_indices = self.get_bin_indices(num_bins)

    def get_bin_indices(self, num_bins):
        # for each trajectory, calculate distance between first and last point
        displacements = torch.sum((self.real_trajectories[:, 0] - self.real_trajectories[:, -1]) ** 2, dim=1) ** 0.5
        # sort by displacement
        sort_indices = torch.argsort(displacements)
        # split into bins of equal size
        bin_indices = torch.chunk(sort_indices, chunks=num_bins)
        return bin_indices

    def shuffle_trajectories_in_bins(self):
        new_bin_indices = []
        for trajectory_bin in self.bin_indices:
            permutation_indices = torch.randperm(trajectory_bin.shape[0])
            new_bin_indices.append(trajectory_bin[permutation_indices])
        self.bin_indices = new_bin_indices

    def get_occupancy(self, generator, local_latents, trajectories):
        B = local_latents.shape[0]
        query_points = trajectories.unsqueeze(0).expand(B, -1, -1, -1)
        query_points = query_points.to(local_latents.device)

        B, n_trajectories, seq_len, _ = query_points.shape

        query_points = query_points.view(B, -1, 3)

        if local_latents.dtype == torch.float16:
            query_points = query_points.half()

        # get occupancies for all trajectories
        with torch.no_grad():
            # z is tensor for shape [B, z_dim]
            occupancy = generator(local_latents=local_latents, xyz=query_points)

        # bin mode doesn't work great with softplus, so use ReLU anyway in that case
        if (self.alpha_activation == 'relu') or (self.mode == 'bin'):
            # anything negative is unoccupied
            occupancy = torch.nn.functional.relu(occupancy)
        elif self.alpha_activation == 'softplus':
            occupancy = torch.nn.functional.softplus(occupancy)

        occupancy = occupancy.view(B, -1, self.seq_len)  # [B, n_trajectories, seq_len]
        occupancy = torch.sum(occupancy, dim=2)  # [B, n_trajectories]
        return occupancy

    def sample_trajectories(self, generator, local_latents):
        """Return trajectories that best traverse a given scene.

        Input:
        -----
        generator: SceneGenerator
            Generator object to be evaluated for occupancy.
        local_latents: torch.Tensor
            Local latent codes of shape [B, local_z_dim, H, W] corresponding to the scenes that will be evaluted.

        Return:
        ------
        Rts: torch.Tensor
            Trajectories of camera extrinsic matrices of shape [B, seq_len, 4, 4].

        """
        B = local_latents.shape[0]

        real_Rts = self.real_Rts.clone()
        if self.jitter_range:
            n_trajectories, seq_len, _, _ = real_Rts.shape

            jitter = torch.rand(size=(n_trajectories, 1, 3), device=real_Rts.device, requires_grad=False)
            jitter = (jitter * 2) - 1  # normalize to [-1, 1]
            jitter = jitter * self.jitter_range
            jitter[:, :, 1] = jitter[:, :, 1] * 0  # no jitter on the y axis

            camera_pose = real_Rts.inverse()
            camera_pose[:, :, :3, 3] = camera_pose[:, :, :3, 3] + jitter
            trajectories = camera_pose[:, :, :3, 3]
            real_Rts = camera_pose.inverse()
        else:
            trajectories = self.real_trajectories

        if self.mode == 'sample':
            occupancy = self.get_occupancy(generator=generator, local_latents=local_latents, trajectories=trajectories)

            # randomly choose 1k trajectories to sample from
            n_subsamples = min(real_Rts.shape[0], 1000)
            subset_indices = torch.multinomial(
                torch.ones(real_Rts.shape[0]), num_samples=n_subsamples, replacement=False
            )

            sample_weights = nn.functional.softmin(occupancy[:, subset_indices] + 1e-8, dim=-1)
            nans = torch.isnan(sample_weights)
            sample_weights[nans] = 1 / 1000
            selected_indices = torch.multinomial(sample_weights, num_samples=1, replacement=False).squeeze(1)

            Rts = real_Rts[subset_indices][selected_indices]

        elif self.mode == 'bin':
            occupancy = self.get_occupancy(generator=generator, local_latents=local_latents, trajectories=trajectories)

            # shuffle trajectories so that we don't always select the first completely unoccupied trajectory
            self.shuffle_trajectories_in_bins()

            Rts = []
            for i in range(len(local_latents)):
                selected_bin = self.bin_indices[np.random.choice(a=self.num_bins)]
                occupancies = occupancy[i, selected_bin]
                most_empty_idx = torch.argmin(occupancies)
                most_empty_idx = selected_bin[most_empty_idx]
                Rts.append(real_Rts[most_empty_idx])
            Rts = torch.stack(Rts, dim=0)

        elif self.mode == 'random':
            weight = torch.ones(real_Rts.shape[0])
            selected_indices = torch.multinomial(weight, num_samples=B, replacement=False)
            Rts = real_Rts[selected_indices]

        Rts = Rts.to(local_latents.device)

        return Rts
