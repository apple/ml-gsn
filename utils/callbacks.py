import os
import copy
import torch
import imageio
from typing import Sequence
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import LightningModule, Trainer

from models.backprojection_utils import backproject
from models.model_utils import collapse_trajectory_dim, expand_trajectory_dim, resize_trajectory
from utils.utils import exclusive_mean


class GSNVizCallback(Callback):
    def __init__(self, log_dir, voxel_res, voxel_size) -> None:
        super().__init__()
        self.viz_dir = os.path.join(log_dir, 'viz')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
        self.voxel_res = voxel_res
        self.voxel_size = voxel_size

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if batch_idx == 0:
            T = 8

            filename = 'real_samples_epoch_{:06d}.png'.format(trainer.current_epoch)
            sample_static(self.viz_dir, filename, collapse_trajectory_dim(batch['rgb'].clone()), nrow=T)

            filename = 'real_floorplan_epoch_{:06d}.png'.format(trainer.current_epoch)
            sample_floorplans(self.viz_dir, filename, self.voxel_res, self.voxel_size, batch)

            filename = 'real_trajectories_epoch_{:06d}.png'.format(trainer.current_epoch)
            sample_trajectories(self.viz_dir, filename, self.voxel_res, self.voxel_size, batch['Rt'].clone())

            del batch['Rt']  # remove Rt so that trajectory sampler fills it in
            for k in batch.keys():
                batch[k] = batch[k].cuda()
            with torch.no_grad():
                z = torch.rand(batch['K'].shape[0], pl_module.z_dim, device=batch['K'].device)
                y_hat_rgb, y_hat_depth, Rt, K = pl_module(z, batch)

            filename = 'fake_samples_epoch_{:06d}.png'.format(trainer.current_epoch)
            sample_static(self.viz_dir, filename, collapse_trajectory_dim(y_hat_rgb), nrow=T)

            fake_batch = {'rgb': y_hat_rgb, 'depth': y_hat_depth, 'Rt': Rt, 'K': K}
            filename = 'fake_floorplan_epoch_{:06d}.png'.format(trainer.current_epoch)
            sample_floorplans(self.viz_dir, filename, self.voxel_res, self.voxel_size, fake_batch)

            filename = 'fake_trajectories_epoch_{:06d}.png'.format(trainer.current_epoch)
            sample_trajectories(self.viz_dir, filename, self.voxel_res, self.voxel_size, Rt)


def sample_static(viz_dir, filename, rgb, nrow=4):
    # save trajectories like a film strip
    filepath = os.path.join(viz_dir, filename)
    save_image(rgb, fp=filepath, nrow=nrow, range=(0, 1))


def sample_video(sample_dir, filename, rgb, n_trajectories=64, frame_duration=0.2):
    video_filepath = os.path.join(sample_dir, 'video')
    if not os.path.exists(video_filepath):
        os.mkdir(video_filepath)
    filepath = os.path.join(video_filepath, filename)

    frames = []
    for i in range(rgb.shape[1]):
        grid = make_grid(rgb[:n_trajectories, i], nrow=int(np.sqrt(n_trajectories)))
        grid = grid.permute(1, 2, 0) * 255
        grid = grid.cpu().numpy().astype(np.uint8)

        frames.append(grid)
    imageio.mimsave(filepath, frames, duration=frame_duration)


def sample_video_single(sample_dir, filename, rgb_in, rgb_pred, frame_duration=0.2):
    video_filepath = os.path.join(sample_dir, 'video')
    if not os.path.exists(video_filepath):
        os.mkdir(video_filepath)
    filepath = os.path.join(video_filepath, filename)

    frames = []
    for frame_in, frame_pred in zip(rgb_in, rgb_pred):
        grid = make_grid([frame_in, frame_pred], nrow=2)
        grid = grid.permute(1, 2, 0) * 255
        grid = grid.cpu().numpy().astype(np.uint8)
        frames.append(grid)

    imageio.mimsave(filepath, frames, duration=frame_duration)


def get_floorplans(data, voxel_res, voxel_size, floorplan_res, batch_size=8):
    B, T, C, H, W = data['rgb'].shape

    volume_length = voxel_res * voxel_size  # recalculate original length of volume
    voxel_size = volume_length / floorplan_res  # scale to new resolution

    if data['depth'].shape[3] != H:
        data['depth'] = resize_trajectory(x=data['depth'], size=H)

    floorplans = []
    for i in range(0, B, batch_size):
        with torch.no_grad():
            volume = backproject(
                voxel_dim=(floorplan_res, floorplan_res, floorplan_res),
                voxel_size=voxel_size,  # should be roughly equivalent to (self.opt.far * 2 / vd)
                world_center=(0, 0, 0),
                Rt=collapse_trajectory_dim(data['Rt'][i : i + batch_size]),
                K=collapse_trajectory_dim(data['K'][i : i + batch_size]),
                features=collapse_trajectory_dim(data['rgb'][i : i + batch_size]),
                depth=collapse_trajectory_dim(data['depth'][i : i + batch_size]),
            )

            volume = expand_trajectory_dim(volume, T=T)
            volume = exclusive_mean(volume, dim=1)  # merge along trajectory dim

            # remove the top half of the scene (i.e the ceiling)
            height = volume.shape[3]
            volume = volume[:, :, :, : height // 2, :]

            # take the first nonzero pixel, as if looking down from a bird's-eye view
            depth_idx = torch.argmax((volume > 0).float(), dim=3, keepdim=True)
            floorplan = torch.gather(volume, 3, depth_idx).squeeze(3)
            floorplans.append(floorplan.cpu())

    floorplans = torch.cat(floorplans, dim=0)
    floorplans = floorplans.permute(0, 1, 3, 2)  # swap height and width dimensions
    return floorplans


def sample_floorplans(viz_dir, filename, voxel_res, voxel_size, data, floorplan_res=200, n_trajectories=8):
    filepath = os.path.join(viz_dir, filename)

    data = copy.deepcopy(data)  # copy so we don't mess up the original dictionary
    data['rgb'] = data['rgb'][:n_trajectories]
    data['depth'] = data['depth'][:n_trajectories]
    data['Rt'] = data['Rt'][:n_trajectories]
    data['K'] = data['K'][:n_trajectories]

    floorplans = get_floorplans(data, voxel_res=voxel_res, voxel_size=voxel_size, floorplan_res=floorplan_res)

    save_image(floorplans, fp=filepath, nrow=2, range=(0, 1), pad_value=1)


def sample_trajectories(viz_dir, filename, voxel_res, voxel_size, Rts):
    filepath = os.path.join(viz_dir, filename)

    xyz = Rts.inverse()[:, :, 0:3, 3].cpu().numpy()
    for i in range(len(xyz)):
        plt.plot(xyz[i, :, 0], xyz[i, :, 2], c='blue', alpha=0.1, linewidth=3)
        plt.scatter(xyz[i, :, 0], xyz[i, :, 2], c='blue', alpha=0.1, linewidth=3)
    plt.xlabel('X')
    plt.ylabel('Z')
    extents = voxel_res * voxel_size / 2
    plt.xlim(-extents, extents)
    plt.ylim(-extents, extents)
    plt.savefig(filepath)
    plt.close()
