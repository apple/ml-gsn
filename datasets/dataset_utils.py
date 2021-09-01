import os
import torch
import numpy as np


def listdir_nohidden(path):
    mylist = [f for f in os.listdir(path) if not f.startswith('.')]
    return mylist


def normalize_trajectory(Rt, center='first', normalize_rotation=True):
    assert center in ['first', 'mid'], 'center must be either "first" or "mid", got {}'.format(center)

    seq_len = Rt.shape[1]

    if center == 'first':
        origin_frame = 0
    elif center == 'mid':
        origin_frame = seq_len // 2
    else:
        # return unmodified Rt
        return Rt

    if normalize_rotation:
        origins = Rt[:, origin_frame : origin_frame + 1].expand_as(Rt).reshape(-1, 4, 4).inverse()
        normalized_Rt = torch.bmm(Rt.view(-1, 4, 4), origins)
        normalized_Rt = normalized_Rt.view(-1, seq_len, 4, 4)
    else:
        camera_pose = Rt.inverse()
        origins = camera_pose[:, origin_frame : origin_frame + 1, :3, 3]
        camera_pose[:, :, :3, 3] = camera_pose[:, :, :3, 3] - origins
        normalized_Rt = camera_pose.inverse()

    return normalized_Rt


def random_rotation_augment(trajectory_Rt):
    # given a trajectory, apply a random rotation
    angle = np.random.randint(-180, 180)
    angle = np.deg2rad(angle)
    _rand_rot = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    rand_rot = torch.eye(4)
    rand_rot[:3, :3] = torch.from_numpy(_rand_rot).float()

    for i in range(len(trajectory_Rt)):
        trajectory_Rt[i] = trajectory_Rt[i].mm(rand_rot)

    return trajectory_Rt
