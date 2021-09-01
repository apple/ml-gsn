import numpy as np
import torch
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as Rot


def rotate_360(n=10):
    Rs = []
    for theta in np.linspace(0, 2 * np.pi, n):
        R = torch.from_numpy(
            np.asarray([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        )
        Rs.append(R)
    Rts = torch.eye(4).unsqueeze(0).repeat(n, 1, 1)
    Rts[:, :3, :3] = torch.stack(Rs)
    return Rts


def rotate_n(n=30):
    theta = np.deg2rad(n)
    R = torch.from_numpy(np.asarray([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]))

    Rt = torch.eye(4).repeat(1, 1)
    Rt[:3, :3] = R
    return Rt


def go_forward(Rt_curr, step=0.25):
    pose = Rt_curr[0, 0]

    # Forward is -z
    pose[2, -1] = pose[2, -1] + step

    return pose.unsqueeze(0).unsqueeze(0)


def go_backward(Rt_curr, step=0.25):
    pose = Rt_curr[0, 0]
    pose[2, -1] = pose[2, -1] - step

    return pose.unsqueeze(0).unsqueeze(0)


def camera_path_interp(Rt_0, Rt_1, n_samples=100):
    t_0 = Rt_0[:3, -1]
    t_1 = Rt_1[:3, -1]

    t_path = []
    alpha = np.linspace(0, 1, n_samples)
    for a in alpha:
        t = (t_0 * (1 - a)) + (t_1 * a)
        t_path.append(t)

    R_0 = Rt_0[:3, :3]
    R_1 = Rt_1[:3, :3]

    Rs = torch.stack([R_0, R_1])
    Rs = Rs.cpu().numpy().tolist()

    slerp = Slerp([0, 1], Rot.from_matrix(Rs))
    times = np.linspace(0, 1, n_samples)
    R_path = slerp(times).as_matrix()

    camera_path = torch.eye(4).unsqueeze(0).repeat(n_samples, 1, 1)
    camera_path[:, :3, :3] = torch.from_numpy(R_path)
    camera_path[:, :3, -1] = torch.stack(t_path)

    return camera_path
