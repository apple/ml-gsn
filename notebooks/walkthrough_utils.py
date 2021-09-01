import torch
import numpy as np
from scipy.interpolate import interp1d

from utils.camera_trajectory import rotate_n


def fit_spline(x, y, n=75, interp_mode='cubic'):
    '''
    x: X coordinates
    y: y coordinates
    n: number of sample points along the trajectory to sample
    interp_mode: interpolation mode, must be one of 'slinear', 'cubic', or 'quadratic'

    from https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
    '''

    points = np.stack([x, y], axis=1)

    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    alpha = np.linspace(0, 1, n)

    interpolator = interp1d(distance, points, kind=interp_mode, axis=0)
    interpolated_points = interpolator(alpha)

    return interpolated_points


def coord2angle(coords):
    # given a series of coordinates, find the direction of each step
    # expects coords to be an array of shape (N, 2)
    diff = coords[1:] - coords[:-1]

    angles = []
    for coord in diff:
        x, y = coord
        angle = np.rad2deg(np.arctan2(y, x))
        angles.append(angle)

    angles.append(angles[-1])  # repeat last angle to make up for the missing value from the diff
    return angles


def get_smooth_trajectory(Rt, n_frames=300, subsample=2):
    # expects input to be a list of Rts, shape [B, 4, 4]
    # subsampling uses less splines, resulting in a smoother trajectory

    assert len(Rt) / subsample > 3, 'Sequence should have at least 4 keypoints after subsampling'

    path = Rt.inverse()[:, :3, 3][:, [0, 2]].clone().cpu().numpy()

    sub_path = path[::subsample]  # subsample for smoother trajectory

    spline_coords = fit_spline(x=sub_path[:, 0], y=sub_path[:, 1], n=n_frames, interp_mode='cubic')
    spline_angles = coord2angle(spline_coords)

    new_Rts = []
    for angle, coords in zip(spline_angles, spline_coords):
        rotate_only = rotate_n(angle + 90)  # re-adjust angle so camera looks forward

        Rt = rotate_only.inverse()
        Rt[0, 3] = coords[0]
        Rt[2, 3] = coords[1]
        Rt = Rt.inverse()
        new_Rts.append(Rt)

    new_Rts = torch.stack(new_Rts, dim=0).unsqueeze(1)
    return new_Rts
