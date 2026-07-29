import os
import glob
import numpy as np
import scipy.interpolate
from scipy.ndimage import convolve

def elastic_distortion(pointcloud, granularity, magnitude):
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3

    for _ in range(2):
        noise = convolve(noise, blurx, mode="constant", cval=0)
        noise = convolve(noise, blury, mode="constant", cval=0)
        noise = convolve(noise, blurz, mode="constant", cval=0)

    ax = [
        np.linspace(d_min, d_min + granularity * (d - 2), d)
        for d_min, d, d_max in zip(
            coords_min - granularity, noise_dim, coords_min + granularity * (noise_dim - 2)
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def apply_augmentation(points, is_train=True):

    if not is_train:
        return points

    if np.random.random() < 0.95:
        for granularity, magnitude in ((0.05, 0.1), (0.2, 0.4)):
            points = elastic_distortion(points, granularity, magnitude)

    jitter = np.random.uniform(-0.02, 0.02, size=points[:, :3].shape)
    points[:, :3] += jitter

    return points
