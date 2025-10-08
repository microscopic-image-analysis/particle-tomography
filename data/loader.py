import h5py
import numpy as np
import scipy
import mrcfile
import pandas as pd
from typing import Optional
from scipy.spatial.transform import Rotation as R
import os


def load_vesicle_data(paths, return_angles=False):
    images = scipy.io.loadmat(paths.projection_file)["projections_FST"].astype(np.float32)
    images = np.transpose(images, (2, 1, 0)).copy() # transpose to N, H, W
    print(f"File contains {images.shape[0]} projection images of size {images.shape[1]}x{images.shape[2]} pixels")

    angles = scipy.io.loadmat(paths.rotations_file)["angles"]
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix().astype(np.float32)  # (N, 3, 3)

    shifts = compute_halfpixel_shifts(rotations, grid_size=images.shape[-1]).astype(np.float32)  # (N, 2)
    if return_angles:
        return (images,  # (N, H, W)
                rotations,  # (N, 3, 3)
                shifts,  # (N, 2)
                angles
                )
    else:
        return (images,  # (N, H, W)
                rotations,  # (N, 3, 3)
                shifts,  # (N, 2)
                )


def load_vesicle_data_ang_ref(paths, return_angles=False):
    images = scipy.io.loadmat(paths.projection_file)["projections_FST"].astype(np.float32)
    images = np.transpose(images, (2, 1, 0)).copy()  # transpose to N, H, W
    print(f"File contains {images.shape[0]} projection images of size {images.shape[1]}x{images.shape[2]} pixels")

    with mrcfile.open(paths.true_volume_file, permissive=True) as mrc:
        true_vol = mrc.data.astype(np.float32)  # (Z, Y, X)

    with h5py.File(paths.rotations_file, 'r') as f:
        noisy_angles = np.array(f['angles'], dtype=np.float32).transpose(1,0)
        noisy_rotations = R.from_euler("zyx", noisy_angles, degrees=True).inv().as_matrix() # (N, 3, 3)

    shifts = compute_halfpixel_shifts(noisy_rotations, grid_size=images.shape[-1]).astype(np.float32)  # (N, 2)

    true_angles = scipy.io.loadmat(paths.true_rotations_file)["angles"]
    print(true_angles.shape)
    true_rotations = R.from_euler("zyx", true_angles, degrees=True).inv().as_matrix().astype(np.float32)

    return (images,  # (N, H, W)
        noisy_rotations,  # (N, 3, 3)
        shifts,  # (N, 2)
        true_vol,  # (Z, Y, X)
        true_rotations
    )


def load_protein_data(paths, return_angles=False):
    images = np.load(paths.projection_file).transpose(2, 1, 0).copy()
    angles = np.loadtxt(paths.rotations_file, dtype=np.float32)
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix()
    if return_angles:
        return images, rotations, angles
    else:
        return images, rotations


def load_thinfilm_data(paths, return_angles=False, normalize=True):
    mat = scipy.io.loadmat(paths.projection_file)
    images = mat['proj_crop'].astype(np.float32).transpose(2, 1, 0).copy()  # (N, H, W)
    if normalize:
        images = images / images.max()  # scales everything so max pixel = 1
    mat = scipy.io.loadmat(paths.rotations_file)
    angles = np.array(mat['final_ang'], dtype=np.float32)
    print(images.shape)
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix().astype(np.float32)
    if return_angles:
        return images, rotations, angles
    else:
        return images, rotations

def load_Pd2_data(paths):
    images = np.array(scipy.io.loadmat(paths.projection_file)["proj"]).transpose(2, 1, 0).copy()
    angles = np.squeeze(scipy.io.loadmat(paths.rotations_file)["ang"])
    vol = np.array(scipy.io.loadmat(paths.true_volume_file)["final_Rec"])
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix()
    return images, rotations, vol

def compute_halfpixel_shifts(rotations: np.ndarray, grid_size: int = 64) -> np.ndarray:
    """
    Compute half-pixel correction shifts using numpy.

    Args:
        rotations: [N, 3, 3] rotation matrices (numpy array)
        grid_size: volume grid size (assumes cubic volume)

    Returns:
        shifts: [N, 2] array of 2D shifts per projection
    """
    pixel_size = 2.0 / grid_size
    t = np.array([-0.5 * pixel_size, -0.5 * pixel_size, -0.5 * pixel_size])  # shape [3]
    t_expanded = np.tile(t, (rotations.shape[0], 1))[:, :, None]  # [N, 3, 1]

    rotated_t = np.matmul(rotations, t_expanded).squeeze(-1)  # [N, 3]
    delta = rotated_t - t  # [N, 3]
    return delta[:, :2]  # [N, 2]
