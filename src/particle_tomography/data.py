import numpy as np
from scipy.spatial.transform import Rotation as R


def load_protein_data(paths, return_angles=False):
    """Load protein projection images and rotations."""
    images = np.load(paths.projection_file).transpose(2, 1, 0).copy().astype(np.float32)
    angles = np.loadtxt(paths.rotations_file, dtype=np.float32)
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix().astype(np.float32)
    if return_angles:
        return images, rotations, angles
    return images, rotations


def load_protein_ground_truth(path):
    """Load protein ground-truth volume from a .npz file."""
    true_volume = np.load(path).transpose(2, 1, 0).copy()
    return true_volume.astype(np.float32)
