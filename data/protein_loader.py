import numpy as np
from scipy.spatial.transform import Rotation as R

def load_protein_data(paths, return_angles=False):
    images = np.load(paths.projection_file).transpose(2, 1, 0).copy()
    angles = np.loadtxt(paths.rotations_file, dtype=np.float32)
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix()
    if return_angles:
        return images, rotations, angles
    else:
        return images, rotations