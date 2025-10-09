import scipy
import numpy as np
import mrcfile
from scipy.spatial.transform import Rotation as R

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

def load_vesicle_ground_truth(path):
    with mrcfile.open(path, permissive=True) as mrc:
        true_vol = mrc.data.astype(np.float32).transpose(2, 1, 0)  # (Z, Y, X)
    return true_vol
