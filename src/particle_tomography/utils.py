import torch
import torch.nn.functional as F
import numpy as np

def compute_gaussian_kernel_1d(kernel_size, sigma, device, dtype):
    """
    Computes 1D Gaussian kernel dynamically (differentiable).

    Args:
        kernel_size (int): Size of the kernel
        sigma (torch.Tensor): Standard deviation of the Gaussian (must be a tensor for differentiability)
        device: Device to place the tensor on
        dtype: Data type of the tensor

    Returns:
        torch.Tensor: 1D Gaussian kernel, normalized to sum to 1
    """
    ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def convolve_separable_3d(voxel_grid, kernel_size, sigma):
    """
    Applies separable 3D convolution to blur a voxel grid using dynamically computed 1D Gaussian kernel.

    Args:
        voxel_grid (torch.Tensor): Voxel grid of shape (D, H, W)
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian

    Returns:
        torch.Tensor: Convolved voxel grid of shape (D, H, W)
    """
    device = voxel_grid.device
    dtype = voxel_grid.dtype

    # Compute kernel dynamically (this is differentiable)
    kernel_1d = compute_gaussian_kernel_1d(kernel_size, sigma, device, dtype)

    r = kernel_size // 2

    # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
    voxel_grid = voxel_grid.unsqueeze(0).unsqueeze(0)

    # Prepare 1D kernels as 3D conv filters
    kernel_z = kernel_1d.view(1, 1, -1, 1, 1)  # (1, 1, k, 1, 1) - convolve along depth
    kernel_y = kernel_1d.view(1, 1, 1, -1, 1)  # (1, 1, 1, k, 1) - convolve along height
    kernel_x = kernel_1d.view(1, 1, 1, 1, -1)  # (1, 1, 1, 1, k) - convolve along width

    # Apply separable convolution in three passes
    # First: convolve along Z (depth) dimension
    out = F.conv3d(voxel_grid, kernel_z, padding=(r, 0, 0))

    # Second: convolve along Y (height) dimension
    out = F.conv3d(out, kernel_y, padding=(0, r, 0))

    # Third: convolve along X (width) dimension
    out = F.conv3d(out, kernel_x, padding=(0, 0, r))

    # Remove batch and channel dimensions: (1, 1, D, H, W) -> (D, H, W)
    return out.squeeze(0).squeeze(0)


def rasterize_and_smooth_3d(points, point_weights=None, grid_size=(64, 64, 64), sigma=0.001, kernel_size=3, dtype=torch.float32, device="cpu"):
    """
    Converts 3D point cloud to voxel grid with trilinear splatting and Gaussian smoothing.

    Args:
        points: [N, 3] coordinates in [-1, 1] range
        point_weights: [N] point weights
        grid_size: (H, D, W)
        sigma: Gaussian smoothing std in voxel units

    Returns:
        [Z, Y, X] voxel grid
    """
    # bring inputs to device
    points = torch.from_numpy(points) if isinstance(points, np.ndarray) else points
    points = points.to(dtype=dtype, device=device)

    if point_weights is None:
        point_weights = torch.ones(points.shape[0], dtype=dtype, device=device)
    else:
        point_weights = torch.from_numpy(point_weights) if isinstance(point_weights, np.ndarray) else point_weights
        point_weights = point_weights.to(dtype=dtype, device=device)

    N, _ = points.shape
    grid_size_z, grid_size_y, grid_size_x = grid_size
    max = np.max(grid_size)
    shift_z_up = (max - grid_size_z + 1) // 2
    shift_y_back = (max - grid_size_y + 1) // 2
    shift_x_left = (max - grid_size_x + 1) // 2


    points = (points + 1) * 0.5
    rescaled_points = points * max - torch.tensor([0.5, 0.5, 0.5], device=device, dtype=dtype)  # (N, 3)
    base = rescaled_points.floor().int()  # (N, 3)
    delta = rescaled_points - base  # (N, 3)
    dx, dy, dz = delta[..., 0], delta[..., 1], delta[..., 2]  # (N, 3), (N, 3), (N, 3)

    # Extract z, y, x coordinates
    base_x, base_y, base_z = base[..., 0] - shift_z_up, base[..., 1] - shift_y_back, base[..., 2] - shift_x_left  # (N, 3), (N, 3), (N, 3)

    # Calculate neighbor coordinates for all 8 neighbors
    neighbors_x = torch.stack([base_x, base_x + 1, base_x, base_x + 1, base_x, base_x + 1, base_x, base_x + 1], dim=-1)  # (N, 8)
    neighbors_y = torch.stack([base_y, base_y, base_y + 1, base_y + 1, base_y, base_y, base_y + 1, base_y + 1], dim=-1)  # (N, 8)
    neighbors_z = torch.stack([base_z, base_z, base_z, base_z, base_z + 1, base_z + 1, base_z + 1, base_z + 1], dim=-1)  # (N, 8)

    # Create validity mask: neighbors are valid if x,y, z are in [0, grid_size-1]
    # i.e., not -1 and not grid_size
    valid_mask = (neighbors_x > -1) & (neighbors_x < grid_size_x) & \
                 (neighbors_y > -1) & (neighbors_y < grid_size_y) & \
                 (neighbors_z > -1) & (neighbors_z < grid_size_z)  # (N, 8)

    neighbors_x_clamped = neighbors_x.clamp(0, grid_size_x - 1)
    neighbors_y_clamped = neighbors_y.clamp(0, grid_size_y - 1)
    neighbors_z_clamped = neighbors_z.clamp(0, grid_size_z - 1)


    # Calculate flat indices
    flat_indices = neighbors_x_clamped + neighbors_y_clamped * grid_size_x + neighbors_z_clamped * grid_size_y * grid_size_x
    # clamp to prevent out_julia of bounds in scatter_add
    indices = flat_indices.view(-1).long()

    # Calculate bilinear weights
    weights_2d = torch.stack([
        (1 - dx) * (1 - dy) * (1 - dz) * point_weights, # top-back-left
        dx * (1 - dy) * (1 - dz) * point_weights,       # top-back-right
        (1 - dx) * dy * (1 - dz) * point_weights,       # top-front-left
        dx * dy * (1 - dz) * point_weights,             # top-front-right
        (1 - dx) * (1 - dy) * dz * point_weights,       # bottom-back-left
        dx * (1 - dy) * dz * point_weights,             # bottom-back-right
        (1 - dx) * dy * dz * point_weights,             # bottom-front-left
        dx * dy * dz * point_weights,                   # bottom-front-right
    ], dim=-1)  # (N, 8)

    weights = weights_2d.view(8 * N)  # (8*N,)
    valid_mask_flat = valid_mask.view(8 * N)

    # Zero out_julia weights for invalid neighbors
    weights = weights * valid_mask_flat.to(dtype)

    # Scatter-add weights to the grid
    weight_field_flat = torch.zeros((np.prod(grid_size)), device=device, dtype=dtype)
    weight_field_flat = weight_field_flat.scatter_add(0, indices, weights)
    voxel_grid = weight_field_flat.view(grid_size_z, grid_size_y, grid_size_x)

    if sigma > 0:
        # Apply 3d convolution using through 3 1d convolutions
        voxel_grid = convolve_separable_3d(voxel_grid, kernel_size, sigma)
    return voxel_grid.contiguous().detach().cpu().numpy()

def quats_to_axis_angle(quats):
    """
    Convert quaternions to axis-angle representation.

    Args:
        quats (torch.Tensor): Quaternions with shape (N, 4) where each quaternion is [w, x, y, z]

    Returns:
        torch.Tensor: Axis-angle representations with shape (N, 4) where each row is [axis_x, axis_y, axis_z, angle_degrees]
    """

    # Normalize input quaternions
    quats = quats / torch.norm(quats, dim=1, keepdim=True)
    w = quats[:, 0]
    xyz = quats[:, 1:]

    # Compute angle in radians and convert to degrees
    angle_rad = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))
    angle_deg = angle_rad * 180.0 / torch.pi
    sin_half_angle = torch.sqrt(1.0 - w ** 2)
    # Compute axis, avoid division by zero
    # Create a mask for where sin_half_angle is too small
    mask = (sin_half_angle > 1e-8).unsqueeze(1)

    # Where sin_half_angle is large enough, divide xyz by it
    safe_axis = xyz / sin_half_angle.unsqueeze(1)

    # Where sin_half_angle is too small, use default axis [1,0,0]
    default_axis = torch.tensor([1.0, 0.0, 0.0], device=quats.device).expand_as(xyz)

    # Combine using the mask
    axis = torch.where(mask, safe_axis, default_axis)

    # Normalize axis to unit length
    axis = axis / torch.norm(axis, dim=1, keepdim=True)

    # Concatenate axis and angle to get the final result
    return torch.cat([axis, angle_deg.unsqueeze(1)], dim=1)


def axis_angle_to_quats(axis_angles):
    """
    Convert axis-angle representations to quaternions.

    Args:
        axis_angles (torch.Tensor): Axis-angle representations with shape (N, 4)
                                   where each row is [axis_x, axis_y, axis_z, angle_degrees]

    Returns:
        torch.Tensor: Quaternions with shape (N, 4) where each quaternion is [w, x, y, z]
    """
    import torch

    # Extract axis and angle
    axis = axis_angles[:, :3]  # First 3 elements are the axis
    angle_deg = axis_angles[:, 3]  # Last element is the angle in degrees

    # Normalize the axis vectors
    axis_norm = torch.norm(axis, dim=1, keepdim=True)
    # Create a mask for non-zero norms
    mask = axis_norm > 1e-8

    # Create normalized axis tensor
    normalized_axis = torch.zeros_like(axis)
    # Only normalize axes with non-zero norms
    normalized_axis = torch.where(mask, axis / axis_norm, normalized_axis)

    # Convert angle from degrees to radians
    angle_rad = angle_deg * torch.pi / 180.0

    # Compute half angles
    half_angle = angle_rad / 2.0

    # Compute sine and cosine of half angles
    cos_half = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)

    # Compute quaternion components
    w = cos_half
    xyz = normalized_axis * sin_half.unsqueeze(1)

    # Combine into quaternion [w, x, y, z]
    quats = torch.cat([w.unsqueeze(1), xyz], dim=1)

    # Ensure quaternions are normalized
    quats = quats / torch.norm(quats, dim=1, keepdim=True)

    return quats


def quat_to_matrix(q):
    """
    q: Tensor of shape (..., 4), assumed to be unit quaternions.
    Returns: (..., 3, 3) rotation matrices
    """
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)

    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    rot = torch.stack([
        torch.stack([1-2*(yy + zz),   2 * (xy - wz),     2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz),   1-2*(xx + zz),     2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy),   2 * (yz + wx),     1-2*(xx + yy)], dim=-1),
    ], dim=-2)

    return rot


def matrix_to_quat(rotation_matrices):
    """
    Convert rotation matrices to unit quaternions using Shepperd's method.

    Args:
        rotation_matrices: Tensor of shape (..., 3, 3) containing rotation matrices

    Returns:
        Tensor of shape (..., 4) containing quaternions in [x, y, z, w] format
    """
    # Get input shape and device
    input_shape = rotation_matrices.shape[:-2]
    device = rotation_matrices.device
    dtype = rotation_matrices.dtype

    # Flatten to handle batch dimensions
    R = rotation_matrices.view(-1, 3, 3)
    batch_size = R.shape[0]

    # Initialize output quaternion tensor
    q = torch.zeros(batch_size, 4, device=device, dtype=dtype)

    # Trace of rotation matrix
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0 (most stable when rotation angle is small)
    mask1 = trace > 0
    if torch.any(mask1):
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        q[mask1, 3] = 0.25 * s  # qw
        q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # qx
        q[mask1, 1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # qy
        q[mask1, 2] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # qz

    # Case 2: R[0,0] is largest diagonal element
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if torch.any(mask2):
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s = 4 * qx
        q[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # qw
        q[mask2, 0] = 0.25 * s  # qx
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # qy
        q[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # qz

    # Case 3: R[1,1] is largest diagonal element
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if torch.any(mask3):
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s = 4 * qy
        q[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # qw
        q[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # qx
        q[mask3, 1] = 0.25 * s  # qy
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # qz

    # Case 4: R[2,2] is largest diagonal element
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if torch.any(mask4):
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s = 4 * qz
        q[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # qw
        q[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # qx
        q[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # qy
        q[mask4, 2] = 0.25 * s  # qz

    # Normalize quaternions to ensure unit length
    q = q / torch.norm(q, dim=1, keepdim=True)

    # Reshape back to original batch dimensions
    return q.view(*input_shape, 4)

def fsc(vol1, vol2, n_bins=20):
    """
    Calculate Fourier Shell Correlation between two 3D volumes.

    Parameters:
    vol1, vol2: numpy arrays of shape (N, N, N)
        Two 3D volumes to compare
    n_bins: int, optional
        Number of frequency shell bins. If None, uses N//2 (default)

    Returns:
    frequencies: numpy array
        Spatial frequencies as fraction of Nyquist frequency
    correlations: numpy array
        FSC values at each frequency shell
    """
    # handle multiple volumes recursively
    if isinstance(vol1, list):
        print("hi from list")
        freqs = []
        fsc_vals = []
        for vol in vol1:
            freq, fsc_val = fsc(vol, vol2, n_bins=20)
            freqs.append(freq)
            fsc_vals.append(fsc_val)
        return freqs, fsc_vals

    assert vol1.shape == vol2.shape, "Volumes must have the same shape"
    assert len(vol1.shape) == 3, "Volumes must be 3D"
    n = vol1.shape[0]

    # Compute 3D FFTs
    fft1 = np.fft.fftshift(np.fft.fftn(vol1))
    fft2 = np.fft.fftshift(np.fft.fftn(vol2))

    # Create coordinate grids
    center = n // 2
    x, y, z = np.meshgrid(np.arange(n) - center,
                          np.arange(n) - center,
                          np.arange(n) - center, indexing='ij')

    # Calculate radial distances from center
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Define frequency shells (up to Nyquist)
    max_radius = center
    n_shells = n_bins if n_bins is not None else max_radius

    frequencies = []
    correlations = []

    for i in range(1, n_shells + 1):
        # Define shell boundaries
        r_inner = (i - 1) * max_radius / n_shells
        r_outer = i * max_radius / n_shells

        # Create mask for current shell
        mask = (r >= r_inner) & (r < r_outer)

        if not np.any(mask):
            continue

        # Extract values in this shell
        f1_shell = fft1[mask]
        f2_shell = fft2[mask]

        # Calculate correlation coefficient
        numerator = np.sum(f1_shell * np.conj(f2_shell)).real
        denom1 = np.sum(np.abs(f1_shell) ** 2).real
        denom2 = np.sum(np.abs(f2_shell) ** 2).real
        if denom1 > 0 and denom2 > 0:
            correlation = numerator / np.sqrt(denom1 * denom2)
            frequencies.append(i / n_shells)  # Fraction of Nyquist
            correlations.append(correlation)
        else:
            raise ValueError

    freqs, fsc_vals = np.array(frequencies), np.array(correlations)
    return freqs, fsc_vals

