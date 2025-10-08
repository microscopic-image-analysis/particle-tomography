import torch
import torch.nn.functional as F
import numpy as np
from particle_tomography.plot import show_images, plot_volume


def rasterize_3d_blocked(points, point_weights=None, grid_size=(64, 64, 64),
                         block_size=256, dtype=torch.float32, device="cpu"):
    """
    Converts 3D point cloud to voxel grid with trilinear splatting using blocked processing.

    Args:
        points: [N, 3] coordinates in [-1, 1] range
        point_weights: [N] point weights
        grid_size: (H, D, W) - (Z, Y, X)
        block_size: Size of each cubic block to process at once
        dtype: Data type for computations
        device: Device to perform computations on

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
    rescaled_points[:, 0] -= shift_x_left
    rescaled_points[:, 1] -= shift_y_back
    rescaled_points[:, 2] -= shift_z_up
    block_shifted_points = torch.zeros_like(rescaled_points, dtype=dtype, device=device)

    # Initialize output grid
    voxel_grid = np.zeros(grid_size, dtype=np.float32)

    # Calculate number of blocks in each dimension
    num_blocks_x = (grid_size_x + block_size - 1) // block_size
    num_blocks_y = (grid_size_y + block_size - 1) // block_size
    num_blocks_z = (grid_size_z + block_size - 1) // block_size


    # Process each block
    for bz in range(num_blocks_z):
        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                # Define block boundaries in grid coordinates
                z_start = bz * block_size
                z_end = min((bz + 1) * block_size, grid_size_z)
                y_start = by * block_size
                y_end = min((by + 1) * block_size, grid_size_y)
                x_start = bx * block_size
                x_end = min((bx + 1) * block_size, grid_size_x)

                # Shift points to local block coordinates
                block_shifted_points[:, 0] = rescaled_points[:, 0] - x_start
                block_shifted_points[:, 1] = rescaled_points[:, 1] - y_start
                block_shifted_points[:, 2] = rescaled_points[:, 2] - z_start

                # Process this block
                block_result = _raster_block(block_shifted_points, point_weights, (block_size, block_size, block_size), dtype, device)
                # Trim block_result if at edges
                block_result = block_result[
                               :z_end - z_start,
                               :y_end - y_start,
                               :x_end - x_start
                               ]

                # Add to main grid
                voxel_grid[z_start:z_end, y_start:y_end, x_start:x_end] += block_result

    return voxel_grid


def _raster_block(points, point_weights, grid_size=(64, 64, 64), dtype=torch.float32, device="cpu"):

    # bring inputs to device
    points = torch.from_numpy(points) if isinstance(points, np.ndarray) else points
    points = points.to(dtype=dtype, device=device)


    N, _ = points.shape
    grid_size_z, grid_size_y, grid_size_x = grid_size

    base = points.floor().int()  # (N, 3)
    delta = points - base  # (N, 3)
    dx, dy, dz = delta[..., 0], delta[..., 1], delta[..., 2]  # (N, 3), (N, 3), (N, 3)

    # Extract z, y, x coordinates
    base_x, base_y, base_z = base[..., 0], base[..., 1], base[..., 2] # (N, 3), (N, 3), (N, 3)

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

    return voxel_grid.contiguous().detach().cpu().numpy()


######################################################################################################
######################################################################################################
######################################################################################################


def compute_gaussian_kernel_1d(kernel_size, sigma, device, dtype):
    """
    Computes 1D Gaussian kernel dynamically (differentiable).

    Args:
        kernel_size (int): Size of the kernel
        sigma (torch.Tensor or float): Standard deviation of the Gaussian
        device: Device to place the tensor on
        dtype: Data type of the tensor

    Returns:
        torch.Tensor: 1D Gaussian kernel, normalized to sum to 1
    """
    ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=dtype)
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor(sigma, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def convolve_separable_3d_blocked(voxel_grid_np, kernel_size, sigma, block_size=32, device="cpu"):
    """
    Applies separable 3D convolution using blocked processing without allocating full tensor on GPU.
    Processes blocks individually and writes results directly to CPU numpy array.

    Args:
        voxel_grid_np (np.ndarray): Input voxel grid on CPU
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian
        block_size (int): Size of processing blocks
        device (str): Device for computation

    Returns:
        np.ndarray: Convolved voxel grid (same memory location as input)
    """
    # Create output array on CPU
    result = np.zeros_like(voxel_grid_np)

    # Convert to tensor for kernel computation
    dummy_tensor = torch.tensor(0.0, device=device, dtype=torch.float32)
    kernel_1d = compute_gaussian_kernel_1d(kernel_size, sigma, device, dummy_tensor.dtype)
    r = kernel_size // 2

    grid_d, grid_h, grid_w = voxel_grid_np.shape

    # Calculate number of blocks in each dimension
    num_blocks_d = (grid_d + block_size - 1) // block_size
    num_blocks_h = (grid_h + block_size - 1) // block_size
    num_blocks_w = (grid_w + block_size - 1) // block_size

    # Process each block individually
    for bd in range(num_blocks_d):
        for bh in range(num_blocks_h):
            for bw in range(num_blocks_w):
                # Define block boundaries
                d_start = bd * block_size
                d_end = min((bd + 1) * block_size, grid_d)
                h_start = bh * block_size
                h_end = min((bh + 1) * block_size, grid_h)
                w_start = bw * block_size
                w_end = min((bw + 1) * block_size, grid_w)

                # Define extended boundaries for convolution (with padding)
                d_start_ext = max(0, d_start - r)
                d_end_ext = min(grid_d, d_end + r)
                h_start_ext = max(0, h_start - r)
                h_end_ext = min(grid_h, h_end + r)
                w_start_ext = max(0, w_start - r)
                w_end_ext = min(grid_w, w_end + r)

                # Extract extended block and move to GPU
                block_ext_np = voxel_grid_np[d_start_ext:d_end_ext,
                               h_start_ext:h_end_ext,
                               w_start_ext:w_end_ext]
                block_ext = torch.from_numpy(block_ext_np).to(device=device, dtype=torch.float32)

                # Apply convolution to this block only
                block_conv = convolve_separable_3d_single_block(block_ext, kernel_1d, r)

                # Calculate extraction indices
                d_offset = d_start - d_start_ext
                h_offset = h_start - h_start_ext
                w_offset = w_start - w_start_ext

                valid_d_size = d_end - d_start
                valid_h_size = h_end - h_start
                valid_w_size = w_end - w_start

                # Extract valid portion and move back to CPU
                valid_result = block_conv[
                               d_offset:d_offset + valid_d_size,
                               h_offset:h_offset + valid_h_size,
                               w_offset:w_offset + valid_w_size
                               ]

                # Write directly to CPU result array
                result[d_start:d_end, h_start:h_end, w_start:w_end] = valid_result.detach().cpu().numpy()

                # Clean up GPU memory for this block
                del block_ext, block_conv, valid_result
                if device != "cpu":
                    torch.cuda.empty_cache()

    return result


def convolve_separable_3d_single_block(block, kernel_1d, r):
    """
    Apply separable 3D convolution to a single block.

    Args:
        block (torch.Tensor): Block of shape (D, H, W)
        kernel_1d (torch.Tensor): 1D Gaussian kernel
        r (int): Kernel radius

    Returns:
        torch.Tensor: Convolved block
    """
    # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
    block = block.unsqueeze(0).unsqueeze(0)

    # Prepare 1D kernels as 3D conv filters
    kernel_z = kernel_1d.view(1, 1, -1, 1, 1)  # (1, 1, k, 1, 1) - convolve along depth
    kernel_y = kernel_1d.view(1, 1, 1, -1, 1)  # (1, 1, 1, k, 1) - convolve along height
    kernel_x = kernel_1d.view(1, 1, 1, 1, -1)  # (1, 1, 1, 1, k) - convolve along width

    # Apply separable convolution in three passes
    # First: convolve along Z (depth) dimension
    out = F.conv3d(block, kernel_z, padding=(r, 0, 0))

    # Second: convolve along Y (height) dimension
    out = F.conv3d(out, kernel_y, padding=(0, r, 0))

    # Third: convolve along X (width) dimension
    out = F.conv3d(out, kernel_x, padding=(0, 0, r))

    # Remove batch and channel dimensions: (1, 1, D, H, W) -> (D, H, W)
    return out.squeeze(0).squeeze(0)


def rasterize_and_smooth_3d_blocked(points, point_weights=None, grid_size=(64, 64, 64),
                                    sigma=0.001, kernel_size=3, block_size=256,
                                    dtype=torch.float32, device="cpu"):
    """
    Complete blocked version of rasterization and smoothing.
    """
    # First do blocked rasterization
    voxel_grid = rasterize_3d_blocked(points, point_weights, grid_size,
                                      block_size, dtype, device)

    if sigma > 0:
        # Apply blocked convolution without full GPU allocation
        voxel_grid = convolve_separable_3d_blocked(
            voxel_grid, kernel_size, sigma, block_size, device
        )

    return voxel_grid


if __name__ == "__main__":
    N = 500
    grid_size = (40, 40, 40)
    weights = torch.ones(N)
    points = torch.rand((N,3)) * 2 - 1
    points[0,0] = 0.0
    points[0,1] = 0.0
    points[0,2] = 0.0
    print("done with first")
    raster2 = rasterize_and_smooth_3d_blocked(points, weights, grid_size, block_size=4, sigma=10, device="cuda")
