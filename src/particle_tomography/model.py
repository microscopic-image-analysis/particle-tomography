import torch
import torch.nn as nn
import torch.nn.functional as F

from rasterizer import DifferentiableRasterizer

from particle_tomography.utils import fsc
from particle_tomography.utils import quat_to_matrix, matrix_to_quat
from particle_tomography.plot import plot_volume, plot_points, plot_weights, plot_fsc
from particle_tomography.raster3d import rasterize_and_smooth_3d_blocked

class ParticleTomographyModel(nn.Module):
    def __init__(
            self, images, rotations, shifts=None, num_points=3000,
            initial_points=None, initial_weights=None, particle_init_mode="isonormal",
            kernel_size=3, start_bandwidth=1.0, dtype=torch.float32, device='cpu',
    ):
        super().__init__()
        self.num_points = num_points
        self.num_images = images.shape[0]   # N
        self.grid_size_y = images.shape[1]  # H
        self.grid_size_x = images.shape[2]  # W
        self.kernel_size = kernel_size      # kernel size used in convolution
        print("Initializing model with kernel size", self.kernel_size)
        self.dtype = torch.float32 # currently only float32 supported
        self.device = device
        self.rasterizer = DifferentiableRasterizer((self.grid_size_y, self.grid_size_x,), kernel_size=self.kernel_size,
                                                   device=self.device)

        # Initialize 3D points (Gaussian centers)
        if initial_points is None:
            self.points = nn.Parameter(
                self.init_particles(n_particles=num_points, mode=particle_init_mode, device=device, dtype=dtype)
            )
        else:
            self.points = nn.Parameter(initial_points.clone().to(dtype=dtype, device=device))
            self.num_points = self.points.shape[0]

        # Initialize log weights for each point
        if initial_weights is None:
            self.log_point_weights = nn.Parameter(
                torch.zeros(self.num_points, dtype=dtype, device=device)  # exp(0) = 1
            )
        else:
            self.log_point_weights = nn.Parameter(
                torch.log(initial_weights.clone().detach()).to(dtype=dtype, device=device)
            )

        self.register_buffer('images', images)  # shape: (N, H, W)
        quats = matrix_to_quat(rotations)  # [x,y,z,w]
        self.rotation_quats = nn.Parameter(quats)
        self.shifts = nn.Parameter(shifts) # shape: (N, 2)

        # Compute initial scaling factors for images
        init_value = 2 * torch.log(torch.tensor(self.grid_size_x, dtype=dtype, device=device)) + torch.log(
            torch.mean(self.images))
        self.log_scale = nn.Parameter(
            torch.full((self.num_images,), init_value, dtype=dtype, device=device)
        )

        # Global noise parameter
        self.log_noise_std = nn.Parameter(
            torch.log(self.images.std())
        )

        # Radius of particles (global)
        log_start_bandwidth = torch.log(
            torch.tensor(start_bandwidth, dtype=dtype, device=device)
        )
        self.log_bandwidth = nn.Parameter(log_start_bandwidth)


    @property
    def rotations_projections(self):
        return quat_to_matrix(self.rotation_quats)[:, :2, :].contiguous()

    @property
    def point_weights(self):
        """Get weights from log_weights using softmax"""
        return torch.softmax(self.log_point_weights, dim=0)

    @property
    def noise_std(self):
        return torch.exp(self.log_noise_std)

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    @property
    def bandwidth(self):
        return torch.exp(self.log_bandwidth)

    # getters
    def get_current_state(self):
        return (self.points.detach().cpu(), self.point_weights.detach().cpu(), self.bandwidth.detach().cpu(),
                self.noise_std.detach().cpu(), self.scale.detach().cpu())

    def get_shifts(self):
        return self.shifts.detach().cpu()


    def get_volume(self):
        """Convert internal volume representation (GMM) to voxel representation."""
        x = self.grid_size_x
        y = self.grid_size_y
        z = max (self.grid_size_x, self.grid_size_y)
        gridsize = (z, y, x)
        vol = rasterize_and_smooth_3d_blocked(self.points.detach(), self.point_weights.detach(), gridsize, sigma=self.bandwidth,
                                   kernel_size=self.kernel_size, device=self.device)
        vol = vol * self.scale.mean().detach().cpu().numpy() # rescale
        return vol

    def get_volume_sparse(self):
        return self.points.detach().cpu(), self.point_weights.detach().cpu(), self.bandwidth.detach().cpu()

    def get_fsc(self, true_vol):
        vol =  self.get_volume()
        freqs, fsc_values = fsc(vol, true_vol)
        return freqs, fsc_values

    def get_r_factor(self):
        indices = range(self.num_images)
        projected, target = self.forward(indices)
        abs_diff = torch.abs(projected - target).sum().item()
        abs_obs = torch.abs(target).sum().item()
        r_factor = abs_diff / abs_obs if abs_obs > 0 else float('inf')
        return r_factor

    # wrappers for visualization
    def plot_volume(self):
        vol = self.get_volume()
        plot_volume(vol)

    def plot_points(self):
        points, _, _ = self.get_volume_sparse()
        plot_points(points)

    def plot_weights(self):
        _, weights, _ =self.get_volume_sparse()
        plot_weights(weights)

    def plot_fsc(self, true_vol, show_plot=True):
        freqs, fsc_values = self.get_fsc(true_vol)
        fig = plot_fsc(freqs, fsc_values)
        if show_plot:
            fig.show()
        return fig

    def save_model(self, path: str):
        """Save relevant model state (points, weights, bandwidth, noise, scale)."""
        state = {
            "points": self.points.detach().cpu(),
            "point_weights": self.point_weights.detach().cpu(),
            "bandwidth": self.bandwidth.detach().cpu(),
            "noise_std": self.noise_std.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "kernel_size": self.kernel_size,
        }
        torch.save(state, path)
        print(f"[ParticleTomographyModel] Saved model state to {path}")

    @classmethod
    def from_saved_state(cls, path: str, images, rotations, shifts=None,
                         dtype=torch.float32, device='cpu',
                         **kwargs):
        """Initialize model from a saved state file."""
        state = torch.load(path, map_location=device)

        # Create a new model instance
        model = cls(
            images=images,
            rotations=rotations,
            shifts=shifts,
            kernel_size=state["kernel_size"],
            dtype=dtype,
            device=device,
            **kwargs
        )

        # Load saved parameters into model
        model.points.data = state["points"].to(device=device, dtype=dtype)
        model.log_point_weights.data = torch.log(state["point_weights"].to(device=device, dtype=dtype))
        model.log_bandwidth.data = torch.log(state["bandwidth"].to(device=device, dtype=dtype))
        model.log_noise_std.data = torch.log(state["noise_std"].to(device=device, dtype=dtype))
        model.log_scale.data = torch.log(state["scale"].to(device=device, dtype=dtype))

        print(f"[ParticleTomographyModel] Loaded model state from {path}")
        return model

    def forward(self, indices):
        """
        indices: Tensor of image indices to use in batch.
        """
        imgs = self.images[indices]  # (B, H, W)
        rots = self.rotations_projections[indices]  # (B, 2, 3)
        shifts = self.shifts[indices]  # (B, 2)

        projections = self.rasterizer(points=self.points, weights=self.point_weights, rotations=rots,
                                      translations=shifts, bandwidth=self.bandwidth)
        scale = self.scale[indices].unsqueeze(-1).unsqueeze(-1)
        scaled_projection_imgs = scale * projections
        return scaled_projection_imgs, imgs


    def loss(self, projected_images, target_images):
        """
        Compute L1 loss.

        Args:
            projected_images (Tensor): Model predictions of shape (N, H, W)
            target_images (Tensor): Ground truth images of the same shape

        Returns:
            Tensor: Scalar loss (negative log-likelihood up to constant)
        """
        residuals = projected_images - target_images
        abs_residuals = torch.sum(torch.abs(residuals))  # sum over all pixels and images
        loss = abs_residuals * torch.exp(-self.log_noise_std) + residuals.numel() * self.log_noise_std
        return loss


    def init_particles(self, n_particles, mode, dtype=torch.float32, device='cpu'):
        """
        Initialize particles based on the specified mode.

        Args:
            n_particles: Number of particles to initialize
            mode: Initialization mode ('isonormal' or 'thinfilm')
            dtype: torch data type (default: torch.float32)
            device: torch device (default: 'cpu')

        Returns:
            torch.Tensor: Shape (n_particles, 3) containing initialized particle positions
        """
        torch.manual_seed(0)
        if mode == 'isonormal':
            particles = 0.3 * torch.randn(n_particles, 3, dtype=dtype, device=device)

        elif mode == 'thinfilm':
            particles = torch.zeros(n_particles, 3, dtype=dtype, device=device)
            particles[:, 0] = 4 * torch.rand(n_particles, dtype=dtype, device=device) - 2
            particles[:, 1] = 2 * torch.rand(n_particles, dtype=dtype, device=device) - 1
            particles[:, 2] = 0.2 * torch.randn(n_particles, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown particle initialization mode: {mode}")

        return particles


    def rejuvenate_GMM(self, rejuv_in_box=True):
        """
        Resample the particles/points inside parameters according to the specified mode.
        mode: "GMM": resample points by interpreting particle positions, weights and bandwidth as parameters
        of a Gaussian Mixture model. New points are sampled from that model and their weights are reset to uniform.
        mode: "backproject_filtered_residuals": resample lowest-weight points to positions with highest
        backprojected residual deviation using filtered backprojection.
        """
        log_weights = self.log_point_weights.detach().clone()
        if rejuv_in_box:
            inside_mask = ((self.points >= -1) & (self.points <= 1)).all(dim=1)
            log_weights[~inside_mask] = -float('inf')

        weights = F.softmax(log_weights, dim=0).to(dtype=torch.float64)
        n_points = weights.size(0)

        dist = torch.distributions.Categorical(probs=weights)
        new_point_idxs = dist.sample((n_points,))
        selected_points = self.points[new_point_idxs]

        # convert bandwith from pixel units to normalized world units (side with the most pixels is mapped to [0, 1])
        scale = self.bandwidth  / max(self.grid_size_x, self.grid_size_y)
        gaussian_noise = scale * torch.randn(n_points, 3, device=self.points.device,
                                                      dtype=self.points.dtype)

        new_points = selected_points + gaussian_noise
        self.points = torch.nn.Parameter(new_points.detach())

        self.log_point_weights = torch.nn.Parameter(
            torch.zeros_like(self.log_point_weights).detach()
        )