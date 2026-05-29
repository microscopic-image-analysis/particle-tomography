import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterizer import DifferentiableRasterizer

from particle_tomography.utils import fsc
from particle_tomography.utils import quat_to_matrix, matrix_to_quat
from particle_tomography.plot import plot_volume, plot_weights, plot_fsc, plot_points
from particle_tomography.raster3d import rasterize_and_smooth_3d_blocked

class ParticleTomographyModel(nn.Module):
    """Sparse particle model for tilt-series tomographic reconstruction.

    The model represents a 3D volume as weighted particles and optimizes
    particle positions, weights, image scales, noise, rotations, and shifts
    against observed projection images.
    """

    def __init__(
            self, images, rotations, shifts=None, num_points=3000,
            initial_points=None, initial_weights=None, particle_init_mode="isonormal",
            kernel_size=3, start_bandwidth=1.0, dtype=torch.float32, device='cpu',
            random_seed=None,
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
        self.generator = self._make_generator(device=device, random_seed=random_seed)
        self.rasterizer = DifferentiableRasterizer((self.grid_size_y, self.grid_size_x,), kernel_size=self.kernel_size,
                                                   device=self.device)

        # Initialize 3D points (Gaussian centers)
        if initial_points is None:
            self.points = nn.Parameter(
                self.init_particles(
                    n_particles=num_points,
                    mode=particle_init_mode,
                    device=device,
                    dtype=dtype,
                    generator=self.generator,
                )
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
        if shifts is None:
            shifts = torch.zeros((self.num_images, 2), dtype=dtype, device=device)
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


    @staticmethod
    def _make_generator(device, random_seed):
        if random_seed is None:
            return None

        torch_device = torch.device(device)
        if torch_device.type not in {"cpu", "cuda"}:
            raise ValueError(f"random_seed is only supported for CPU/CUDA devices, got {device!r}")

        generator = torch.Generator(device=torch_device)
        generator.manual_seed(random_seed)
        return generator


    @property
    def rotations_projections(self):
        return quat_to_matrix(self.rotation_quats)[:, :2, :].contiguous()

    @property
    def rotations(self):
        return quat_to_matrix(self.rotation_quats)

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
        """Return detached CPU copies of the optimized model parameters."""
        return (self.points.detach().cpu(), self.point_weights.detach().cpu(), self.bandwidth.detach().cpu(),
                self.noise_std.detach().cpu(), self.scale.detach().cpu())

    def get_shifts(self):
        """Return detached CPU copies of the optimized image shifts."""
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
        """Return the sparse particle representation as points, weights, and bandwidth."""
        return self.points.detach().cpu(), self.point_weights.detach().cpu(), self.bandwidth.detach().cpu()

    def get_fsc(self, true_vol):
        """
        Compute the Fourier Shell Correlation (FSC) between the reconstructed
        volume from this object and a provided ground-truth volume.
        """
        vol = self.get_volume()
        freqs, fsc_values = fsc(vol, true_vol)
        return freqs, fsc_values

    def get_r_factor(self):
        """Compute the R-factor between current projections and observed images."""
        indices = range(self.num_images)
        projected, target = self.forward(indices)
        abs_diff = torch.abs(projected - target).sum().item()
        abs_obs = torch.abs(target).sum().item()
        r_factor = abs_diff / abs_obs if abs_obs > 0 else float('inf')
        return r_factor

    # wrappers for visualization
    def plot_volume(self, camera=None, path=None):
        """Render the reconstructed dense volume interactively or save it to ``path``."""
        vol = self.get_volume()
        plot_volume(vol, camera=camera, path=path)

    def plot_points(self, bounding_box=(1.25, 1.25, 1.25)):
        """Plot the current particle locations."""
        points, _, _ = self.get_volume_sparse()
        plot_points(points, side=bounding_box)

    def plot_weights(self):
        """Plot the current particle weights."""
        _, weights, _ =self.get_volume_sparse()
        plot_weights(weights)

    def plot_fsc(self, true_vol, show_plot=True):
        """Plot the Fourier Shell Correlation against a ground-truth volume."""
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
            "grid_size_x": self.grid_size_x,
            "grid_size_y": self.grid_size_y
        }
        torch.save(state, path)
        print(f"[ParticleTomographyModel] Saved model state to {path}")

    @classmethod
    def from_saved_state(cls, path: str, images=None, rotations=None, shifts=None,
                         dtype=torch.float32, device='cpu',
                         **kwargs):
        """Initialize model from a saved state file."""
        state = torch.load(path, map_location=device)
        if images is None:
            images = torch.zeros((1, state["grid_size_y"], state["grid_size_x"]), dtype=dtype, device=device) # 1 dummy image
        if rotations is None:
            rotations = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) # 1 dummy rotation
        if shifts is None:
            shifts = torch.zeros((images.shape[0], 2), dtype=dtype, device=device)

        # Create a new model instance
        model = cls(
            images=images,
            rotations=rotations,
            shifts=shifts,
            num_points=state["points"].shape[0],
            kernel_size=state["kernel_size"],
            dtype=dtype,
            device=device,
            **kwargs
        )

        # Load saved parameters into model
        points = state["points"].to(device=device, dtype=dtype)  # shape (N, 3)
        point_weights = state["point_weights"].to(device=device, dtype=dtype)  # shape (N,)
        bandwidth = state["bandwidth"].to(device=device, dtype=dtype)
        noise_std = state["noise_std"].to(device=device, dtype=dtype)
        scale = state["scale"].to(device=device, dtype=dtype)

        # set parameters
        model.points.data = points
        model.log_point_weights.data = torch.log(point_weights)
        model.log_bandwidth.data = torch.log(bandwidth)
        model.log_noise_std.data = torch.log(noise_std)
        model.log_scale.data = torch.log(scale)

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
        """
        residuals = projected_images - target_images
        abs_residuals = torch.sum(torch.abs(residuals))  # sum over all pixels and images
        loss = abs_residuals * torch.exp(-self.log_noise_std) + residuals.numel() * 2 * self.log_noise_std
        return loss


    def init_particles(self, n_particles, mode, dtype=torch.float32, device='cpu', generator=None):
        """Initialize particles without mutating PyTorch's global random state."""
        if mode == 'isonormal':
            particles = 0.3 * torch.randn(n_particles, 3, dtype=dtype, device=device, generator=generator)

        elif mode == 'thinfilm':
            particles = torch.zeros(n_particles, 3, dtype=dtype, device=device)
            particles[:, 0] = 4 * torch.rand(n_particles, dtype=dtype, device=device, generator=generator) - 2
            particles[:, 1] = 2 * torch.rand(n_particles, dtype=dtype, device=device, generator=generator) - 1
            particles[:, 2] = 0.2 * torch.randn(n_particles, dtype=dtype, device=device, generator=generator)
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

        new_point_idxs = torch.multinomial(
            weights,
            n_points,
            replacement=True,
            generator=self.generator,
        )
        selected_points = self.points[new_point_idxs]

        # convert bandwith from pixel units to normalized world units (side with the most pixels is mapped to [0, 1])
        scale = self.bandwidth  / max(self.grid_size_x, self.grid_size_y)
        gaussian_noise = scale * torch.randn(
            n_points,
            3,
            device=self.points.device,
            dtype=self.points.dtype,
            generator=self.generator,
        )

        new_points = selected_points + gaussian_noise
        self.points = torch.nn.Parameter(new_points.detach())

        self.log_point_weights = torch.nn.Parameter(
            torch.zeros_like(self.log_point_weights).detach()
        )