import torch
import numpy as np

from .train import ParticleTomographyTrainer
from .model import ParticleTomographyModel
from .training_plan import build_simple_plan


def reconstruct( # public API TODO: add docstring
    images,
    rotations,
    shifts=None,
    num_points=None,
    particle_init_mode="isonormal",
    initial_points=None,
    initial_weights=None,
    kernel_size=3,
    training_plan=None,
    total_iterations=2000,
    batch_size=None,
    lr=2.5e-3,
    num_rejuvenates=1,
    rejuv_in_box=False,
    geom_start_fraction=1.0,
    device='cpu',
):
    dtype = torch.float32 # Currently only float32 supported

    # Convert everything to torch tensors and check shapes
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    images = images.to(dtype=dtype, device=device)

    if isinstance(rotations, np.ndarray):
        rotations = torch.from_numpy(rotations)
    rotations = rotations.to(dtype=dtype, device=device)

    N, H, W = images.shape

    if shifts is None:
        shifts = torch.zeros((N, 2), dtype=dtype, device=device)
    else:
        if isinstance(shifts, np.ndarray):
            shifts = torch.from_numpy(shifts)
        shifts = shifts.to(dtype=dtype, device=device)
        assert shifts.shape == (N, 2), f"shifts must have shape (N,2), got {shifts.shape}"

    if num_points is None:
        num_points = 10_000 + H*W//2            # simple heuristic

    if batch_size is None:
        batch_size = N  # full batch training by default

    if kernel_size is None:
        pixels = max(H, W)
        kernel_size = 3 + (pixels // 100) * 2   # simple heuristic

    # create model
    model = ParticleTomographyModel(
        images=images,
        rotations=rotations,
        shifts=shifts,
        num_points=num_points,
        particle_init_mode=particle_init_mode,
        initial_points=initial_points,
        initial_weights=initial_weights,
        kernel_size=kernel_size,
        dtype=torch.float32,
        device=device
    )

    # build training plan if not provided
    if training_plan is not None:
        plan = training_plan
    else:
        plan = build_simple_plan(total_iterations=total_iterations, batch_size=batch_size, lr=lr,
                                 geom_start_fraction=geom_start_fraction, num_rejuvenates=num_rejuvenates,
                                 rejuv_in_box=rejuv_in_box)

    # Create trainer and run training
    trainer = ParticleTomographyTrainer(model, plan.get_steps())
    trainer.fit()

    return trainer.get_model()
