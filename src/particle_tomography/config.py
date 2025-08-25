from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Callable
import torch
import numpy as np


# ========== Input Data Configuration ==========
@dataclass
class InputDataConfig:
    projection_file: Path                        # Path to projection images
    rotations_file: Path                         # Path to rotations for each image
    shifts_file: Optional[Path] = None           # Optional shifts
    initial_points_file: Optional[Path] = None   # Optional initial points

# ========== Model Configuration ==========
@dataclass
class ModelConfig:
    n_particles: int = 5000                         # Number of particles to use in the model
    particle_init_mode: str = "isonormal"           # Particle initialization mode (e.g., "isonormal", "thinfilm")
    kernel_size: int = 3                            # kernel_size used in convolution. Has to be odd.
    dtype: torch.dtype = torch.float32              # PyTorch dtype to use. Currently only float32 supported
    device: str = "cpu"                             # Device to use ("cpu" or "cuda")


# ========== Training Step Configs ==========
@dataclass
class StepConfig:
    """Base class for all training steps"""
    pass

@dataclass
class TrainingStep(StepConfig):
    n_iterations: int               # Number of optimization iterations
    learn_rate: float               # Learning rate for gradient-based optimization
    batch_size: int                 # Batch size used in training
    geom_start_fraction: float = 1.0  # fraction of iteration after which to optimize for geometry (i.e. rotations, shifts)

@dataclass
class GMMRejuvenateStep(StepConfig):
    rejuv_in_box: bool = True  # Whether to restrict to a box

@dataclass
class SaveImagesStep:
    out_dir: Path
    slice_thickness: int = 10       # slice thickness used to make projections
    logging_prefix: str = "final"
    true_volume_path: Optional[Path] = None
    true_volume_loader: Optional[Callable[[Path], np.ndarray]] = None

    def load_true_volume(self) -> Optional[np.ndarray]:
        if self.true_volume_path and self.true_volume_loader:
            return self.true_volume_loader(self.true_volume_path)
        return None

    def __post_init__(self):
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class TrainingConfig:
    steps: List[Union[TrainingStep, GMMRejuvenateStep]] = field(default_factory=list)


@dataclass
class ParticleTomographyConfig:
    input_paths: InputDataConfig               # Nested config with paths to input data
    model: ModelConfig                         # Model-related parameters
    training: TrainingConfig                   # Training logic and steps
