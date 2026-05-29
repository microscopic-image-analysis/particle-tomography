from pathlib import Path

import numpy as np
import torch

from particle_tomography import ParticleTomographyModel, reconstruct
from particle_tomography.config import InputDataConfig
from particle_tomography.data import load_protein_data, load_protein_ground_truth
from particle_tomography.utils import fsc


ROOT = Path(__file__).resolve().parents[1]
PROTEIN_DIR = ROOT / "data" / "protein"


def tiny_inputs():
    images = np.ones((2, 4, 4), dtype=np.float32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], images.shape[0], axis=0)
    shifts = np.zeros((images.shape[0], 2), dtype=np.float32)
    return images, rotations, shifts


def test_bundled_protein_loader_shapes():
    paths = InputDataConfig(
        projection_file=PROTEIN_DIR / "projections_6b3r_wedge40_step-2.0_pixel-132_blur-0.6299118258855213.npz",
        rotations_file=PROTEIN_DIR / "projections_6b3r_wedge40_step-2.0_euler_angles.txt",
    )

    images, rotations, angles = load_protein_data(paths, return_angles=True)
    true_volume = load_protein_ground_truth(PROTEIN_DIR / "6b3r_pixel-132_blur-0.6299118258855213.npz")

    assert images.shape == (71, 132, 132)
    assert rotations.shape == (71, 3, 3)
    assert angles.shape == (71, 3)
    assert true_volume.shape == (132, 132, 132)
    assert images.dtype == np.float32
    assert true_volume.dtype == np.float32


def test_model_initialization_is_cpu_only_and_seed_isolated():
    images, rotations, shifts = tiny_inputs()
    images = torch.from_numpy(images)
    rotations = torch.from_numpy(rotations)
    shifts = torch.from_numpy(shifts)

    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)

    model = ParticleTomographyModel(
        images,
        rotations,
        shifts,
        num_points=5,
        device="cpu",
        random_seed=0,
    )

    assert torch.equal(torch.rand(4), expected)
    assert model.points.shape == (5, 3)
    assert model.get_shifts().shape == (2, 2)
    assert model.points.device.type == "cpu"


def test_tiny_reconstruction_runs_on_cpu_and_reports_r_factor():
    images, rotations, shifts = tiny_inputs()

    model = reconstruct(
        images,
        rotations,
        shifts=shifts,
        num_points=6,
        total_iterations=0,
        num_rejuvenates=0,
        batch_size=2,
        device="cpu",
        random_seed=0,
    )

    assert model.points.shape == (6, 3)
    assert np.isfinite(model.get_r_factor())


def test_model_save_and_load_roundtrip(tmp_path):
    images, rotations, shifts = tiny_inputs()
    model = reconstruct(
        images,
        rotations,
        shifts=shifts,
        num_points=6,
        total_iterations=0,
        num_rejuvenates=0,
        batch_size=2,
        device="cpu",
        random_seed=0,
    )

    path = tmp_path / "model.pt"
    model.save_model(path)
    loaded = ParticleTomographyModel.from_saved_state(path, device="cpu")

    assert loaded.num_points == model.num_points
    assert loaded.points.shape == model.points.shape
    assert loaded.get_shifts().shape == (1, 2)
    assert torch.allclose(loaded.points.detach().cpu(), model.points.detach().cpu())


def test_fsc_identical_volume_is_one():
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(4, 4, 4)).astype(np.float32)
    frequencies, correlations = fsc(volume, volume, n_bins=2)

    assert frequencies.shape == correlations.shape
    assert np.allclose(correlations, 1.0)
