from pathlib import Path
import time
import torch
import numpy as np
from particle_tomography import ParticleTomographyModel

from particle_tomography.runner import reconstruct
from particle_tomography.training_plan import build_plan_from_config
from particle_tomography.config import TrainingStep, GMMRejuvenateStep, SaveImagesStep, \
    ParticleTomographyConfig, InputDataConfig, ModelConfig, TrainingConfig
from data.protein_loader import load_protein_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # two levels up from scripts/
DATA_DIR = PROJECT_ROOT / "data" / "protein"
OUT_DIR = PROJECT_ROOT / "out" / "protein"
TRUE_VOL_PATH = DATA_DIR / "6b3r_pixel-132_blur-0.6299118258855213.npz"


def load_protein_ground_truth(path):
    """Load protein ground truth volume from .npz file"""
    true_volume = np.load(path).transpose(2, 1, 0).copy()
    return true_volume.astype(np.float32)


def build_protein_config():
    return ParticleTomographyConfig(
        input_paths=InputDataConfig(
            projection_file=DATA_DIR / "projections_6b3r_wedge40_step-2.0_pixel-132_blur-0.6299118258855213.npz",
            rotations_file=DATA_DIR / "projections_6b3r_wedge40_step-2.0_euler_angles.txt",
        ),
        model=ModelConfig(
            n_particles=20000,
            kernel_size=3,
            particle_init_mode="isonormal",
            device="cuda",
            dtype=torch.float32
        ),
        training=TrainingConfig(
            steps=[
                TrainingStep(1250, 2.5e-3, 71, 1.0),
                GMMRejuvenateStep(rejuv_in_box=True),
                TrainingStep(1250, 2.5e-3, 71, 1.0),
                # SaveImagesStep(
                #     out_dir=OUT_DIR,
                #     slice_thickness=10,
                #     logging_prefix="final",
                #     true_volume_path=TRUE_VOL_PATH,
                #     true_volume_loader=load_protein_ground_truth
                # ),
            ]
        )
    )


def main():
    config = build_protein_config()
    images, rotations, angles = load_protein_data(config.input_paths, return_angles=True)

    training_plan = build_plan_from_config(config.training)
    model = reconstruct(images,
                        rotations,
                        shifts=None,
                        num_points=config.model.n_particles,
                        particle_init_mode=config.model.particle_init_mode,
                        kernel_size=config.model.kernel_size,
                        training_plan=training_plan,
                        device=config.model.device,
                        )
    model.save_model("model_protein.pt")
    # model = ParticleTomographyModel.from_saved_state("model_protein.pt")

    model.plot_points()
    # model.plot_weights()
    model.plot_volume()
    points, weights, bandwidth = model.get_volume_sparse()


    true_volume = load_protein_ground_truth(TRUE_VOL_PATH)
    model.plot_fsc(true_volume)

if __name__ == "__main__":
    torch.manual_seed(0)
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Execution time: {t2 - t1:.2f} seconds")