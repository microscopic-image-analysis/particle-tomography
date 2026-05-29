import argparse
from pathlib import Path
import time

import torch

from particle_tomography.runner import reconstruct
from particle_tomography.training_plan import build_plan_from_config
from particle_tomography.config import (
    TrainingStep,
    GMMRejuvenateStep,
    ParticleTomographyConfig,
    InputDataConfig,
    ModelConfig,
    TrainingConfig,
)
from particle_tomography.data import load_protein_data, load_protein_ground_truth

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "protein"
TRUE_VOL_PATH = DATA_DIR / "6b3r_pixel-132_blur-0.6299118258855213.npz"


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
            device="auto",
            dtype=torch.float32,
            random_seed=0,
        ),
        training=TrainingConfig(
            steps=[
                TrainingStep(1250, 2.5e-3, 71, 1.0),
                GMMRejuvenateStep(rejuv_in_box=True),
                TrainingStep(1250, 2.5e-3, 71, 1.0),
            ]
        )
    )


def main(plot=False, device=None):
    config = build_protein_config()
    if device is not None:
        config.model.device = device

    images, rotations, _ = load_protein_data(config.input_paths, return_angles=True)

    training_plan = build_plan_from_config(config.training)
    model = reconstruct(
        images,
        rotations,
        shifts=None,
        num_points=config.model.n_particles,
        particle_init_mode=config.model.particle_init_mode,
        kernel_size=config.model.kernel_size,
        training_plan=training_plan,
        device=config.model.device,
        random_seed=config.model.random_seed,
    )
    model.save_model(PROJECT_ROOT / "model_protein.pt")

    true_volume = load_protein_ground_truth(TRUE_VOL_PATH)
    _, fsc_values = model.get_fsc(true_volume)
    print(f"FSC at Nyquist: {fsc_values[-1]:.3f}")

    if plot:
        model.plot_points()
        # model.plot_weights()
        model.plot_volume()
        model.plot_fsc(true_volume)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the protein reconstruction example.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: 'auto', 'cpu', 'cuda', or a torch device string such as 'cuda:0'.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open interactive point, volume, and FSC plots after reconstruction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    main(plot=args.plot, device=args.device)
    t2 = time.time()
    print(f"Execution time: {t2 - t1:.2f} seconds")
