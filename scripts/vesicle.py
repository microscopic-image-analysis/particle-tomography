from pathlib import Path
import mrcfile
import time
import torch
import numpy as np
import sys
import pathlib

from particle_tomography import reconstruct
from particle_tomography.training_plan import build_plan_from_config
from particle_tomography.config import TrainingStep, GMMRejuvenateStep, SaveImagesStep, \
    ParticleTomographyConfig, InputDataConfig, ModelConfig, TrainingConfig
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from data.loader import load_vesicle_data
from particle_tomography.plot import show_images, plot_volume

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # two levels up from scripts/
DATA_DIR = PROJECT_ROOT / "data" / "vesicle"
OUT_DIR = PROJECT_ROOT / "out" / "vesicle"
TRUE_VOL_PATH = DATA_DIR / "vesicle.mrc"

def load_vesicle_ground_truth(path):
    with mrcfile.open(path, permissive=True) as mrc:
        true_vol = mrc.data.astype(np.float32).transpose(2, 1, 0)  # (Z, Y, X)
    return true_vol

def build_vesicle_config():
    return ParticleTomographyConfig(
        input_paths=InputDataConfig(
            projection_file=DATA_DIR / "projections_vesicle.mat",
            rotations_file=DATA_DIR / "projections_vesicle_euler_angles.mat"
        ),
        model=ModelConfig(
            n_particles=5000,
            particle_init_mode="isonormal",
            kernel_size=3,
            dtype=torch.float32,
            device="cuda"
        ),
        training=TrainingConfig(
            steps=[
                TrainingStep(1000, 2.5e-3, 41, 1.0),
                GMMRejuvenateStep(rejuv_in_box=False),
                TrainingStep(1000, 2.5e-3, 41, 1.0),
                SaveImagesStep(
                    out_dir=OUT_DIR,
                    slice_thickness=10,
                    logging_prefix="final",
                    true_volume_path=TRUE_VOL_PATH,
                    true_volume_loader=load_vesicle_ground_truth
                )
            ]
        )
    )

def main():
    config = build_vesicle_config()
    images, rotations, shifts, angles = load_vesicle_data(config.input_paths, return_angles=True)
    training_plan = build_plan_from_config(config.training)
    model = reconstruct(images,
                        rotations,
                        shifts=shifts,
                        num_points=config.model.n_particles,
                        particle_init_mode=config.model.particle_init_mode,
                        kernel_size=config.model.kernel_size,
                        training_plan=training_plan,
                        device=config.model.device,
                        )

    # model.plot_volume()
    # model.plot_points()
    # model.plot_weights()
    # true_volume = load_vesicle_ground_truth(TRUE_VOL_PATH)
    # model.plot_fsc(true_volume)


if __name__ == "__main__":

    torch.manual_seed(0)
    t1 = time.time()
    main()
    t2 = time.time()
    print("runtime:",t2 - t1)