import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .utils import fsc
from .plot import plot_fsc

def save_output(points, weights, bandwidth, reconstruction, true_volume, logging_prefix, outdir, slice_thickness):
    print("Saving output in", outdir)
    reconstruction = reconstruction.copy()
    os.makedirs(outdir, exist_ok=True)
    # Save reconstruction as npz
    np.savez(os.path.join(outdir, f"{logging_prefix}_volume_reconstruction.npz"),
             reconstruction=reconstruction)

    if points is not None:
        np.savez(os.path.join(outdir, f"{logging_prefix}-particle_reconstruction.npz"),
                 points=points, weights=weights, bandwidth=bandwidth)

    # Make and save projections
    projections = make_projections(reconstruction, slice_thickness, true_volume)
    save_image(os.path.join(outdir, logging_prefix), projections)

    # FSC plot if true_volume exists
    if true_volume is not None:
        frequencies, correlations = fsc(reconstruction, true_volume, 20)  # Assumes fsc function exists
        fig = plot_fsc(frequencies, correlations)
        fig.savefig(outdir / "fsc_plot.png", bbox_inches='tight')


def make_projections(reconstruction, slice_thickness=10, true_volume=None):
    proj_data = {
        "projection": {"reconstruction": project(reconstruction)},
        "slice": {"reconstruction": project(reconstruction, slice_thickness)}
    }
    if true_volume is not None:
        proj_data["projection"]["ground_truth"] = project(true_volume)
        proj_data["slice"]["ground_truth"] = project(true_volume, slice_thickness)
    return proj_data


def project(img3d, slice_thickness=None):
    projections = {"xy": 0, "xz": 1, "yz": 2}  # axis indices for numpy

    def do_project(axis):
        if slice_thickness is None:
            return np.sum(img3d, axis=axis)
        else:
            center = img3d.shape[axis] // 2
            lo = max(0, center - slice_thickness // 2)
            hi = min(img3d.shape[axis], center + slice_thickness // 2)
            slices = [slice(None)] * img3d.ndim
            slices[axis] = slice(lo, hi)
            return np.sum(img3d[tuple(slices)], axis=axis)

    result = {}
    for name, axis in projections.items():
        proj = do_project(axis)
        # Scale to 0-1
        proj = (proj - proj.min()) / (proj.max() - proj.min())
        result[name] = np.clip(proj, 0, 1)
    return result


def save_image(name, value):
    if isinstance(value, dict):
        for k, v in value.items():
            save_image(f"{name}_{k}", v)
    else:
        # normalize
        value = (value - value.min()) / (value.max() - value.min())
        # Convert to 8-bit and save as PNG
        img_array = (value * 255).astype(np.uint8)
        Image.fromarray(img_array.transpose(1,0), mode='L').save(f"{name}.png")

