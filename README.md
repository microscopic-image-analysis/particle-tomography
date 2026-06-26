# particle-tomography

`particle-tomography` is a Python package for reconstructing sparse 3D particle-based representations from projection images. It provides a high-level `reconstruct(...)` API built on PyTorch and `differentiable-rasterizer`, with optional CUDA acceleration.

## Installation

`particle-tomography` requires PyTorch and `differentiable-rasterizer`. The default rasterizer backend uses a PyTorch implementation, so users do not need the CUDA Toolkit just to run the package.

For GPU systems, install the PyTorch build that matches your CUDA setup before installing this package. See the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for the correct command for your platform.

Install `differentiable-rasterizer` first, then clone and install this repository:

```bash
git clone https://github.com/microscopic-image-analysis/differentiable-rasterizer
cd differentiable-rasterizer
pip install -e .

cd ..
git clone https://github.com/microscopic-image-analysis/particle-tomography
cd particle-tomography
pip install -e .
```

To use the optional custom CUDA rasterizer backend, build it from the local `differentiable-rasterizer` clone before installing or running `particle-tomography`:

```bash
cd differentiable-rasterizer
python build_cuda.py
cd ../particle-tomography
pip install -e .
```

## Quick Start

Verify the installation by running the protein example from the `particle-tomography` repository root:

```bash
python scripts/protein.py
```

By default, the example uses `--device auto`, which selects CUDA when available and otherwise uses CPU. To show interactive plots after reconstruction, run:

```bash
python scripts/protein.py --plot
```

The script reconstructs a simulated protein from projection images, saves the sparse model state as `model_protein.pt`, and prints reconstruction metrics including the FSC at Nyquist.

The main package API is `reconstruct(...)`:

```python
from particle_tomography import reconstruct

model = reconstruct(
    images,
    rotations,
    shifts=None,
    num_points=10_000,
    device="auto",
)
```

The returned model can be converted to dense or sparse representations:

```python
voxel_volume = model.get_volume()
points, weights, bandwidth = model.get_volume_sparse()
```

See the documentation and example scripts for advanced training plans, plotting, and evaluation utilities.

## Tests

A small CPU-only smoke test suite checks the bundled protein loader, model initialization, a tiny reconstruction, save/load, FSC, R-factor, and operation without CUDA.

```bash
pip install -e .[test]
pytest
```

These tests are intended as a quick functionality check, not as a full reproduction of the paper experiments.
