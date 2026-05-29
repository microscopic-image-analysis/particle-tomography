## Installation

`particle-tomography` requires PyTorch and `differentiable-rasterizer`. The default rasterizer install uses a PyTorch implementation, so users do not need the CUDA Toolkit just to run the package.

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

---

## Quick Start
After installation, you can verify that the package and the differentiable rasterizer are working correctly by running the protein example from the repository root:
```bash
python scripts/protein.py
```

The module form is equivalent:

```bash
python -m scripts.protein
```

By default, the example uses `--device auto`, which selects CUDA when available and otherwise uses CPU. It does not open interactive plots unless requested:

```bash
python scripts/protein.py --plot
```
The expected out is similar to:
```bash
CUDA extension detected!
Initializing model with kernel size 3
Epoch 1250/1250, Loss per image: -11839626.0000
  Noise Std: 0.0075
  R-factor: 0.0143
  Current bandwidth: 0.7306609749794006
Epoch 1250/1250, Loss per image: -16081655.0000
  Noise Std: 0.0006
  R-factor: 0.0107
  Current bandwidth: 0.7076988220214844
[ParticleTomographyModel] Saved model state to model_protein.pt
FSC at Nyquist: 0.853
Execution time: 9.75 seconds
```

The script reconstructs a simulated protein from projection images, saves the sparse model state as `model_protein.pt`, and prints the FSC at Nyquist. Use `--plot` to open interactive point-cloud, volume, and FSC plots after reconstruction.

Internally (after building an optional config structure for the protein dataset), the script executes:
```python
from particle_tomography import reconstruct
    
model = reconstruct(images,
                        rotations,
                        shifts=None,
                        num_points=config.model.n_particles,
                        particle_init_mode=config.model.particle_init_mode,
                        kernel_size=config.model.kernel_size,
                        training_plan=training_plan,
                        device=config.model.device,
                        )
```
The reconstruct function is the main API of the package. It takes as input the data, model parameters and training parameters. Training parameters are either passed as a training_plan object for maximal control (by defining training steps), or by directly specifying training parameters such as total_iterations, batch_size, lr, etc. For more details, please see the documentation. The reconstruct function returns a model object. It can be used to obtain the reconstructed volume in the desired format:
```python
voxel_volume = model.get_volume()  # shape (132, 132, 132)
points, weights, bandwidth = model.get_volume_sparse()
```
or to visualize the results:
```python
model.plot_volume()
model.plot_points()
model.plot_weights()
true_volume = load_protein_ground_truth(TRUE_VOL_PATH)
model.plot_fsc(true_volume)
```

---

## Tests

A small CPU-only smoke test suite checks the bundled protein loader, model initialization, a tiny reconstruction, save/load, FSC, R-factor, and operation without CUDA. Install the test extra and run:

```bash
pip install -e .[test]
pytest
```

These tests are intended as a quick functionality check, not as a full reproduction of the paper experiments.

