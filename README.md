## Installation


This package depends on **PyTorch**. Please install the version of PyTorch that matches your system and CUDA version **before** installing `particle-tomography`. Check the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for details.

This package also depends on **differentiable-rasterizer**. Please install it first and follow the instructions [here](https://github.com/microscopic-image-analysis/differentiable-rasterizer).

Then clone this repository and install:

```bash
git clone https://github.com/microscopic-image-analysis/particle-tomography
cd particle-tomography
pip install -e .
```

---

## Quick Start
After installation, you can verify that the package and the differentiable rasterizer are working correctly by running the protein.py script:
```bash
python scripts/protein.py
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

The script will reconstruct a simulated protein from projection images. When run successfully, a folder out with projection and slice images of the ground truth volume and the reconstruction is created. It also creates a plot of the FSC correlation curve as a function of frequency. The reconstructed volume (dense voxel representation of the volume) and the model state (sparse representation of the volume) are also saved.

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



