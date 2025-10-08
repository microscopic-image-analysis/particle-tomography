## Installation


This package depends on **PyTorch**. Please install the version of PyTorch that matches your system and CUDA version **before** installing `particle-tomography`.

Check the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for details.

Examples:

```bash
# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify your installation:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

This package also depends on **differentiable-rasterizer**. Please install it first and follow the instructions [here](https://github.com/microscopic-image-analysis/differentiable-rasterizer).

Then clone this repository and install:

```bash
git clone https://github.com/microscopic-image-analysis/particle-tomography
cd particle-tomography
pip install -e .
```

---

## Quick Start
After installation, you can verify that the package and the differentiable rasterizer are working correctly by running the vesicle.py script:
```bash
python scripts/vesicle.py
```
The expected out is similar to:
```bash
CUDA extension detected!
File contains 41 projection images of size 64x64 pixels
Initializing model with kernel size 3
Epoch 1000/1000, Loss per image: -335788.6875
  Noise Std: 0.0762
  R-factor: 0.0811
  Current bandwidth: 0.25612708926200867
Epoch 1000/1000, Loss per image: -372517.8750
  Noise Std: 0.0400
  R-factor: 0.0740
  Current bandwidth: 0.1942751556634903
/home/yourname/anaconda3/lib/python3.12/site-packages/mrcfile/mrcinterpreter.py:216: RuntimeWarning: Unrecognised machine stamp: 0x00 0x00 0x00 0x00
  warnings.warn(str(err), RuntimeWarning)
Saving output in /path/to/particle-tomography/out/vesicle
FSC at Nyquist: 0.859
runtime: 4.665709018707275
```

The script will reconstruct a biological vesicle from 41 projection images. When run successfully, a folder out with projection and slice images of the ground truth volume and the reconstruction is created. It also creates a plot of the FSC correlation curve as a function of frequency. The reconstructed volume (dense voxel representation of the volume) and the model state (sparse representation of the volume) are also saved.

Internally--after building an optional config structure for the vesicle dataset--the script executes:
```python
    from particle_tomography import reconstruct
    
    model = reconstruct(images,
                        rotations,
                        shifts=shifts,
                        num_points=config.model.n_particles,
                        particle_init_mode=config.model.particle_init_mode,
                        kernel_size=config.model.kernel_size,
                        training_plan=training_plan,
                        device=config.model.device,
                        )
```
The reconstruct function is the main API of the package. It takes as input the data, model parameters and training parameters. Training parameters are either passed as a training_plan object for maximal control (by defining training steps), or by directly specifying training parameters such as total_iterations, batch_size, lr, etc. For more details, please see the documentation. The reconstruct function returns a model object. It can be used to obtain the reconstructed volume in the desired format:
```python
    voxel_volume = model.get_volume()  # shape (64, 64, 64)
    points, weights, bandwidth = model.get_volume_sparse()
```
or to visualize the results:
```python
    model.plot_volume()
    model.plot_points()
    model.plot_weights()
    true_volume = load_vesicle_ground_truth(TRUE_VOL_PATH)
    model.plot_fsc(true_volume)
```



