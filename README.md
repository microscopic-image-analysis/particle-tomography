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

This package depends on **differentiable-rasterizer**. Please install it first and follow the instructions [here](https://github.com/microscopic-image-analysis/differentiable-rasterizer).

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
/home/tobias/anaconda3/lib/python3.12/site-packages/mrcfile/mrcinterpreter.py:216: RuntimeWarning: Unrecognised machine stamp: 0x00 0x00 0x00 0x00
  warnings.warn(str(err), RuntimeWarning)
Saving output in /home/tobias/Desktop/test2/particle-tomography/out/vesicle
FSC at Nyquist: 0.859
4.665709018707275
```

The script will reconstruct a biological vesicle from 41 projection images. When run successfully, a folder out with projection and slice images of the ground truth volume and the reconstruction is created. It also creates a plot of the FSC correlation curve as a function of frequency. The reconstructed volume (dense voxel representation of the volume) and the model state (sparse representation of the volume) are also saved.


