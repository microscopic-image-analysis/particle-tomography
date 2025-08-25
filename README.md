## Installation


This package depends on **PyTorch**, but PyTorch GPU builds are not available on PyPI by default.  
Please install the version of PyTorch that matches your system and CUDA version **before** installing `particle-tomography`.

Check the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for details.

Examples:

```bash
# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify your installation:
```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

Then clone this repository and run the installation command:


```bash
git clone https://github.com/yourusername/particle-tomography.git
cd particle-tomography
pip install .
```

This will also fetch the dependency
[differentiable-rasterizer](https://github.com/microscopic-image-analysis/differentiable-rasterizer).

---

## Quick Start

```python
import numpy as np
from particle_tomography import particle_tomography

# Example inputs
images = np.random.rand(41, 64, 64)  # 41 projections of size 64x64
rotations = np.random.rand(41, 3, 3) # random rotation matrices

# Run reconstruction (uses GPU if available, otherwise falls back to CPU)
reconstructed = particle_tomography(images, rotations, num_points=5000, total_iterations=2000)
```

---
