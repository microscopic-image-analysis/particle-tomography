from setuptools import setup, find_packages

setup(
    name="particle-tomography",
    version="0.1.0",
    description="A package for 3D tomographic reconstruction using point clouds instead of voxel grids.",
    author="Tobias Pretschold",
    author_email="tobias.pretschold@uni-jena.de",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "plotly",
        "matplotlib",
        "pyvista",
        "differentiable-rasterizer @ git+https://github.com/microscopic-image-analysis/differentiable-rasterizer",
    ],
)
