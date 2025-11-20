import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    import pyvista as pv
except ImportError:
    pv = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def plot_volume(vol, cmap="viridis", camera=None, path=None, side=(1, 1, 1)):
    """
    Visualize a 3D volume (Z, Y, X) using PyVista with optional camera control,
    normalizing the axes to [-side_i, side_i] instead of voxel indices.
    """
    if pv is None:
        raise RuntimeError("pyvista not installed. Use `pip install pyvista`")

    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()

    nz, ny, nx = vol.shape

    # --- Create a uniform grid centered at 0 with dimensions normalized ---
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))

    # Compute spacing so the axes run from -side_i to +side_i
    grid.spacing = (
        (2*side[0]) / nx,
        (2*side[1]) / ny,
        (2*side[2]) / nz
    )

    # Translate grid so center is at 0
    grid.origin = (-side[0], -side[1], -side[2])

    # Add volume data
    grid["values"] = vol.flatten(order="C")

    # create plotter
    if path is not None:
        plotter = pv.Plotter(off_screen=True, window_size=[2000, 2000])
    else:
        plotter = pv.Plotter()
    plotter.show_axes()
    plotter.show_grid()

    plotter.add_volume(grid, cmap=cmap, scalars="values", show_scalar_bar=False)

    # --- Camera control ---
    if camera is not None:
        plotter.camera_position = camera
    else:
        plotter.view_isometric()

    # --- Show or save ---
    if path is not None:
        plotter.show(screenshot=path)
        print(f"Saved volume image to {path}")
    else:
        plotter.show()

def plot_points(X, side=(1.25, 1.25, 1.25), title=None):
    """
    Plot a 3D point cloud with optional ground truth and title.

    Args:
        X (torch.Tensor or np.ndarray): Predicted volume, shape (N, 3).
        Y (torch.Tensor or np.ndarray, optional): Ground truth points, shape (M, 3).
        side (tuple): Half-widths for x, y, z axes.
        title (str, optional): Plot title.
    """
    if go is None:
        raise RuntimeError("plotly not installed. Use `pip install particle-tomography[plotting]`")
    fig = go.Figure(data=[
        go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            name="pred", mode='markers',
            marker=dict(size=3, line=dict(width=1, color='darkblue'))
        )
    ])


    # Calculate aspect ratios proportional to the side dimensions
    max_side = max(side)
    aspect_x = side[0] / max_side
    aspect_y = side[1] / max_side
    aspect_z = side[2] / max_side

    fig.update_layout(
        title=title,
        template="simple_white",
        width=700,
        height=700,
        scene=dict(
            aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
            xaxis=dict(range=[-side[0], side[0]]),
            yaxis=dict(range=[-side[1], side[1]]),
            zaxis=dict(range=[-side[2], side[2]])
        )
    )

    fig.show()
    return


def show_images(images, indices=None):
    """
    Display images from a batch or a single image.

    Args:
        images: Tensor/array of shape [B, H, W] or [H, W]
        indices: int, range, list, or None (default: first 2 for batch, 0 for single)
    """
    # Convert to numpy and normalize dimensions
    if hasattr(images, 'cpu'):  # torch tensor
        img_array = images.cpu().numpy()
    else:
        img_array = np.asarray(images)

    if img_array.ndim == 2:
        img_array = img_array[None, ...]  # [H, W] -> [1, H, W]
        single_image = True
    else:
        single_image = False

    B, H, W = img_array.shape
    img_array = img_array.transpose(0, 2, 1)  # Transpose to match matlab format

    # Handle indices
    if indices is None:
        indices = [0] if single_image else list(range(min(2, B)))
    elif isinstance(indices, int):
        indices = [indices]
    else:
        indices = list(indices)

    # Filter valid indices
    valid_indices = [i for i in indices if 0 <= i < B]
    if not valid_indices:
        print(f"No valid indices. Batch size: {B}")
        return

    # Display images
    for i in valid_indices:
        plt.figure(figsize=(6, 6))
        plt.imshow(img_array[i], extent=[0, W, H, 0], interpolation='nearest', cmap='viridis')
        plt.xlabel("Y")
        plt.ylabel("X")
        plt.title("Raster output" if single_image else f"Raster output for image {i}")
        plt.colorbar()
        plt.show()


def plot_fsc(
    frequencies, correlations,
    title="Fourier Shell Correlation",
    threshold_lines=(0.5, 0.143),
    figsize=(8, 6),
    show=False,
):
    """
    Plot FSC curve and return the matplotlib Figure.

    Parameters
    ----------
    frequencies : np.ndarray
        Normalized spatial frequencies (0–1 scale).
    correlations : np.ndarray
        FSC values corresponding to frequencies.
    title : str
        Title of the plot.
    threshold_lines : tuple[float]
        Horizontal threshold values to plot.
    figsize : tuple[int, int]
        Size of the matplotlib figure.
    show : bool
        If True, call plt.show() for interactive display.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot FSC curve
    ax.plot(frequencies * 100, correlations, 'b-', linewidth=2, label='FSC')

    # Add threshold lines
    colors = ['red', 'orange', 'green']
    labels = ['0.5 threshold', '0.143 threshold (gold standard)', 'Custom threshold']

    for i, threshold in enumerate(threshold_lines):
        color = colors[i] if i < len(colors) else 'gray'
        label = labels[i] if i < len(labels) else f'{threshold} threshold'
        ax.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=label)

        # Add resolution estimates at threshold crossings
        crossing_indices = np.where((correlations[:-1] >= threshold) & (correlations[1:] < threshold))[0]
        if len(crossing_indices) > 0:
            idx = crossing_indices[0]
            y1, y2 = correlations[idx], correlations[idx + 1]
            x1, x2 = frequencies[idx], frequencies[idx + 1]
            crossing_freq = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            ax.annotate(
                f'FSC={threshold:.3f}\n{crossing_freq * 100:.1f}% Nyquist',
                xy=(crossing_freq * 100, threshold),
                xytext=(crossing_freq * 100 + 10, threshold + 0.15),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # Axis formatting
    ax.set_xlabel('Spatial Frequency (% of Nyquist)', fontsize=12)
    ax.set_ylabel('Fourier Shell Correlation', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    print(f"FSC at Nyquist: {correlations[-1]:.3f}")

    if show:
        plt.show()

    return fig


def plot_weights(weights, title="Model Weights", xlabel="Index", ylabel="Weight Value"):
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(weights)), weights, marker='.', linestyle='none', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()