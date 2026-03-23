# -*- coding: utf-8 -*-
"""
Visualization for landlab RiverFlowDynamics simulations

@author: angel
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os

# Additional imports for Jupyter notebook display
try:
    from IPython.display import Image as IPImage, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("Warning: IPython not available. GIF display in notebook will not work.")
    

def get_nodes(grid, nrows, ncols):
    """
    Get nodes along the cross section.
    
    Parameters:
    -----------
    grid : RasterModelGrid
        Landlab grid object
    nrows, ncols : int
        Grid dimensions
        
    Returns:
    --------
    plot_nodes : array
        Node indices along water
    plot_distances : array
        Distances along water
    """
    plot_nodes = []
    plot_distances = []
    
    # Calculate plot endpoints
    x_start, y_start = grid.x_of_node[ncols - 1], grid.y_of_node[ncols - 1]  # Upper-left
    x_end, y_end = grid.x_of_node[nrows * ncols - ncols], grid.y_of_node[nrows * ncols - ncols]  # Lower-right
    
    # Total plot distance
    total_distance = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    
    # Sample points along plot
    n_samples = min(nrows, ncols)
    
    for i in range(n_samples):
        t = i / (n_samples - 1)  # Parameter from 0 to 1
        x_target = x_start + t * (x_end - x_start)
        y_target = y_start + t * (y_end - y_start)
        
        # Find closest node
        distances = np.sqrt((grid.x_of_node - x_target)**2 + (grid.y_of_node - y_target)**2)
        closest_node = np.argmin(distances)
        
        plot_nodes.append(closest_node)
        plot_distances.append(t * total_distance)
    
    return np.array(plot_nodes), np.array(plot_distances)


def create_combined_plot(
    grid,
    timestep,
    field_name="surface_water__depth",
    figsize=(15, 7),
    colormap="Blues",
    center_x=25.0,
    center_y=25.0,
    radius=10.5,
    use_smooth_3d=True,
    fine_factor=3,
    figure_title="Circular Dam Break - RiverFlowDynamics",
):

    try:
        from scipy.interpolate import RegularGridInterpolator
        SCIPY_LOCAL = True
    except ImportError:
        SCIPY_LOCAL = False

    field_data = grid.at_node[field_name]
    nrows, ncols = grid.shape

    Z_orig = field_data.reshape(nrows, ncols)
    X_orig = grid.x_of_node.reshape(nrows, ncols)
    Y_orig = grid.y_of_node.reshape(nrows, ncols)

    x1d = X_orig[0, :]
    y1d = Y_orig[:, 0]

    center_row = np.argmin(np.abs(y1d - center_y))
    x_profile = X_orig[center_row, :]
    h_profile = Z_orig[center_row, :]

    # -------------------------------------------------
    # FIXED limits
    # -------------------------------------------------
    depth_min = 0.0
    depth_max = 12.0

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.05, 1.0, 0.05])

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 2])
    cbar_ax = fig.add_subplot(gs[0, 3])

    fig.suptitle(
        f"{figure_title} - Time Step: {timestep}",
        fontsize=15,
        fontweight="bold"
    )

    # -------------------------
    # 3D surface
    # -------------------------
    if use_smooth_3d and SCIPY_LOCAL:
        interp = RegularGridInterpolator(
            (y1d, x1d),
            Z_orig,
            method="linear",
            bounds_error=False,
            fill_value=None
        )

        nx = ncols * fine_factor
        ny = nrows * fine_factor
        xf = np.linspace(x1d.min(), x1d.max(), nx)
        yf = np.linspace(y1d.min(), y1d.max(), ny)
        X_plot, Y_plot = np.meshgrid(xf, yf)

        pts = np.column_stack([Y_plot.ravel(), X_plot.ravel()])
        Z_plot = interp(pts).reshape(Y_plot.shape)
    else:
        X_plot, Y_plot, Z_plot = X_orig, Y_orig, Z_orig

    surface = ax1.plot_surface(
        X_plot, Y_plot, Z_plot,
        cmap=colormap,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=True,
        alpha=0.98,
        vmin=depth_min,
        vmax=depth_max
    )

    ax1.contour(
        X_plot, Y_plot, Z_plot,
        zdir="z",
        offset=depth_min,
        levels=10,
        cmap=colormap,
        linewidths=0.8
    )

    ax1.set_xlabel("x [m]", fontsize=11, labelpad=8)
    ax1.set_ylabel("y [m]", fontsize=11, labelpad=8)
    ax1.set_zlabel("h [m]", fontsize=11, labelpad=8)
    ax1.set_xlim(x1d.min(), x1d.max())
    ax1.set_ylim(y1d.min(), y1d.max())
    ax1.set_zlim(depth_min, depth_max)
    ax1.view_init(elev=28, azim=-135)

    try:
        ax1.set_box_aspect((1, 1, 0.30))
    except Exception:
        pass

    ax1.grid(True, alpha=0.2)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    colorbar = plt.colorbar(surface, cax=cbar_ax)
    colorbar.set_label("Water Depth [m]", fontsize=11)
    colorbar.set_ticks(np.arange(0, 13, 2))

    # -------------------------
    # 2D centerline profile
    # -------------------------
    ax2.plot(
        x_profile,
        h_profile,
        color="deepskyblue",
        linewidth=2.8,
        label=f"Centerline depth (y = {y1d[center_row]:.1f} m)",
        zorder=3
    )
    ax2.fill_between(
        x_profile,
        0.0,
        h_profile,
        color="deepskyblue",
        alpha=0.35,
        zorder=2
    )

    ax2.axhline(1.0, color="gray", lw=1.2, ls="--", alpha=0.8, label="Outer depth")
    ax2.axvline(center_x - radius, color="black", lw=1.2, ls=":", alpha=0.7)
    ax2.axvline(center_x + radius, color="black", lw=1.2, ls=":", alpha=0.7)

    ax2.set_xlim(x1d.min(), x1d.max())
    ax2.set_ylim(0.0, 12.0)   # fixed y-axis
    ax2.set_title("Centerline Profile", fontsize=13, fontweight="bold")
    ax2.set_xlabel("x [m]", fontsize=11)
    ax2.set_ylabel("Water Depth [m]", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax2.legend(loc="upper right", framealpha=1.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    profile_text = f"center = ({center_x:.1f}, {center_y:.1f}) m, radius = {radius:.1f} m"
    ax2.text(
        0.02, 0.96, profile_text,
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8")
    )

    return fig, (ax1, ax2, cbar_ax)


# Set professional plot style
def set_plot_style():
    """Set matplotlib style for professional plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

# Initialize with professional style
set_plot_style()


def create_gif_animation(input_folder="simulation_frames", 
                        output_filename="dam_break_animation.gif",
                        total_duration_seconds=3,
                        loop=0,
                        optimize=True):
    """
    Create a GIF animation from saved PNG frames.
    
    Parameters:
    -----------
    input_folder : str
        Folder containing the PNG frames
    output_filename : str
        Name of output GIF file
    total_duration_seconds : float
        Total duration of the animation in seconds
    loop : int
        Number of loops (0 = infinite loop)
    optimize : bool
        Whether to optimize the GIF file size
    """

    
    # Get all PNG files in the correct order
    frame_pattern = os.path.join(input_folder, "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))
    
    if not frame_files:
        print(f"No frame files found in {input_folder}")
        return
    
    print(f"Found {len(frame_files)} frames")
    
    # Calculate duration per frame in milliseconds
    duration_per_frame = int((total_duration_seconds * 1000) / len(frame_files))
    
    print(f"Total duration: {total_duration_seconds} seconds")
    print(f"Duration per frame: {duration_per_frame} ms")
    
    # Load all images
    images = []
    for i, frame_file in enumerate(frame_files):
        try:
            img = Image.open(frame_file)
            # Convert to RGB if necessary (removes alpha channel)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading {frame_file}: {e}")
    
    if not images:
        print("No images were successfully loaded")
        return
    
    # Create and save the GIF
    print(f"Creating GIF animation: {output_filename}")
    
    try:
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=duration_per_frame,
            loop=loop,
            optimize=optimize,
            quality=85  # Adjust quality vs file size
        )
        
        print("GIF created successfully!")
        
    except Exception as e:
        print(f"Error creating GIF: {e}")

def show_gif_in_notebook(gif_filename):
    """Display the created GIF in the Jupyter notebook."""
    if not IPYTHON_AVAILABLE:
        print("IPython not available. Cannot display GIF in notebook.")
        print(f"GIF saved as: {gif_filename}")
        return
    
    try:
        display(IPImage(filename=gif_filename))
        print(f"Displaying: {gif_filename}")
    except Exception as e:
        print(f"Error displaying GIF: {e}")
        
def cleanup_frames(folder="simulation_frames", confirm=True):
    """
    Delete all frame files to free up space.
    
    Parameters:
    -----------
    folder : str
        Folder containing frames to delete
    confirm : bool
        Whether to ask for confirmation
    """
    import glob
    import os
    
    frame_files = glob.glob(os.path.join(folder, "frame_*.png"))
    
    if not frame_files:
        print("No frame files found to delete")
        return
    
    if confirm:
        response = input(f"Delete {len(frame_files)} frame files? (y/N): ")
        if response.lower() != 'y':
            print("Cleanup cancelled")
            return
    
    for frame_file in frame_files:
        try:
            os.remove(frame_file)
        except Exception as e:
            print(f"Error deleting {frame_file}: {e}")
    
    print(f"✅ Deleted {len(frame_files)} frame files")
    
    # Optionally remove the directory if empty
    try:
        os.rmdir(folder)
        print(f"✅ Removed empty directory: {folder}")
    except:
        pass  # Directory not empty or other issue

