# -*- coding: utf-8 -*-
"""
Visualization for landlab RiverFlowDynamics simulations

@author: angel
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
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
    
# Import required libraries
try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Basic plotting will be used.")


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


def create_combined_plot(grid, timestep, field_name="surface_water__depth", 
                        figsize=(18, 7), colormap='viridis'):
    """
    Create combined 3D and cross-section plot for RiverFlowDynamics visualization.
    
    Parameters:
    -----------
    grid : RasterModelGrid
        Landlab grid object
    timestep : int
        Current timestep number
    field_name : str
        Field to plot (default: "surface_water__depth")
    figsize : tuple
        Figure size (width, height) - increased width for better spacing
    colormap : str
        Colormap name
    """
    
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available. Using basic 3D rendering.")
    
    # Get data
    field_data = grid.at_node[field_name]
    nrows, ncols = grid.shape
    
    # Get plot cross-section data
    plot_nodes, plot_distances = get_nodes(grid, nrows, ncols)
    
    # Create figure with larger width and strategic layout
    fig = plt.figure(figsize=figsize)
    
    # Set main title for entire figure
    fig.suptitle(f'Circular Dam Break - RiverFlowDynamics - Time Step: {timestep}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Use subplots with better proportions and more left margin for 3D labels
    gs = fig.add_gridspec(1, 8, width_ratios=[0.8, 3.5, 0.3, 0.2, 0.3, 2.5, 0.2, 0.2])
    
    # 3D plot - position in columns 1 (more left margin for y-label)
    ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    
    # 2D plot - position in columns 5 (better proportioned)
    ax2 = fig.add_subplot(gs[0, 5])
    
    # Prepare 3D data
    Z_orig = field_data.reshape(nrows, ncols)
    X_orig = grid.x_of_node.reshape(nrows, ncols)
    Y_orig = grid.y_of_node.reshape(nrows, ncols)
    
    if SCIPY_AVAILABLE:
        # Apply smoothing and upsampling for clean visualization
        upsample_factor = 5  # Increased for better smoothness
        smooth_sigma = 2.0   # Increased smoothing
        
        # Step 1: Smooth original data more aggressively
        Z_smoothed = gaussian_filter(Z_orig, sigma=smooth_sigma)
        
        # Step 2: Create high-resolution grid
        x_min, x_max = X_orig.min(), X_orig.max()
        y_min, y_max = Y_orig.min(), Y_orig.max()
        
        new_nrows = nrows * upsample_factor
        new_ncols = ncols * upsample_factor
        
        x_new = np.linspace(x_min, x_max, new_ncols)
        y_new = np.linspace(y_min, y_max, new_nrows)
        X_new, Y_new = np.meshgrid(x_new, y_new)
        
        # Step 3: Interpolate to high resolution with better method
        points = np.column_stack((X_orig.ravel(), Y_orig.ravel()))
        values = Z_smoothed.ravel()
        
        try:
            Z_cubic = griddata(points, values, (X_new, Y_new), method='cubic')
            Z_linear = griddata(points, values, (X_new, Y_new), method='linear')
            
            # Fill NaN values from cubic with linear
            nan_mask = np.isnan(Z_cubic)
            Z_new = Z_cubic.copy()
            Z_new[nan_mask] = Z_linear[nan_mask]
            
            # Apply additional smoothing to interpolated result
            Z_new = gaussian_filter(Z_new, sigma=1.0)
            
        except:
            # Fallback to linear interpolation with smoothing
            Z_new = griddata(points, values, (X_new, Y_new), method='linear')
            Z_new = gaussian_filter(Z_new, sigma=1.0)
        
        # Final cleanup
        Z_new = np.nan_to_num(Z_new, nan=Z_orig.mean())
        
        # Use high-resolution data for plotting
        X_plot, Y_plot, Z_plot = X_new, Y_new, Z_new
        
    else:
        # If scipy not available, apply basic smoothing to original data
        print("Using basic smoothing (scipy not available)")
        Z_plot = Z_orig
        X_plot, Y_plot = X_orig, Y_orig
    
    # Create 3D surface with proper smoothing settings
    surface = ax1.plot_surface(X_plot, Y_plot, Z_plot, 
                              cmap=colormap,
                              alpha=0.98,
                              linewidth=0,
                              antialiased=True,
                              shade=True,
                              rasterized=True,
                              vmin=Z_plot.min(),
                              vmax=Z_plot.max(),
                              rstride=1,  # Use every point for smoothness
                              cstride=1)  # Use every point for smoothness
    
    # Style 3D plot with aggressive label padding to push y-label away from edge
    ax1.set_xlabel('x (m)', fontsize=11, labelpad=20)
    ax1.set_ylabel('y (m)', fontsize=11, labelpad=30)  # Very large padding
    # Don't set z-label, we'll add it as text instead
    ax1.view_init(elev=20, azim=45)
    
    # Add z-label as text annotation for better control
    ax1.text2D(0.975, 0.525, 'h (m)', transform=ax1.transAxes, fontsize=11, 
               rotation=90, verticalalignment='center')
    
    # Clean 3D background
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_alpha(0.1)
    ax1.yaxis.pane.set_alpha(0.1)
    ax1.zaxis.pane.set_alpha(0.1)
    
    # Clean 3D background
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_alpha(0.1)
    ax1.yaxis.pane.set_alpha(0.1)
    ax1.zaxis.pane.set_alpha(0.1)
    
    # Clean 3D background
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_alpha(0.1)
    ax1.yaxis.pane.set_alpha(0.1)
    ax1.zaxis.pane.set_alpha(0.1)
    
    # Process cross-section data
    mask = (plot_distances >= 10) & (plot_distances <= 60)
    distances_subset = plot_distances[mask]
    nodes_subset = plot_nodes[mask]
    depths_subset = field_data[nodes_subset]
    distances_relabeled = distances_subset - 10
    
    # Create cross-section plot (no title)
    ax2.plot(distances_relabeled, depths_subset, 'b-', linewidth=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x (m)', fontsize=11)
    ax2.set_ylabel('Water Depth (m)', fontsize=11)
    
    # Position colorbar in the gap between plots (column 3)
    cbar_ax = fig.add_subplot(gs[0, 3])
    colorbar = plt.colorbar(surface, cax=cbar_ax)
    colorbar.set_label('h (m)', fontsize=11)
    
    # Left margin visibility
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.15)
    
    plt.show()
    
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
    from PIL import Image
    import glob
    import os
    
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
            print(f"Loaded frame {i+1}/{len(frame_files)}: {os.path.basename(frame_file)}")
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
        
        # Get file size for info
        file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
        
        print(f"✅ GIF created successfully!")
        print(f"📁 Filename: {output_filename}")
        print(f"📏 File size: {file_size:.2f} MB")
        print(f"🎬 Frames: {len(images)}")
        print(f"⏱️ Duration: {total_duration_seconds} seconds")
        print(f"🔄 Frame rate: {len(images)/total_duration_seconds:.1f} fps")
        
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

