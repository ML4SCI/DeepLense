import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display

def visual_ptyphi(coordinates, event_id):
    """
    Visualizes the 3D points for the specified event_id from the coordinates dictionary.

    Parameters:
    - coordinates (dict): A dictionary where keys are event IDs and values are arrays of coordinates.
    - event_id (str): The event ID to visualize.
    """
    event_data = coordinates.get(event_id)
    if event_data is None:
        print(f"Event ID {event_id} not found.")
        return

    # Check if data has the correct number of dimensions for plotting
    if len(event_data.shape) < 2 or event_data.shape[1] != 3:
        print("Data must have exactly 3 dimensions for visualization (e.g., (pt, y, phi)).")
        return

    # Extract the coordinates
    pt = event_data[:, 0]  # Assuming 'pt' is the first column
    y = event_data[:, 1]   # Assuming 'y' is the second column
    phi = event_data[:, 2] # Assuming 'phi' is the third column

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pt, y, phi, c='r', marker='o')
    ax.set_xlabel('P_t (Transverse Momentum)')
    ax.set_ylabel('y (y-Cartesian coordinate)')
    ax.set_zlabel('Phi (Azimuthal Angle)')
    ax.set_title(f'3D Visualization for {event_id}')

    plt.grid(True)
    plt.show()

def interactive_plot_coordinates(coordinates):
    """
    Creates an interactive widget to select and visualize events from the coordinates dictionary.

    Parameters:
    - coordinates (dict): A dictionary where keys are event IDs and values are arrays of coordinates.
    """
    event_ids = list(coordinates.keys())
    event_selector = widgets.Dropdown(
        options=event_ids,
        value=event_ids[0] if event_ids else None,
        description='Event:',
    )

    def on_event_change(change):
        visual_ptyphi(coordinates, change.new)

    event_selector.observe(on_event_change, names='value')
    display(event_selector)
