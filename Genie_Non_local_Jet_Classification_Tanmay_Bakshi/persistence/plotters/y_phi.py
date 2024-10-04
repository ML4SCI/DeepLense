import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

def visual_y_phi(coordinates, event_id):
    """
    Visualizes the 2D points (y vs. phi) for the specified event_id from the coordinates dictionary.

    Parameters:
    - coordinates (dict): A dictionary where keys are event IDs and values are arrays of coordinates.
    - event_id (str): The event ID to visualize.
    """
    event_data = coordinates.get(event_id)
    if event_data is None:
        print(f"Event ID {event_id} not found.")
        return

    # Check if data has the correct number of dimensions for plotting
    if len(event_data.shape) < 2 or event_data.shape[1] < 2:
        print("Data must have at least 2 dimensions for visualization (e.g., (y, phi)).")
        return

    # Extract 'y' and 'phi' assuming 'y' is the first column and 'phi' is the second column
    y = event_data[:, 0]   # Assuming 'y' is the first column
    phi = event_data[:, 1] # Assuming 'phi' is the second column

    # Create a 2D scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(y, phi, c='r', marker='o')
    ax.set_xlabel('y (Cartesian coordinate)')
    ax.set_ylabel('Phi (Azimuthal Angle)')
    ax.set_title(f'2D Visualization for {event_id}')

    plt.grid(True)
    plt.show()

def interactive_plot_y_phi(coordinates):
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
        visual_y_phi(coordinates, change.new)

    event_selector.observe(on_event_change, names='value')
    display(event_selector)

# Example usage:
# coordinates = {'event1': np.array([[1, 2], [3, 4]]), 'event2': np.array([[5, 6], [7, 8]])}
# interactive_plot_y_phi(coordinates)
