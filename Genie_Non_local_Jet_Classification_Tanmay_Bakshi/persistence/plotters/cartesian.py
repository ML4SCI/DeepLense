import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ipywidgets as widgets
from IPython.display import display

def standardize_and_normalize(coordinates):
    """
    Standardizes and normalizes the 3D coordinates.

    Parameters:
    - coordinates (np.ndarray): Array of coordinates with shape (n, 3).

    Returns:
    - standardized_and_normalized_coordinates (np.ndarray): Standardized and normalized 3D coordinates.
    """
    if coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have exactly 3 columns for (x, y, z) or similar.")

    # Standardization
    scaler = StandardScaler()
    standardized_coordinates = scaler.fit_transform(coordinates)

    # Normalization
    normalizer = MinMaxScaler()
    normalized_coordinates = normalizer.fit_transform(standardized_coordinates)

    return normalized_coordinates

def visualize_coordinates(coordinates, event_id):
    """
    Visualizes the 3D points for the specified event_id after standardizing and normalizing the coordinates.

    Parameters:
    - coordinates (dict): A dictionary where keys are event IDs and values are arrays of 3D coordinates.
    - event_id (str): The event ID to visualize.
    """
    event_data = coordinates.get(event_id)
    if event_data is None:
        print(f"Event ID {event_id} not found.")
        return

    # Check if data has the correct number of dimensions for plotting
    if len(event_data.shape) < 2 or event_data.shape[1] != 3:
        print("Data must have exactly 3 dimensions for visualization.")
        return

    # Apply standardization and normalization
    standardized_normalized_coordinates = standardize_and_normalize(event_data)

    # Extract the standardized and normalized coordinates
    x = standardized_normalized_coordinates[:, 0]  # Assuming x is the first column
    y = standardized_normalized_coordinates[:, 1]  # Assuming y is the second column
    z = standardized_normalized_coordinates[:, 2]  # Assuming z is the third column

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Visualization for {event_id}')

    plt.show()

def interactive_plot_coordinates(coordinates):
    """
    Creates an interactive widget to select and visualize events from the coordinates dictionary.

    Parameters:
    - coordinates (dict): A dictionary where keys are event IDs and values are arrays of 3D coordinates.
    """
    event_ids = list(coordinates.keys())
    event_selector = widgets.Dropdown(
        options=event_ids,
        value=event_ids[0] if event_ids else None,
        description='Event:',
    )

    def on_event_change(change):
        visualize_coordinates(coordinates, change.new)

    event_selector.observe(on_event_change, names='value')
    display(event_selector)

# Example usage:
# coordinates = {'event1': np.array([[1, 2, 3], [4, 5, 6]]), 'event2': np.array([[7, 8, 9], [10, 11, 12]])}
# interactive_plot_coordinates(coordinates)
