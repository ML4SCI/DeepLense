import numpy as np
import pandas as pd
import multipers as mp
import multipers.grids as mpg
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import awkward as ak
import dask.array as da
import dask.dataframe as dd
from multipers.ml.convolutions import KDE as KernelDensity
from coordinates_extract import extract_coordinates

class PersistenceModule:
    def __init__(self, coordinates):
        """
        Initialize the PersistenceModule with a coordinates dictionary.

        Parameters:
        coordinates : dict
            A dictionary where keys are event IDs and values are arrays of coordinates
            with each entry containing (x, y, z, and possibly other features).
        """
        self.coordinates = coordinates
        self.dataset = self._convert_to_dataset(coordinates)

    def _convert_to_dataset(self, coordinates):
        """
        Convert the coordinates dictionary to a dataset (Awkward Array or Dask Array).

        Parameters:
        coordinates : dict
            A dictionary where keys are event IDs and values are arrays of coordinates.

        Returns:
        dataset : Awkward Array or Dask Array
            A dataset where each entry contains the coordinates.
        """
        # Convert the dictionary values to a list of arrays
        data_arrays = list(coordinates.values())

        # Create an Awkward Array from the list of arrays
        dataset = da.Array(data_arrays)

        return dataset



    def compute_simplex_trees_multi(self):
        """
        Compute the SimplexTreeMulti for each entry in the dataset using 2-parameter filtration (Rips + co-density)

        Returns:
        simplex_trees : list of mp.SimplexTreeMulti
            A list containing the SimplexTreeMulti for each entry in the dataset.
        """
        simplex_trees = []
        # Iterate over each entry in the dataset
        for entry in self._iterate_dataset(self.dataset):
            # Extract the (x, y, z, t) coordinates
            event = np.array(entry)
            n_coords = event.shape[1]
            # Create a RipsComplex using the first 3 coordinates (x, y, z)
            rips_complex = gd.RipsComplex(points=event, sparse=0.8)
            simplex_tree = rips_complex.create_simplex_tree()

            # Create a SimplexTreeMulti with 2 parameters
            st_multi = mp.SimplexTreeMulti(simplex_tree, num_parameters=2)

            # Calculate the co-log-density for the second parameter
            codensity = - KernelDensity(bandwidth=0.2).fit(event).score_samples(event[:, :3])

            # Fill the second parameter with the co-log-density
            st_multi.fill_lowerstar(codensity, parameter=1)
            # print("Before edge collapses :", st_multi.num_simplices)
            st_multi.collapse_edges(-2)  # This should take less than 20s. -1 is for maximal "simple" collapses, -2 is for maximal "harder" collapses.
            # print("After edge collapses :", st_multi.num_simplices)
            st_multi.expansion(2)  # Be careful, we have to expand the dimension to 2 before computing degree 1 homology.
            # print("After expansion :", st_multi.num_simplices)
            # Append the SimplexTreeMulti to the list
            simplex_trees.append(st_multi)

        return simplex_trees



    def compute_simplex_trees(self):
        """
        Compute a 1-parameter SimplexTree Object for each entry in the dataset using a suitable Complex (Vietoris-Rips for now)

        Returns:
        simplex_trees : list of mp.SimplexTree
        A list containing the SimplexTreeObject for each entry in the dataset.

        """
        simplex_trees = []

        for entry in self._iterate_dataset(self.dataset):
            event = np.array(entry)

            simplex_tree = gd.RipsComplex(points=event, sparse=0.8).create_simplex_tree(max_dimension = 2)
            simplex_trees.append(simplex_tree)

        return simplex_trees


    def _iterate_dataset(self, dataset):
        """
        A helper function to iterate over the dataset.

        Parameters:
        dataset : Awkward Array or Dask Array
            The dataset to iterate over.

        Yields:
        entry : np.ndarray
            Each entry in the dataset as a numpy array.
        """
        if isinstance(dataset, ak.Array):
            for entry in dataset:
                yield entry
        elif isinstance(dataset, da.Array):
            for entry in dataset.to_delayed():
                yield entry.compute()
        else:
            raise TypeError("Dataset must be an Awkward Array or Dask Array.")



if __name__ == '__main__':
    PersMod = PersistenceModule()