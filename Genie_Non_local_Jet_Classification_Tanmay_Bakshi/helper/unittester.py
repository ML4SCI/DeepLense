from preprocess_dask import _extract_coords

import unittest
import pandas as pd
import numpy as np
import awkward as ak
from collections import OrderedDict


class TestExtractCoords(unittest.TestCase):

    def setUp(self):
        # Setup any initial data or state before each test
        self.df = pd.DataFrame({
            'PX_0': [1, 9],
            'PY_0': [3, 0],
            'PZ_0': [5, 13],
            'E_0': [7, 15],
            'PX_1': [0, 10],
            'PY_1': [4, 12],
            'PZ_1': [6, 14],
            'E_1': [8, 16],
            'is_signal_new': [1, 0],
            'ttv': [0, 1]
        })

    def test_extract_coords_basic(self):
        # Run the function with a small example dataframe
        result = _extract_coords(self.df)

        # Test that result has all expected keys
        expected_keys = ['x', 'y', 'z', 't', 'phi', 'rho', 'theta', 'eta', 'jet_nparticles', 'label', 'train_val_test']
        self.assertTrue(all(key in result for key in expected_keys))

        # Additional checks on the actual values
        self.assertEqual(result['jet_nparticles'].shape, (2,))
        self.assertEqual(result['label'].shape, (2, 2))

    def test_extract_coords_with_empty_energy(self):
        # Modify df so that _e is zero, testing the masking operation
        self.df['E_0'] = 0
        self.df['E_1'] = 0
        result = _extract_coords(self.df)

        # Expect no particles since all energy values are zero
        self.assertTrue(np.all(result['jet_nparticles'] == 0))

    def test_extract_coords_partial_data(self):
        # Test the function with a subset of data (start and stop parameters)
        result = _extract_coords(self.df, start=0, stop=1)

        # Ensure only one row is processed
        self.assertEqual(result['jet_nparticles'].shape[0], 1)
        self.assertEqual(result['label'].shape[0], 1)

    def test_extract_coords_column_prefix(self):
        # Test if column prefixes in _col_list are correctly handled
        cols = ['PX_0', 'PX_1', 'PY_0', 'PY_1']
        self.assertTrue(all(col in self.df.columns for col in cols))

    def test_extract_coords_invalid_input(self):
        # Pass a DataFrame missing expected columns and test for errors
        with self.assertRaises(KeyError):
            invalid_df = pd.DataFrame({
                'AX_0': [1, 2],
                'AY_0': [3, 4],
            })
            _extract_coords(invalid_df)


if __name__ == '__main__':
    unittest.main()