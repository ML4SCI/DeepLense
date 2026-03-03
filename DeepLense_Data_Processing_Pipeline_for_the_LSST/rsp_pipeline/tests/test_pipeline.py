"""
test_pipeline.py
----------------
Unit tests for the lsst_deeplense pipeline components.

All tests run **offline** — no Rubin Science Platform token or network
access is required.  Heavy dependencies (pyvo, torch, torchvision) are
mocked or skipped where necessary so the suite passes in a bare
Python + numpy environment.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers to inject lightweight mocks before importing the modules under test
# ---------------------------------------------------------------------------

def _mock_pyvo():
    """Inject a minimal pyvo stub into sys.modules."""
    pyvo = types.ModuleType("pyvo")
    pyvo.dal = types.ModuleType("pyvo.dal")
    pyvo.auth = types.ModuleType("pyvo.auth")

    class FakeCredStore:
        def set_password(self, *a, **kw):
            pass

    pyvo.auth.CredentialStore = FakeCredStore
    sys.modules.setdefault("pyvo", pyvo)
    sys.modules.setdefault("pyvo.dal", pyvo.dal)
    sys.modules.setdefault("pyvo.auth", pyvo.auth)
    return pyvo


def _mock_torch():
    """Inject a minimal torch/torchvision stub."""
    torch_mod = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)

        def numpy(self):
            return self._arr

        def __repr__(self):
            return f"FakeTensor({self._arr.shape})"

    torch_mod.from_numpy = lambda a: FakeTensor(a)

    utils_data = types.ModuleType("torch.utils.data")

    class FakeDataset:
        pass

    utils_data.Dataset = FakeDataset
    torch_mod.utils = types.SimpleNamespace(data=utils_data)
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.utils", torch_mod.utils)
    sys.modules.setdefault("torch.utils.data", utils_data)
    return torch_mod


# ---------------------------------------------------------------------------
# Normaliser tests
# ---------------------------------------------------------------------------

class TestNormaliser(unittest.TestCase):
    def _make_img(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random((64, 64)).astype(np.float32) * 100.0

    def test_minmax_range(self):
        from lsst_deeplense.preprocessing.normalise import Normaliser
        norm = Normaliser(strategy="minmax")
        out = norm(self._make_img())
        self.assertAlmostEqual(float(out.min()), 0.0, places=5)
        self.assertAlmostEqual(float(out.max()), 1.0, places=5)

    def test_zscore_mean_std(self):
        from lsst_deeplense.preprocessing.normalise import Normaliser
        norm = Normaliser(strategy="zscore")
        out = norm(self._make_img())
        self.assertAlmostEqual(float(out.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out.std()), 1.0, places=2)

    def test_asinh_range(self):
        from lsst_deeplense.preprocessing.normalise import Normaliser
        norm = Normaliser(strategy="asinh")
        out = norm(self._make_img())
        self.assertGreaterEqual(float(out.min()), 0.0 - 1e-5)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_invalid_strategy(self):
        from lsst_deeplense.preprocessing.normalise import Normaliser
        with self.assertRaises(ValueError):
            Normaliser(strategy="log")

    def test_sigma_clip(self):
        from lsst_deeplense.preprocessing.normalise import Normaliser
        img = self._make_img()
        # Insert extreme outliers
        img[0, 0] = 1e6
        norm = Normaliser(strategy="minmax", clip_sigma=3.0)
        out = norm(img)
        self.assertAlmostEqual(float(out.max()), 1.0, places=5)

    def test_multichannel_input(self):
        """3-channel input should work without error."""
        from lsst_deeplense.preprocessing.normalise import Normaliser
        norm = Normaliser(strategy="asinh")
        img3 = np.random.rand(3, 64, 64).astype(np.float32)
        # Normaliser operates per-channel when called in a loop (dataset does this)
        for c in range(3):
            out = norm(img3[c])
            self.assertEqual(out.shape, (64, 64))


# ---------------------------------------------------------------------------
# CutoutExtractor tests
# ---------------------------------------------------------------------------

class TestCutoutExtractor(unittest.TestCase):
    def _make_big_img(self, h: int = 256, w: int = 256) -> np.ndarray:
        return np.random.rand(h, w).astype(np.float32)

    def test_extract_shape(self):
        from lsst_deeplense.preprocessing.cutout import CutoutExtractor
        ext = CutoutExtractor(output_size=64)
        patch = ext.extract(self._make_big_img())
        self.assertEqual(patch.shape, (1, 64, 64))

    def test_extract_with_half_size(self):
        from lsst_deeplense.preprocessing.cutout import CutoutExtractor
        ext = CutoutExtractor(output_size=64, half_size=32)
        patch = ext.extract(self._make_big_img(), centre_row=128, centre_col=128)
        self.assertEqual(patch.shape, (1, 64, 64))

    def test_stack_bands(self):
        from lsst_deeplense.preprocessing.cutout import CutoutExtractor
        ext = CutoutExtractor(output_size=64)
        bands = {
            "g": np.random.rand(50, 50).astype(np.float32),
            "r": np.random.rand(50, 50).astype(np.float32),
            "i": np.random.rand(50, 50).astype(np.float32),
        }
        stacked = ext.stack_bands(bands, band_order=("g", "r", "i"))
        self.assertEqual(stacked.shape, (3, 64, 64))
        self.assertEqual(stacked.dtype, np.float32)

    def test_stack_missing_band_raises(self):
        from lsst_deeplense.preprocessing.cutout import CutoutExtractor
        ext = CutoutExtractor(output_size=64)
        bands = {"g": np.random.rand(64, 64).astype(np.float32)}
        with self.assertRaises(KeyError):
            ext.stack_bands(bands, band_order=("g", "r", "i"))

    def test_already_correct_size_passthrough(self):
        from lsst_deeplense.preprocessing.cutout import CutoutExtractor
        ext = CutoutExtractor(output_size=64)
        img = np.ones((64, 64), dtype=np.float32)
        resized = ext._resize(img)
        np.testing.assert_array_equal(resized, img)


# ---------------------------------------------------------------------------
# TAPClient ADQL builder test (no network)
# ---------------------------------------------------------------------------

class TestTAPClientADQL(unittest.TestCase):
    def test_adql_contains_coords(self):
        from lsst_deeplense.data_access.tap_client import RubinTAPClient
        adql = RubinTAPClient._build_cone_adql(
            ra=150.1,
            dec=2.2,
            radius_deg=0.5,
            band="i",
            mag_limit=24.0,
            snr_min=10.0,
            table="dp02_dc2_catalogs.Object",
            max_rows=1000,
        )
        self.assertIn("150.1", adql)
        self.assertIn("2.2", adql)
        self.assertIn("0.5", adql)
        self.assertIn("i_cModelMag", adql)
        self.assertIn("detect_isPrimary", adql)
        self.assertIn("TOP 1000", adql)

    def test_adql_snr_filter(self):
        from lsst_deeplense.data_access.tap_client import RubinTAPClient
        adql = RubinTAPClient._build_cone_adql(
            ra=0.0, dec=0.0, radius_deg=1.0,
            band="r", mag_limit=25.0, snr_min=5.0,
            table="dp02_dc2_catalogs.Object", max_rows=100,
        )
        self.assertIn("r_psfFlux", adql)
        self.assertIn("5.0", adql)


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------

class TestConfigLoader(unittest.TestCase):
    def test_load_valid_config(self):
        """The bundled pipeline_config.yaml should load without error."""
        from lsst_deeplense.utils.config import load_config
        config_path = (
            Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"
        )
        if not config_path.exists():
            self.skipTest("pipeline_config.yaml not found")
        cfg = load_config(config_path)
        for key in ("data", "preprocessing", "pipeline"):
            self.assertIn(key, cfg)

    def test_missing_file_raises(self):
        from lsst_deeplense.utils.config import load_config
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# I/O helpers tests
# ---------------------------------------------------------------------------

class TestIOHelpers(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()

    def test_save_and_load_roundtrip(self):
        from lsst_deeplense.utils.io import load_candidates_csv, save_candidates_csv
        df = pd.DataFrame({"ra": [1.0, 2.0], "dec": [3.0, 4.0], "label": [0, 1]})
        path = Path(self._tmpdir) / "candidates.csv"
        save_candidates_csv(df, path)
        loaded = load_candidates_csv(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_overwrite_false_raises(self):
        from lsst_deeplense.utils.io import save_candidates_csv
        df = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
        path = Path(self._tmpdir) / "dup.csv"
        save_candidates_csv(df, path)
        with self.assertRaises(FileExistsError):
            save_candidates_csv(df, path, overwrite=False)

    def test_missing_columns_raises(self):
        from lsst_deeplense.utils.io import load_candidates_csv, save_candidates_csv
        df = pd.DataFrame({"ra": [1.0]})  # missing 'dec'
        path = Path(self._tmpdir) / "bad.csv"
        df.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            load_candidates_csv(path)


# ---------------------------------------------------------------------------
# RubinLensDataset offline tests
# ---------------------------------------------------------------------------

class TestRubinLensDatasetOffline(unittest.TestCase):
    """Test RubinLensDataset.from_npy_dir without any network access."""

    def setUp(self):
        import tempfile
        self._tmpdir = Path(tempfile.mkdtemp())
        # Create fake .npy files
        self._positions = [(150.1, 2.2), (150.2, 2.3), (150.3, 2.4)]
        for ra, dec in self._positions:
            arr = np.random.rand(3, 64, 64).astype(np.float32)
            np.save(str(self._tmpdir / f"{ra:.6f}_{dec:.6f}.npy"), arr)

    def test_from_npy_dir_length(self):
        _mock_torch()
        from lsst_deeplense.dataset.rubin_dataset import RubinLensDataset
        ds = RubinLensDataset.from_npy_dir(self._tmpdir)
        self.assertEqual(len(ds), 3)

    def test_from_npy_dir_with_labels(self):
        _mock_torch()
        label_df = pd.DataFrame(
            [{"ra": ra, "dec": dec, "label": i}
             for i, (ra, dec) in enumerate(self._positions)]
        )
        label_csv = self._tmpdir / "labels.csv"
        label_df.to_csv(label_csv, index=False)

        from lsst_deeplense.dataset.rubin_dataset import RubinLensDataset
        ds = RubinLensDataset.from_npy_dir(self._tmpdir, label_csv=label_csv)
        self.assertTrue(ds._has_labels)

    def test_empty_dir_raises(self):
        import tempfile
        empty = Path(tempfile.mkdtemp())
        _mock_torch()
        from lsst_deeplense.dataset.rubin_dataset import RubinLensDataset
        with self.assertRaises(FileNotFoundError):
            RubinLensDataset.from_npy_dir(empty)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
