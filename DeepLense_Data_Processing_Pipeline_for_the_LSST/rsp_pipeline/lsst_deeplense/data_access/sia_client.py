"""
sia_client.py
-------------
Simple Image Access v2 (SIA2) client to retrieve calibrated image cutouts
from the Rubin Science Platform around known or candidate lens positions.

The returned images are ``numpy`` arrays ready to be fed into the
DeepLense preprocessing pipeline or directly into a PyTorch Dataset.

Usage example
-------------
>>> from lsst_deeplense.data_access import RubinSIAClient
>>> sia = RubinSIAClient()
>>> images = sia.fetch_cutouts(
...     ra=150.1, dec=2.2, size_arcsec=10.0, bands=["g", "r", "i"]
... )
>>> # images is a dict  band -> np.ndarray (H, W)
"""

from __future__ import annotations

import io
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SIA_URL = "https://data.lsst.cloud/api/image/siav2"
_DEFAULT_CUTOUT_ARCSEC = 10.0  # ~50 px at 0.2 arcsec/px (LSST native scale)


class RubinSIAClient:
    """
    Retrieve image cutouts from the Rubin Science Platform via SIA v2.

    Parameters
    ----------
    sia_url : str, optional
        Override the default RSP SIAv2 endpoint.
    token : str, optional
        RSP access token.  Falls back to ``RSP_TOKEN`` env var.
    timeout : int
        HTTP timeout in seconds (default 120).
    """

    def __init__(
        self,
        sia_url: str = _DEFAULT_SIA_URL,
        token: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        try:
            import pyvo  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pyvo is required for SIA access.  "
                "Install it with:  pip install pyvo"
            ) from exc

        import pyvo

        self._token = token or os.environ.get("RSP_TOKEN", "")
        self._sia_url = sia_url
        self._timeout = timeout

        credential = self._build_credential(pyvo) if self._token else None
        self._service = pyvo.dal.SIAService(sia_url, credential)
        logger.info("RubinSIAClient initialised against %s", sia_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_cutouts(
        self,
        ra: float,
        dec: float,
        size_arcsec: float = _DEFAULT_CUTOUT_ARCSEC,
        bands: List[str] = None,
        collection: str = "LSST/DP0.2",
        image_type: str = "deepCoadd",
    ) -> Dict[str, np.ndarray]:
        """
        Download calibrated coadd cutouts in one or more bands.

        Parameters
        ----------
        ra, dec : float
            Sky position in degrees (ICRS).
        size_arcsec : float
            Cutout half-width in arcseconds.
        bands : list[str], optional
            Photometric bands to retrieve (default: ``['g', 'r', 'i']``).
        collection : str
            RSP data collection (default: DP0.2 DC2 simulated survey).
        image_type : str
            Dataset type (``'deepCoadd'`` or ``'calexp'``).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of band name → float32 numpy array (H × W).
            Missing or failed bands are omitted with a warning.
        """
        if bands is None:
            bands = ["g", "r", "i"]

        size_deg = size_arcsec / 3600.0
        results: Dict[str, np.ndarray] = {}

        for band in bands:
            try:
                arr = self._fetch_single_band(
                    ra=ra,
                    dec=dec,
                    size_deg=size_deg,
                    band=band,
                    collection=collection,
                    image_type=image_type,
                )
                if arr is not None:
                    results[band] = arr
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to retrieve band '%s' at (%.4f, %.4f): %s",
                    band,
                    ra,
                    dec,
                    exc,
                )

        if not results:
            logger.error(
                "No cutouts retrieved for (ra=%.4f, dec=%.4f).", ra, dec
            )
        return results

    def fetch_cutouts_batch(
        self,
        positions: List[Dict],
        size_arcsec: float = _DEFAULT_CUTOUT_ARCSEC,
        bands: List[str] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Retrieve cutouts for a batch of sky positions.

        Parameters
        ----------
        positions : list[dict]
            Each dict must contain ``'ra'`` and ``'dec'`` keys (floats).
        size_arcsec : float
            Cutout half-width in arcseconds.
        bands : list[str], optional
            Bands to retrieve per position.

        Returns
        -------
        list[dict[str, np.ndarray]]
            One dict per input position, in the same order.
        """
        return [
            self.fetch_cutouts(
                ra=pos["ra"],
                dec=pos["dec"],
                size_arcsec=size_arcsec,
                bands=bands,
            )
            for pos in positions
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_single_band(
        self,
        ra: float,
        dec: float,
        size_deg: float,
        band: str,
        collection: str,
        image_type: str,
    ) -> Optional[np.ndarray]:
        """Return a float32 array for a single band or None on failure."""
        try:
            from astropy.io import fits  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "astropy is required for FITS parsing.  "
                "Install it with:  pip install astropy"
            ) from exc

        result_table = self._service.search(
            pos=(ra, dec),
            size=size_deg,
            band=band,
            collection=collection,
        )

        if len(result_table) == 0:
            logger.debug(
                "No SIA results for band=%s at (%.4f, %.4f)", band, ra, dec
            )
            return None

        # Take the first (best) result
        row = result_table[0]
        access_url = row.getdataurl()

        import urllib.request  # noqa: PLC0415

        req = urllib.request.Request(
            access_url,
            headers={"Authorization": f"Bearer {self._token}"}
            if self._token
            else {},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read()

        with fits.open(io.BytesIO(raw)) as hdul:
            # Science extension is typically index 1 for coadds
            sci_ext = 1 if len(hdul) > 1 else 0
            data = hdul[sci_ext].data

        if data is None:
            return None

        return data.astype(np.float32)

    def _build_credential(self, pyvo_module):
        store = pyvo_module.auth.CredentialStore()
        store.set_password(
            "x-oauth-basic",
            self._token,
            "https://data.lsst.cloud",
        )
        return store
