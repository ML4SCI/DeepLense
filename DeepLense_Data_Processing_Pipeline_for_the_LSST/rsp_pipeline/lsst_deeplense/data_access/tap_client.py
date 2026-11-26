"""
tap_client.py
-------------
Thin wrapper around pyvo TAP to query the Rubin Science Platform
DP1 / DP0.2 object catalogs for strong gravitational lens candidates.

Authentication
--------------
Set the environment variable RSP_TOKEN to your Rubin Science Platform
access token.  Tokens are issued at https://data.lsst.cloud/

Usage example
-------------
>>> from lsst_deeplense.data_access import RubinTAPClient
>>> client = RubinTAPClient()
>>> candidates = client.query_lens_candidates(ra=150.1, dec=2.2, radius_deg=0.5)
>>> print(candidates.head())
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Public Rubin Science Platform TAP endpoint (DP0.2 / DP1)
_DEFAULT_TAP_URL = "https://data.lsst.cloud/api/tap"

# Minimum SNR and magnitude cuts that approximate a realistic lens-candidate
# pre-selection (tunable via query kwargs).
_DEFAULT_MAGLIM = 24.0
_DEFAULT_SNR_MIN = 10.0


class RubinTAPClient:
    """
    Query the Rubin Science Platform Table Access Protocol (TAP) service
    to retrieve photometric catalogs suitable for lens-candidate selection.

    Parameters
    ----------
    tap_url : str, optional
        Override the default RSP TAP endpoint.
    token : str, optional
        RSP access token.  Falls back to the ``RSP_TOKEN`` environment
        variable if not supplied.
    timeout : int
        HTTP request timeout in seconds (default 120).
    """

    def __init__(
        self,
        tap_url: str = _DEFAULT_TAP_URL,
        token: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        try:
            import pyvo  # noqa: F401 – optional dependency
        except ImportError as exc:
            raise ImportError(
                "pyvo is required for TAP access.  "
                "Install it with:  pip install pyvo"
            ) from exc

        import pyvo

        self._token = token or os.environ.get("RSP_TOKEN", "")
        if not self._token:
            logger.warning(
                "No RSP_TOKEN found.  Anonymous access may be rate-limited "
                "or unavailable for proprietary data releases."
            )

        credential = (
            pyvo.auth.CredentialStore()
            if not self._token
            else self._build_credential(pyvo)
        )

        self._service = pyvo.dal.TAPService(tap_url, credential)
        self._timeout = timeout
        logger.info("RubinTAPClient initialised against %s", tap_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_lens_candidates(
        self,
        ra: float,
        dec: float,
        radius_deg: float = 1.0,
        band: str = "i",
        mag_limit: float = _DEFAULT_MAGLIM,
        snr_min: float = _DEFAULT_SNR_MIN,
        max_rows: int = 10_000,
        table: str = "dp02_dc2_catalogs.Object",
    ) -> pd.DataFrame:
        """
        Cone-search the RSP object catalog and return a DataFrame of
        photometric candidates for lens-finding.

        Parameters
        ----------
        ra, dec : float
            Centre of the search cone in degrees (ICRS).
        radius_deg : float
            Search radius in degrees.
        band : str
            Photometric band to apply the magnitude cut (default ``'i'``).
        mag_limit : float
            Faint magnitude limit (AB mag).
        snr_min : float
            Minimum signal-to-noise ratio in the chosen band.
        max_rows : int
            Row cap for the TAP query.
        table : str
            Fully-qualified RSP catalog table name.

        Returns
        -------
        pd.DataFrame
            Columns include ``objectId``, ``ra``, ``dec``,
            ``{band}_psfFlux``, ``{band}_psfFluxErr``,
            ``{band}_cModelMag``, ``detect_isPrimary``.
        """
        adql = self._build_cone_adql(
            ra=ra,
            dec=dec,
            radius_deg=radius_deg,
            band=band,
            mag_limit=mag_limit,
            snr_min=snr_min,
            table=table,
            max_rows=max_rows,
        )
        logger.debug("Executing ADQL:\n%s", adql)
        result = self._service.search(adql, maxrec=max_rows)
        df = result.to_table().to_pandas()
        logger.info(
            "query_lens_candidates returned %d rows "
            "(ra=%.4f, dec=%.4f, r=%.3f deg)",
            len(df),
            ra,
            dec,
            radius_deg,
        )
        return df

    def query_by_ids(
        self,
        object_ids: list[int],
        table: str = "dp02_dc2_catalogs.Object",
    ) -> pd.DataFrame:
        """
        Fetch catalog rows for a specific list of ``objectId`` values.

        Useful when you already have a candidate list (e.g. from a
        lens-finder model) and want to pull the full photometry.

        Parameters
        ----------
        object_ids : list[int]
            List of RSP ``objectId`` integers.
        table : str
            Fully-qualified catalog table name.

        Returns
        -------
        pd.DataFrame
        """
        if not object_ids:
            return pd.DataFrame()

        id_list = ", ".join(str(oid) for oid in object_ids)
        adql = (
            f"SELECT * FROM {table} "
            f"WHERE objectId IN ({id_list}) "
            f"AND detect_isPrimary = 1"
        )
        result = self._service.search(adql, maxrec=len(object_ids) + 10)
        return result.to_table().to_pandas()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cone_adql(
        ra: float,
        dec: float,
        radius_deg: float,
        band: str,
        mag_limit: float,
        snr_min: float,
        table: str,
        max_rows: int,
    ) -> str:
        flux_col = f"{band}_psfFlux"
        flux_err_col = f"{band}_psfFluxErr"
        mag_col = f"{band}_cModelMag"
        return (
            f"SELECT TOP {max_rows} "
            f"objectId, coord_ra AS ra, coord_dec AS dec, "
            f"{flux_col}, {flux_err_col}, {mag_col}, "
            f"detect_isPrimary "
            f"FROM {table} "
            f"WHERE CONTAINS("
            f"  POINT('ICRS', coord_ra, coord_dec), "
            f"  CIRCLE('ICRS', {ra}, {dec}, {radius_deg})"
            f") = 1 "
            f"AND detect_isPrimary = 1 "
            f"AND {mag_col} < {mag_limit} "
            f"AND {flux_col} / NULLIF({flux_err_col}, 0) > {snr_min}"
        )

    def _build_credential(self, pyvo_module) -> "pyvo.auth.CredentialStore":  # type: ignore[name-defined]
        store = pyvo_module.auth.CredentialStore()
        store.set_password(
            "x-oauth-basic",
            self._token,
            "https://data.lsst.cloud",
        )
        return store
