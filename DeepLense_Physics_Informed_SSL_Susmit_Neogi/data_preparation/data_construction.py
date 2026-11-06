import argparse
import csv
import os
from pathlib import Path

import numpy as np
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import util

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Priors on the simulation parameters
# ---------------------------------------------------------------------------
# Einstein radius is the *label* — we sample it broadly and store it
# verbatim. The other params are nuisance variables; we randomize them
# so the dataset is not degenerate.
EINSTEIN_RADIUS_RANGE = (0.5, 2.5)          # arcseconds
ELLIPTICITY_RANGE = (0.0, 0.3)              # |e1|, |e2| of the lens
LENS_CENTER_RANGE = (-0.2, 0.2)             # arcsec, offset of lens from image center

SOURCE_R_RANGE = (0.05, 0.4)                # Sersic half-light radius, arcsec
SOURCE_E_RANGE = (0.0, 0.4)                 # source ellipticity
SOURCE_CENTER_RANGE = (-0.5, 0.5)           # arcsec, source position (drives arc vs ring)
SERSIC_INDEX_RANGE = (1.0, 4.0)             # Sersic n
SOURCE_AMPLITUDE_RANGE = (50.0, 500.0)      # counts / s

PIXEL_SIZE = 0.08                           # arcsec / pixel
IMAGE_SIZE = 64                             # pixels per side
PSF_FWHM = 0.1                              # arcsec — Gaussian PSF

# Background & noise — tweak to taste
BACKGROUND_NOISE = "SKY_NOISE"              # or "NONE" for noiseless
SKY_BRIGHTNESS = 0.0                        # e / s
EXPOSURE_TIME = 1000.0                      # s
READ_NOISE = 5.0                            # e / s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_kwargs(einstein_radius, rng):
    """Sample nuisance parameters and return lens/source kwargs_lists."""
    lens_model_list = ["SIE"]
    lens_model = LensModel(lens_model_list)

    # SIE: theta_E, e1, e2, center_x, center_y
    e1 = rng.uniform(-ELLIPTICITY_RANGE[1], ELLIPTICITY_RANGE[1])
    e2 = rng.uniform(-ELLIPTICITY_RANGE[1], ELLIPTICITY_RANGE[1])
    cx = rng.uniform(*LENS_CENTER_RANGE)
    cy = rng.uniform(*LENS_CENTER_RANGE)
    lens_kwargs = [{
        "theta_E": float(einstein_radius),
        "e1": float(e1),
        "e2": float(e2),
        "center_x": float(cx),
        "center_y": float(cy),
    }]

    # Sersic source: amp, R_sersic, n_sersic, e1, e2, center_x, center_y
    light_model_list = ["SERSIC_ELLIPSE"]
    light_model = LightModel(light_model_list)
    source_kwargs = [{
        "amp": float(rng.uniform(*SOURCE_AMPLITUDE_RANGE)),
        "R_sersic": float(rng.uniform(*SOURCE_R_RANGE)),
        "n_sersic": float(rng.uniform(*SERSIC_INDEX_RANGE)),
        "e1": float(rng.uniform(-SOURCE_E_RANGE[1], SOURCE_E_RANGE[1])),
        "e2": float(rng.uniform(-SOURCE_E_RANGE[1], SOURCE_E_RANGE[1])),
        "center_x": float(rng.uniform(*SOURCE_CENTER_RANGE)),
        "center_y": float(rng.uniform(*SOURCE_CENTER_RANGE)),
    }]

    return lens_model, light_model, lens_kwargs, source_kwargs, \
        lens_model_list, light_model_list


def make_data_and_psf():
    """Construct the (fixed) ImageData and PSF — built once, reused per sim."""
    numpix = IMAGE_SIZE
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(
        numPix=numpix, deltapix=PIXEL_SIZE, center_ra=0.0, center_dec=0.0,
        subgrid_res=1, inverse=False
    )
    kwargs_data = {
        "image_data": np.zeros((numpix, numpix)),
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": Mpix2coord,
        "background_rms": 0.5,
    }
    data = ImageData(**kwargs_data)

    psf = PSF(psf_type="GAUSSIAN", fwhm=PSF_FWHM, pixel_size=PIXEL_SIZE)
    return data, psf


def simulate_one(data, psf, lens_model, light_model, lens_kwargs, source_kwargs,
                 lens_model_list, light_model_list):
    """Run one simulation. Returns the (noisy) image as a 2D np.ndarray."""
    model = ImageModel(
        data_class=data,
        psf_class=psf,
        lens_model_class=lens_model,
        source_model_class=light_model,
        kwargs_numerics={"supersampling_factor": 1},
    )
    # NOTE: ImageModel.image takes the kwargs *lists* directly,
    # not a dict-wrapped kwargs_model/kwargs_params pair.
    image = model.image(kwargs_lens=lens_kwargs, kwargs_source=source_kwargs)
    rng_noise = np.random.default_rng()  # for the noise realization
    if BACKGROUND_NOISE != "NONE":
        # Gaussian background + Poisson source noise approximation
        bkg_var = (SKY_BRIGHTNESS + READ_NOISE ** 2 / EXPOSURE_TIME) / EXPOSURE_TIME
        image = rng_noise.normal(
            loc=image,
            scale=np.sqrt(np.maximum(image / EXPOSURE_TIME, 0) + bkg_var),
        )
    return np.asarray(image, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def progress_iter(iterable, total, enabled=True):
    """Wrap an iterable in a progress bar, with a simple fallback."""
    if not enabled:
        return iterable

    if tqdm is not None:
        return tqdm(iterable, total=total, desc="Generating simulations", unit="img")

    class SimpleProgress:
        def __init__(self, wrapped, count):
            self.wrapped = wrapped
            self.count = count

        def __iter__(self):
            step = max(1, self.count // 100)
            for index in self.wrapped:
                yield index
                done = index + 1
                if done == self.count or done % step == 0:
                    pct = 100 * done / self.count
                    print(f"\rGenerating simulations: {done}/{self.count} ({pct:5.1f}%)", end="")
            print()

    return SimpleProgress(iterable, total)


def generate(n_samples, out_dir, seed=None, show_progress=True):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    label_path = out_dir / "labels.csv"
    fieldnames = [
        "filename",
        "einstein_radius",  # <-- the label
        "lens_e1", "lens_e2", "lens_cx", "lens_cy",
        "src_amp", "src_R", "src_n",
        "src_e1", "src_e2", "src_cx", "src_cy",
    ]

    data, psf = make_data_and_psf()  # built once; kwargs vary per sim

    with open(label_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for i in progress_iter(range(n_samples), n_samples, show_progress):
            theta_E = float(rng.uniform(*EINSTEIN_RADIUS_RANGE))
            (lens_model, light_model, lens_kwargs, source_kwargs,
             lens_model_list, light_model_list) = build_kwargs(theta_E, rng)

            try:
                img = simulate_one(data, psf, lens_model, light_model,
                                   lens_kwargs, source_kwargs,
                                   lens_model_list, light_model_list)
            except Exception as e:
                # If a particular draw produces a numerical failure (rare),
                # skip and continue — we still have plenty of samples.
                print(f"[{i}] simulation failed: {type(e).__name__}: {e}")
                continue

            fname = f"img_{i:06d}.npy"
            np.save(image_dir / fname, img)

            writer.writerow({
                "filename": fname,
                "einstein_radius": theta_E,
                "lens_e1": lens_kwargs[0]["e1"],
                "lens_e2": lens_kwargs[0]["e2"],
                "lens_cx": lens_kwargs[0]["center_x"],
                "lens_cy": lens_kwargs[0]["center_y"],
                "src_amp": source_kwargs[0]["amp"],
                "src_R": source_kwargs[0]["R_sersic"],
                "src_n": source_kwargs[0]["n_sersic"],
                "src_e1": source_kwargs[0]["e1"],
                "src_e2": source_kwargs[0]["e2"],
                "src_cx": source_kwargs[0]["center_x"],
                "src_cy": source_kwargs[0]["center_y"],
            })

    print(f"Done. Wrote {n_samples} images to {image_dir} and labels to {label_path}.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1000, help="number of lensing images to simulate")
    p.add_argument("--out", type=str, default="data", help="output directory")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (for reproducibility)")
    p.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.n, args.out, args.seed, not args.no_progress)
