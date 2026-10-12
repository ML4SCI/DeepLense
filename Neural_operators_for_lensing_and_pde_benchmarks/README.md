# Neural Operators for Lensing and PDE Benchmarks

This contains the architectures:- BALANI-NO, FNO, ALNO, LNO, PDNO, UNO, U-FNO, HC-UNO, CNN,
DeepONet, POD-DeepONet evaluated across Darcy flow, Navier-Stokes
vorticity, and gravitational lensing tasks.

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for a full 7-seed x 11-architecture run in reasonable
time; each script also runs on CPU (slower) with no code changes.

## Files

| File | What it reproduces |
|---|---|
| `train_darcy.py` | Downloads Darcy, trains/evals all 11 architectures, 7 seeds, @16/@32 |
| `train_ns.py` | Downloads Navier-Stokes, trains/evals all 11 architectures, 7 seeds, @32/@64 |
| `train_lensing.py` | Trains/evals all 11 architectures on lensing, 7 seeds, WDM/CDM/Axion. Set `LENSING_VARIANT = "sub"` or `"full"` at the top to switch tasks |
| `train_lensing_deeponet_variants.py` | Separate ablation: 7 DeepONet branch/trunk variants (Vanilla, Stacked, Conv, Fourier, Attention, POD, BelNet) + an FNO baseline row, 3 seeds, rel_L2 + SSIM |
| `train_darcy_pooling_ablation.py` | Workshop paper: CNN, FNO, U-FNO, UNO, HC-UNO + no-pool ablation of UNO/HC-UNO, 3 seeds, Darcy @16/@32. Separate architecture set from the 11-arch comparison |

Each script is fully self-contained (download → load → train → eval).

## Reproducing each table

```bash
python train_darcy.py      # writes darcy_all11_7seed_summary.csv
python train_ns.py         # writes ns_all11_7seed_summary.csv

LENSING_DATA_ROOT=/path/to/modela python train_lensing.py
# edit LENSING_VARIANT ("sub" / "full") at the top and re-run for the other lensing table

python train_lensing_deeponet_variants.py   # writes all7_summary_table.csv
```

`LENSING_DATA_ROOT` must point at a directory containing `val/` and `test/`
subfolders, each holding per-class subfolders (`wdm/`, `cdm/`, `axion/`) of
`.npz` files with `kappa`/`psi` (full) or `kappa_sub`/`psi_sub` (subhalo)
arrays. `train_lensing_deeponet_variants.py` instead expects a
`manifest.csv`-indexed dataset (see its data-loading section) and is not
compatible with the same `LENSING_DATA_ROOT` layout.

## Seeds

The 11-architecture tables (Darcy, NS, both lensing variants) use the same 7
seeds: `13, 42, 97, 7, 64, 128, 2024`. The DeepONet-variant ablation uses 3
seeds: `0, 1, 2`.

## What's in `configs/hyperparameters.md`

A table of everything that differs *across* datasets (batch size, optimizer,
`FMAX` formula, the two different `evaluate()` conventions used by Darcy/NS
vs. lensing, and how the DeepONet-variant study's data pipeline differs from
the main lensing pipeline).
