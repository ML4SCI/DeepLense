# Hyperparameters and cross-dataset discrepancies

All 11 architectures are param-matched per dataset to BALANI-NO's parameter
count (~112K). Exact per-architecture configs live in the `MODEL_ZOO` list at
the top of each `train_*.py` script — this file is only for the things that
differ *across* datasets, since those differences are easy to lose track of
and matter for how the results should be read.

| | Darcy | Navier-Stokes | Lensing |
|---|---|---|---|
| Epochs | 60 | 60 | 60 |
| Batch size | 16 | 16 | 32 |
| Optimizer | AdamW, wd=1e-5 | AdamW, wd=1e-5 | Adam, wd=1e-5 (no "W") |
| Grad clipping | none | none | clip_grad_norm_ = 1.0 |
| Seeds | 13, 42, 97, 7, 64, 128, 2024 | 13, 42, 97, 7, 64, 128, 2024 | 13, 42, 97, 7, 64, 128, 2024 |
| Train / test resolution | 16 / 32 (zero-shot = resolution) | 32 / 64 (zero-shot = resolution) | 64 / 64 (zero-shot = physics: CDM, Axion) |
| Coord grid | external, concatenated before model | external, concatenated before model | internal, registered buffer per model |
| `evaluate()` formula | mean of per-sample relative-L2 | mean of per-sample relative-L2 | sum(numerator norms) / sum(denominator norms) over the whole loader |
| UNO/HC-UNO `FMAX` | `train_res // 2` = 8 | `train_res // 2` = 16 | `RES // 4` = 16 |
| DeepONet `branch_res` | 16 | 16 (fixed from an initial 32 -- see commit history / PR notes) | native (RES=64, full image passed to branch) |

**Why `evaluate()` differs for lensing:** the lensing per-architecture cells
this was built from all used the sum-of-norms formula consistently, so it
was kept as-is for lensing rather than silently switched to match Darcy/NS.
Both are valid relative-error statistics but they are NOT numerically
interchangeable — do not compare a lensing rel_L2 number directly against a
Darcy/NS rel_L2 number as if they were the same metric.

**Why lensing's grid handling differs:** Darcy/NS build the coordinate grid
once and concatenate it into the input tensor externally (so the model just
sees an already-augmented tensor). The lensing architectures instead
register the grid as a buffer inside `__init__` at construction time (`res=`
is passed to the constructor) and concatenate it inside `forward()`. Both
give the model the same information; this is a code-organization difference,
not a modeling difference.

## Lensing: two separate tasks, same architectures/training loop

`train_lensing.py` has a `LENSING_VARIANT` constant at the top:
- `"full"` — kappa / psi (full-field convergence and lensing potential)
- `"sub"` — kappa_sub / psi_sub (subhalo-only convergence and potential)

Both variants share identical file-list construction, train/test/zero-shot
splits, architectures, training loop, and evaluation — only the two `.npz`
field keys read differ. Run the script once per variant to reproduce both
tables; outputs are suffixed with the variant name so they don't collide
(`lensing_all11_full_7seed_summary.csv`, `lensing_all11_sub_7seed_summary.csv`).

