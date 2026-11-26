"""
Validates dataset integrity before training.

Checks that .npy files exist for every expected class and index,
that LR and HR image counts match, and that array shapes are
consistent with the configured image dimensions and scale factor.

Run from the module root:
    python validate_dataset.py --config dataset_config.yaml

Exits with code 0 if all checks pass, 1 otherwise.
"""

import argparse
import os
import sys

import numpy as np
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check_directory(directory, label):
    """Return True if directory exists, print error otherwise."""
    if not os.path.isdir(directory):
        print(f"[FAIL] {label}: directory not found: {directory}")
        return False
    return True


def check_class_files(data_dir, class_name, num_samples, aux="_sim_"):
    """
    Verify that {class_name}/{class_name}{aux}{i}.npy exists for
    i in [0, num_samples) and that all arrays share the same shape.
    Returns (missing_count, shape_mismatches, expected_shape).
    """
    missing = []
    shapes = {}
    class_dir = os.path.join(data_dir, class_name)

    if not os.path.isdir(class_dir):
        print(f"  [FAIL] class directory missing: {class_dir}")
        return num_samples, 0, None

    for i in range(num_samples):
        filename = f"{class_name}{aux}{i}.npy"
        filepath = os.path.join(class_dir, filename)
        if not os.path.isfile(filepath):
            missing.append(i)
            continue
        try:
            arr = np.load(filepath)
            shapes[i] = arr.shape
        except Exception as e:
            print(f"  [WARN] could not load {filepath}: {e}")
            missing.append(i)

    if not shapes:
        return len(missing), 0, None

    # determine the most common shape as the expected one
    shape_counts = {}
    for s in shapes.values():
        shape_counts[s] = shape_counts.get(s, 0) + 1
    expected_shape = max(shape_counts, key=shape_counts.get)

    mismatches = sum(1 for s in shapes.values() if s != expected_shape)
    return len(missing), mismatches, expected_shape


def check_hr_lr_consistency(
    data_dir, lr_classes, hr_classes, num_samples
):
    """
    Verify that HR and LR evaluation sets have matching file counts
    per class pair (e.g. no_sub <-> no_sub_HR).
    """
    errors = 0
    for lr_cls, hr_cls in zip(lr_classes, hr_classes):
        lr_dir = os.path.join(data_dir, lr_cls)
        hr_dir = os.path.join(data_dir, hr_cls)

        lr_count = 0
        hr_count = 0
        if os.path.isdir(lr_dir):
            lr_count = len(
                [f for f in os.listdir(lr_dir) if f.endswith(".npy")]
            )
        if os.path.isdir(hr_dir):
            hr_count = len(
                [f for f in os.listdir(hr_dir) if f.endswith(".npy")]
            )

        if lr_count != hr_count:
            print(
                f"  [FAIL] count mismatch: {lr_cls}={lr_count} vs "
                f"{hr_cls}={hr_count}"
            )
            errors += 1
        else:
            print(f"  [OK]   {lr_cls}={lr_count}, {hr_cls}={hr_count}")
    return errors


def check_shape_vs_config(shape, expected_dim, label):
    """Check that array spatial dims match the configured image_shape."""
    if shape is None:
        return 0  # already reported as missing
    spatial = shape[-1]  # last dim is the image width for square images
    if spatial != expected_dim:
        print(
            f"  [FAIL] {label}: shape {shape}, expected "
            f"spatial dim {expected_dim}"
        )
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset before training"
    )
    parser.add_argument(
        "--config",
        default="dataset_config.yaml",
        help="path to dataset config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_samples = cfg["num_samples"]
    image_shape = cfg["image_shape"]
    magnification = cfg["magnification"]
    hr_shape = image_shape * magnification

    total_errors = 0

    for model_name, model_cfg in cfg["models"].items():
        data_dir = model_cfg["data_dir"]
        print(f"\n{'='*50}")
        print(f"Validating {model_name}: {data_dir}")
        print(f"{'='*50}")

        if not check_directory(data_dir, model_name):
            total_errors += 1
            continue

        # check training classes (LR simulations)
        print(f"\n  Training classes (simulations):")
        for cls in cfg["classes"]["train"]:
            missing, mismatches, shape = check_class_files(
                data_dir, cls, num_samples, aux="_sim_"
            )
            status = "OK" if (missing == 0 and mismatches == 0) else "FAIL"
            print(
                f"  [{status}] {cls}: {num_samples - missing}/{num_samples} "
                f"files, {mismatches} shape mismatches, shape={shape}"
            )
            total_errors += int(missing > 0) + int(mismatches > 0)

            # validate spatial dimensions against config
            total_errors += check_shape_vs_config(
                shape, image_shape, f"{cls} LR"
            )

        # check deflection angle files
        print(f"\n  Deflection angles:")
        for cls in cfg["classes"]["train"]:
            missing, mismatches, shape = check_class_files(
                data_dir, cls, num_samples, aux="_alpha_"
            )
            status = "OK" if (missing == 0 and mismatches == 0) else "FAIL"
            print(
                f"  [{status}] {cls}: {num_samples - missing}/{num_samples} "
                f"files, {mismatches} shape mismatches, shape={shape}"
            )
            total_errors += int(missing > 0) + int(mismatches > 0)

        # check eval HR classes and LR/HR count consistency
        eval_dir = model_cfg.get("eval_dir", data_dir)
        print(f"\n  Evaluation HR/LR consistency ({eval_dir}):")
        total_errors += check_hr_lr_consistency(
            eval_dir,
            cfg["classes"]["eval_lr"],
            cfg["classes"]["eval_hr"],
            num_samples,
        )

        # check HR image shapes match expected upscaled dimensions
        print(f"\n  HR shape check (expected spatial dim: {hr_shape}):")
        for cls in cfg["classes"]["eval_hr"]:
            _, _, shape = check_class_files(
                eval_dir, cls, num_samples, aux="_sim_"
            )
            total_errors += check_shape_vs_config(
                shape, hr_shape, f"{cls} HR"
            )

    print(f"\n{'='*50}")
    if total_errors == 0:
        print("All checks passed.")
    else:
        print(f"{total_errors} issue(s) found.")
    print(f"{'='*50}")
    sys.exit(0 if total_errors == 0 else 1)


if __name__ == "__main__":
    main()
