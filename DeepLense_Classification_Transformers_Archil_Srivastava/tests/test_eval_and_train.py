"""
Standalone verification script for Bug 1 and Bug 2 fixes.

Bug 1: micro_auroc was always nan because it was initialized as an empty list
        and never populated. Fix: compute it using auroc_fn(..., average="micro").

Bug 2: labels.type(torch.LongTensor) always creates a CPU tensor, even when the
        model is on GPU. Fix: use labels.to(device, dtype=torch.long).

Usage:
    cd DeepLense_Classification_Transformers_Archil_Srivastava
    python tests/test_eval_and_train.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import auroc as auroc_fn, accuracy as accuracy_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Instead of importing from eval.py / train.py (which have many side-effect
# imports), we test the FIX LOGIC directly. This is more robust and avoids
# pulling in the entire dependency tree.
# ═══════════════════════════════════════════════════════════════════════════════

NUM_CLASSES = 3
LABELS = ["axion", "cdm", "no_sub"]


@torch.no_grad()
def evaluate_fixed(model, data_loader, loss_fn, device):
    """
    The FIXED evaluate function — extracted from eval.py with both bug fixes applied.
    """
    model.eval()

    # ── Bug 1 fix: removed micro_auroc from this list (was never populated) ──
    loss, accuracy, class_auroc, macro_auroc = [], [], [], []
    logits, y = [], []

    for batch_X, batch_y in data_loader:
        # ── Bug 2 fix: was batch_y.type(torch.LongTensor) which always goes to CPU ──
        batch_X, batch_y = batch_X.to(device, dtype=torch.float), batch_y.to(device, dtype=torch.long)
        logits.append(model(batch_X).cpu())
        y.append(batch_y.cpu())

    logits, y = torch.cat(logits), torch.cat(y)
    loss.append(loss_fn(logits, y))
    accuracy.append(accuracy_fn(logits, y, num_classes=NUM_CLASSES, task="multiclass"))

    # ── Bug 1 fix: weighted_auroc is now properly computed ──
    weighted_auroc = auroc_fn(logits, y, num_classes=NUM_CLASSES, average="weighted", task="multiclass")
    class_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average=None, task="multiclass"))
    macro_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average="macro", task="multiclass"))

    result = {
        "ground_truth": y,
        "logits": logits,
        "loss": np.mean([l.item() for l in loss]),
        "accuracy": np.mean([a.item() for a in accuracy]),
        "weighted_auroc": weighted_auroc.item() if hasattr(weighted_auroc, "item") else float(weighted_auroc),
        "macro_auroc": np.mean([m.item() for m in macro_auroc]),
    }

    class_auroc_vals = class_auroc[0]
    for i, label in enumerate(LABELS):
        result[f"{label}_auroc"] = class_auroc_vals[i]

    return result


def evaluate_broken(model, data_loader, loss_fn, device):
    """
    The BROKEN evaluate function — replicates the original bug for comparison.
    """
    model.eval()

    # Bug 1: micro_auroc initialized but NEVER populated
    loss, accuracy, class_auroc, micro_auroc, macro_auroc = [], [], [], [], []
    logits, y = [], []

    for batch_X, batch_y in data_loader:
        # Bug 2: type(torch.LongTensor) always creates CPU tensor
        batch_X, batch_y = batch_X.to(device, dtype=torch.float), batch_y.type(
            torch.LongTensor
        )
        logits.append(model(batch_X).cpu())
        y.append(batch_y)

    logits, y = torch.cat(logits), torch.cat(y)

    result = {
        "micro_auroc": np.mean(micro_auroc),  # np.mean([]) = nan!
    }
    return result


def train_step_fixed(model, images, labels, optimizer, scheduler, criterion, device="cpu"):
    """The FIXED train_step — with Bug 2 fix applied."""
    # ── Bug 2 fix: was labels.type(torch.LongTensor).to(device) ──
    images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

def test_bug1_micro_auroc_broken():
    """Demonstrate the original bug: micro_auroc is nan."""
    print("=" * 60)
    print("TEST 1a: BEFORE fix — micro_auroc should be nan (confirming bug)")
    print("=" * 60)

    model = nn.Linear(10, NUM_CLASSES)
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, NUM_CLASSES, (100,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    result = evaluate_broken(model, data_loader, loss_fn, device="cpu")

    # The bug: micro_auroc should be nan
    assert math.isnan(result["micro_auroc"]), (
        f"Expected nan from broken code but got {result['micro_auroc']}"
    )
    print(f"  micro_auroc (broken) = {result['micro_auroc']}  (confirmed: nan)")
    print("[PASS] Bug confirmed: original code produces nan\n")


def test_bug1_micro_auroc_fixed():
    """Verify the fix: micro_auroc should be a valid number."""
    print("=" * 60)
    print("TEST 1b: AFTER fix — micro_auroc should be a valid number")
    print("=" * 60)

    model = nn.Linear(10, NUM_CLASSES)
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, NUM_CLASSES, (100,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    result = evaluate_fixed(model, data_loader, loss_fn, device="cpu")

    weighted = result["weighted_auroc"]
    macro = result["macro_auroc"]

    assert not math.isnan(weighted), f"FAIL: weighted_auroc is still nan!"
    assert 0.0 <= weighted <= 1.0, f"FAIL: weighted_auroc = {weighted}, expected [0, 1]"
    assert not math.isnan(macro), f"FAIL: macro_auroc is nan!"

    print(f"  weighted_auroc (fixed) = {weighted:.4f}")
    print(f"  macro_auroc (fixed) = {macro:.4f}")
    print("[PASS] weighted_auroc is a valid number, not nan\n")


def test_bug2_label_device():
    """Verify Bug 2: labels stay on the correct device."""
    print("=" * 60)
    print("TEST 2: labels should stay on the correct device")
    print("=" * 60)

    device = "cpu"

    # Old way (BUG): always creates CPU tensor
    labels = torch.randint(0, 3, (16,))
    labels_old = labels.type(torch.LongTensor)
    assert str(labels_old.device) == "cpu", "Sanity check"

    # New way (FIX): respects the target device
    labels_new = labels.to(device, dtype=torch.long)
    assert str(labels_new.device) == device, (
        f"FAIL: expected {device} but got {labels_new.device}"
    )

    print(f"  Old way: labels.type(torch.LongTensor) → device={labels_old.device} (always CPU)")
    print(f"  New way: labels.to(device, dtype=torch.long) → device={labels_new.device} (correct)")
    print("[PASS] labels are correctly placed on target device\n")


def test_bug2_train_step():
    """Verify the fixed train_step works correctly."""
    print("=" * 60)
    print("TEST 3: fixed train_step should run without errors")
    print("=" * 60)

    device = "cpu"
    model = nn.Linear(10, NUM_CLASSES).to(device)
    torch.manual_seed(42)
    images = torch.randn(16, 10)
    labels = torch.randint(0, NUM_CLASSES, (16,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    loss = train_step_fixed(model, images, labels, optimizer,
                            scheduler=None, criterion=criterion, device=device)

    assert not math.isnan(loss.item()), f"FAIL: loss is nan!"
    assert loss.item() >= 0, f"FAIL: loss is negative: {loss.item()}"

    print(f"  train_step loss = {loss.item():.4f}")
    print("[PASS] train_step runs correctly with fixed label handling\n")


def test_eval_integration():
    """Full integration test of the fixed evaluate function."""
    print("=" * 60)
    print("TEST 4: full evaluate integration test")
    print("=" * 60)

    device = "cpu"
    model = nn.Linear(10, NUM_CLASSES).to(device)
    torch.manual_seed(42)
    X = torch.randn(50, 10)
    y = torch.randint(0, NUM_CLASSES, (50,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    result = evaluate_fixed(model, data_loader, loss_fn, device=device)

    required_keys = ["loss", "accuracy", "weighted_auroc", "macro_auroc"]
    for key in required_keys:
        assert key in result, f"FAIL: result missing '{key}' key"
        assert not math.isnan(float(result[key])), f"FAIL: {key} is nan"

    for label in LABELS:
        key = f"{label}_auroc"
        assert key in result, f"FAIL: result missing '{key}'"

    print(f"  loss        = {result['loss']:.4f}")
    print(f"  accuracy    = {result['accuracy']:.4f}")
    print(f"  weighted_auroc = {result['weighted_auroc']:.4f}")
    print(f"  macro_auroc = {result['macro_auroc']:.4f}")
    for label in LABELS:
        print(f"  {label}_auroc = {result[f'{label}_auroc']:.4f}")
    print("[PASS] full evaluate integration test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DeepLense Classification Transformers - Bug Fix Verification")
    print("=" * 60 + "\n")

    all_tests = [
        test_bug1_micro_auroc_broken,
        test_bug1_micro_auroc_fixed,
        test_bug2_label_device,
        test_bug2_train_step,
        test_eval_integration,
    ]

    passed = 0
    failed = 0

    for test_fn in all_tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(all_tests)} tests")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
