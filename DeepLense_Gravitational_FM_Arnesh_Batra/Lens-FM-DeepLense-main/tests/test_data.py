import numpy as np

from lens_lejepa.data import (
    AxionMassDataset,
    LensClassificationDataset,
    SyntheticSuperResolutionDataset,
    UnlabelledLensDataset,
    few_shot_split,
    quantile_split,
    quantile_subsample,
    stratified_split,
)
from lens_lejepa.data.io import load_lens_image


def test_loader_handles_every_storage_format(tmp_path):
    image = np.random.rand(20, 20).astype(np.float32)
    np.save(tmp_path / "plain.npy", image)
    np.save(tmp_path / "channel.npy", image[None])
    wrapped = np.empty(2, dtype=object)
    wrapped[0], wrapped[1] = image, 1e-23
    np.save(tmp_path / "axion.npy", wrapped, allow_pickle=True)
    for name in ("plain", "channel", "axion"):
        assert load_lens_image(tmp_path / f"{name}.npy").shape == (1, 20, 20)


def test_datasets(fake_root):
    cls = LensClassificationDataset(fake_root, image_size=32)
    item = cls[0]
    assert item["image"].shape == (1, 32, 32)
    assert abs(float(item["image"].mean())) < 1e-4  # standardised
    assert sorted(set(cls.labels.tolist())) == [0, 1, 2]

    reg = AxionMassDataset(fake_root, image_size=32)
    assert len(reg) == 12 and np.all((reg.targets >= -24) & (reg.targets <= -22))

    sr = SyntheticSuperResolutionDataset(fake_root, image_size=32)
    pair = sr[0]
    assert pair["high_res"].shape == (1, 64, 64) and pair["low_res"].shape == (1, 32, 32)
    assert pair["high_res"].min() >= 0 and pair["high_res"].max() <= 1

    ssl = UnlabelledLensDataset(fake_root, image_size=32, max_samples=10)
    assert len(ssl) == 10 and set(ssl[0]) == {"image"}


def test_stratified_split_is_disjoint_and_deterministic():
    labels = np.repeat([0, 1, 2], 100)
    train, val = stratified_split(labels, 0.8, seed=42)
    assert not set(train) & set(val) and len(train) + len(val) == 300
    assert np.bincount(labels[train]).tolist() == [80, 80, 80]
    assert np.array_equal(train, stratified_split(labels, 0.8, seed=42)[0])


def test_few_shot_split():
    labels = np.repeat([0, 1, 2], 100)
    train, val = few_shot_split(labels, shots_per_class=20, val_fraction=0.25, seed=0)
    assert np.bincount(labels[train]).tolist() == [20, 20, 20]
    assert np.bincount(labels[val]).tolist() == [5, 5, 5]
    assert not set(train) & set(val)


def test_quantile_split_covers_range():
    values = np.random.RandomState(0).uniform(-24, -22, 1000)
    train, val = quantile_split(values, 0.8, seed=0)
    assert not set(train) & set(val) and len(train) + len(val) == 1000
    assert values[val].min() < -23.7 and values[val].max() > -22.3
    subset = quantile_subsample(values, 100, seed=0)
    assert len(subset) == len(set(subset)) == 100


def test_shift_image_and_training_jitter(fake_root):
    from lens_lejepa.data.io import shift_image

    image = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    shifted = shift_image(image, 1, -1)
    assert shifted[0, 1, 0] == image[0, 0, 1] and shifted[0, 0, 0] == image.min()
    jittered = LensClassificationDataset(fake_root, 32, jitter=2)
    assert jittered[0]["image"].shape == (1, 32, 32)
