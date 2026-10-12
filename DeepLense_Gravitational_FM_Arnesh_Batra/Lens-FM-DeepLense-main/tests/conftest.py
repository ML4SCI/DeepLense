import numpy as np
import pytest
import torch

torch.set_num_threads(4)


@pytest.fixture
def fake_root(tmp_path):
    rng = np.random.RandomState(0)
    for name in ("axion", "cdm", "no_sub"):
        (tmp_path / name).mkdir()
        for i in range(12):
            image = rng.rand(64, 64).astype(np.float32)
            if name == "axion":
                payload = np.empty(2, dtype=object)
                payload[0], payload[1] = image, 10 ** rng.uniform(-24, -22)
                np.save(tmp_path / name / f"{i}.npy", payload, allow_pickle=True)
            else:
                np.save(tmp_path / name / f"{i}.npy", image)
    return tmp_path

