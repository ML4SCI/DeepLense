from ray import tune
import numpy as np

SWIN_CONFIG = {
    "network_type": "Swin",
    "pretrained": False,
    "image_size": 128,
    "batch_size": 32,
    "num_epochs": 15,
    "optimizer_config": {
        "name": "AdamW",
        "weight_decay": 0.01,
        "lr": 0.001,
        "momentum": 0.9,
        "betas": (0.9, 0.999),
        "warmup_epoch": 3,
    },
    "lr_schedule_config": {
        "use_lr_schedule": True,
        "step_lr": {"gamma": 0.5, "step_size": 20,},
        "reduce_on_plateau": {
            "factor": 0.1,
            "patience": 4,
            "threshold": 0.0000001,
            "verbose": True,
        },
    },
    "channels": 1,
    "network_config": {
        "patch_size": 4,
        "window_size": 4,
        "embed_dim": 50,
        "in_chans": 1,
        "drop_path_rate": 0,
        "depths": (2, 2, 2, 3),
        "num_heads": (2, 2, 2, 2),
        "mlp_ratio": 1,
    },
}


SWIN_RAY_CONFIG = {
    "network_type": "Swin",
    "pretrained": False,
    "image_size": 128,
    "batch_size": 32,
    "num_epochs": 15,
    "optimizer_config": {
        "name": "AdamW",
        "weight_decay": 0.01,
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "momentum": tune.uniform(0.7, 0.99),
        "betas": (0.9, 0.999),
        "warmup_epoch": 3,
    },
    "lr_schedule_config": {
        "use_lr_schedule": True,
        "step_lr": {"gamma": 0.5, "step_size": 20,},
        "reduce_on_plateau": {
            "factor": 0.1,
            "patience": 4,
            "threshold": 0.0000001,
            "verbose": True,
        },
    },
    "channels": 1,
    "network_config": {
        "patch_size": 4,
        "window_size": 4,
        "embed_dim": tune.randint(64, 256),
        "in_chans": 1,
        "drop_path_rate": 0,
        "depths": (2, 2, 2, 3),
        "num_heads": (2, 2, 2, 2),
        "mlp_ratio": tune.choice([1, 2, 3]),
    },
}
