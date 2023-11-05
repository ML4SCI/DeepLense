from ray import tune
import numpy as np

LEVIT_CONFIG = {
    "network_type": "LeViT",
    "pretrained": False,
    "image_size": 224,
    "batch_size": 64,
    "num_epochs": 15,
    "optimizer_config": {
        "name": "AdamW",
        "weight_decay": 1e-4, #0.01
        "lr": 0.001,
        "momentum": 0.9,
        "betas": (0.9, 0.999),
        "warmup_epoch": 3,
    },
    "out_features": 128, 
    "optimizer_finetune_config": {
        "name": "AdamW",
        "weight_decay": 0.01, #0.01
        "lr": 3e-4,
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
        "stages": 4,  # number of stages
        "dim": (128, 256, 128),  # dimensions at each stage
        "depth": 7,  # transformer of depth 4 at each stage
        "heads": (5, 6, 7),  # heads at each stage
        "mlp_mult": 3,
        "dropout": 0.1,
    },
}


LEVIT_RAY_CONFIG = {
    "network_type": "LeViT",
    "pretrained": False,
    "image_size": 224,
    "batch_size": 128,
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
        "stages": tune.randint(2, 5),  # number of stages
        "dim": (64, 128, 128),  # dimensions at each stage
        "depth": tune.randint(3, 6),  # transformer of depth 4 at each stage
        "heads": (2, 4, 5),  # heads at each stage
        "mlp_mult": tune.randint(2, 4),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
    },
}
