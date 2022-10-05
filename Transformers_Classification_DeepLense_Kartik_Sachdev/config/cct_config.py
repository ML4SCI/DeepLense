from ray import tune
import numpy as np

CCT_CONFIG = {
    "network_type": "CCT",
    "pretrained": False,
    "image_size": 224,
    "batch_size": 128,
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
        "embedding_dim": 256,
        "n_conv_layers": 2,
        "kernel_size": 7,
        "stride": 2,
        "padding": 3,
        "pooling_kernel_size": 3,
        "pooling_stride": 2,
        "pooling_padding": 1,
        "num_layers": 6,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "positional_embedding": "learnable",  # ['sine', 'learnable', 'none']
    },
}

CCT_RAY_CONFIG = {
    "network_type": "CCT",
    "pretrained": False,
    "image_size": 224,
    "batch_size": tune.choice([16, 32, 64, 128]),
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
        "embedding_dim": 256,
        "n_conv_layers": 2,
        "kernel_size": 7,
        "stride": 2,
        "padding": 3,
        "pooling_kernel_size": 3,
        "pooling_stride": 2,
        "pooling_padding": 1,
        "num_layers": 6,
        "num_heads": tune.randint(1, 5),
        "mlp_ratio": tune.randint(1, 3),
        "positional_embedding": tune.choice(["learnable", "sine", "none"]),
    },
}

