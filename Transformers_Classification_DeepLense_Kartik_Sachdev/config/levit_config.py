LEVIT_CONFIG = {
    "network_type": "LeViT",
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
        "stages": 3,  # number of stages
        "dim": (64, 128, 128),  # dimensions at each stage
        "depth": 5,  # transformer of depth 4 at each stage
        "heads": (2, 4, 5),  # heads at each stage
        "mlp_mult": 2,
        "dropout": 0.1,
    },
}
