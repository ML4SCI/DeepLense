CROSSFORMER_CONFIG = {
    "network_type": "CrossFormer",
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
        "dim": (32, 64, 128, 256),  # dimension at each stage
        "depth": (2, 2, 4, 2),  # depth of transformer at each stage
        "global_window_size": (8, 4, 2, 1),  # global window sizes at each stage
        "local_window_size": 7,  # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
    },
}

