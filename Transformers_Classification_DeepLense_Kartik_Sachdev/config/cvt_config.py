from ray import tune
import numpy as np

CvT_CONFIG = {
    "network_type": "CvT",
    "pretrained": False,
    "image_size": 224,
    "batch_size": 64,
    "num_epochs": 15,
    "optimizer_config": {
        "name": "AdamW",
        "weight_decay": 1e-4,  # 0.01
        "lr": 0.001,
        "momentum": 0.9,
        "betas": (0.9, 0.999),
        "warmup_epoch": 3,
    },
    "out_features": 128,
    "optimizer_finetune_config": {
        "name": "AdamW",
        "weight_decay": 0.01,  # 0.01
        "lr": 0.001,
        "momentum": 0.9,
        "betas": (0.9, 0.999),
        "warmup_epoch": 3,
    },
    "lr_schedule_config": {
        "use_lr_schedule": True,
        "step_lr": {
            "gamma": 0.5,
            "step_size": 20,
        },
        "reduce_on_plateau": {
            "factor": 0.1,
            "patience": 4,
            "threshold": 0.0000001,
            "verbose": True,
        },
    },
    "channels": 1,
    "network_config": {
        "s1_emb_dim": 128,  # stage 1 - dimension
        "s1_emb_kernel": 7,  # stage 1 - conv kernel size
        "s1_emb_stride": 4,  # stage 1 - conv stride
        "s1_proj_kernel": 3,  # stage 1 - attention ds-conv kernel size
        "s1_kv_proj_stride": 2,  # stage 1 - attention key / value projection stride
        "s1_heads": 4,  # stage 1 - heads
        "s1_depth": 4,  # stage 1 - depth
        "s1_mlp_mult": 8,  # stage 1 - feedforward expansion factor
        "s2_emb_dim": 256,  # stage 2 - (same as above)
        "s2_emb_kernel": 3,
        "s2_emb_stride": 2,
        "s2_proj_kernel": 3,
        "s2_kv_proj_stride": 2,
        "s2_heads": 6,
        "s2_depth": 6,
        "s2_mlp_mult": 8,
        "s3_emb_dim": 512,  # stage 3 - (same as above)
        "s3_emb_kernel": 3,
        "s3_emb_stride": 2,
        "s3_proj_kernel": 3,
        "s3_kv_proj_stride": 2,
        "s3_heads": 7,
        "s3_depth": 7,
        "s3_mlp_mult": 8,
        "stages": ["s1", "s2", "s3"],
        "mlp_last": 512,
        "dropout": 0.1,
    },
}


CvT_RAY_CONFIG = {
    "network_type": "CvT",
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
        "step_lr": {
            "gamma": 0.5,
            "step_size": 20,
        },
        "reduce_on_plateau": {
            "factor": 0.1,
            "patience": 4,
            "threshold": 0.0000001,
            "verbose": True,
        },
    },
    "channels": 1,
    "network_config": {
        "s1_emb_dim": 64,  # stage 1 - dimension
        "s1_emb_kernel": 7,  # stage 1 - conv kernel size
        "s1_emb_stride": 4,  # stage 1 - conv stride
        "s1_proj_kernel": 3,  # stage 1 - attention ds-conv kernel size
        "s1_kv_proj_stride": 2,  # stage 1 - attention key / value projection stride
        "s1_heads": tune.randint(1, 3),  # stage 1 - heads
        "s1_depth": tune.randint(1, 3),  # stage 1 - depth
        "s1_mlp_mult": 4,  # stage 1 - feedforward expansion factor
        "s2_emb_dim": 128,  # stage 2 - (same as above)
        "s2_emb_kernel": 3,
        "s2_emb_stride": 2,
        "s2_proj_kernel": 3,
        "s2_kv_proj_stride": 2,
        "s2_heads": tune.randint(1, 4),
        "s2_depth": tune.randint(1, 3),
        "s2_mlp_mult": 4,
        "mlp_last": tune.randint(128, 512),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
    },
}
