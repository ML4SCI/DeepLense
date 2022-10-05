from ray import tune
import numpy as np

TWINSSVT_CONFIG = {
    "network_type": "TwinsSVT",
    "pretrained": False,
    "image_size": 224,
    "batch_size": 64,
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
        "s1_emb_dim": 16,  # stage 1 - patch embedding projected dimension
        "s1_patch_size": 4,  # stage 1 - patch size for patch embedding
        "s1_local_patch_size": 7,  # stage 1 - patch size for local attention
        "s1_global_k": 7,  # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
        "s1_depth": 1,  # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
        "s2_emb_dim": 16,  # stage 2 (same as above)
        "s2_patch_size": 2,
        "s2_local_patch_size": 7,
        "s2_global_k": 7,
        "s2_depth": 2,
        "s3_emb_dim": 16,  # stage 3 (same as above)
        "s3_patch_size": 2,
        "s3_local_patch_size": 7,
        "s3_global_k": 7,
        "s3_depth": 3,
        "s4_emb_dim": 16,  # stage 4 (same as above)
        "s4_patch_size": 2,
        "s4_local_patch_size": 7,
        "s4_global_k": 7,
        "s4_depth": 1,
        "peg_kernel_size": 3,  # positional encoding generator kernel size
        "dropout": 0.0,  # dropout
        "heads": 5,
    },
}


TWINSSVT_RAY_CONFIG = {
    "network_type": "TwinsSVT",
    "pretrained": False,
    "image_size": 224,
    "batch_size": tune.choice([16, 32, 64]),
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
        "s1_emb_dim": 16,  # stage 1 - patch embedding projected dimension
        "s1_patch_size": 4,  # stage 1 - patch size for patch embedding
        "s1_local_patch_size": 7,  # stage 1 - patch size for local attention
        "s1_global_k": 7,  # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
        "s1_depth": 1,  # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
        "s2_emb_dim": 16,  # stage 2 (same as above)
        "s2_patch_size": 2,
        "s2_local_patch_size": 7,
        "s2_global_k": 7,
        "s2_depth": 2,
        "s3_emb_dim": 16,  # stage 3 (same as above)
        "s3_patch_size": 2,
        "s3_local_patch_size": 7,
        "s3_global_k": 7,
        "s3_depth": 3,
        "s4_emb_dim": 16,  # stage 4 (same as above)
        "s4_patch_size": 2,
        "s4_local_patch_size": 7,
        "s4_global_k": 7,
        "s4_depth": tune.randint(1, 3),
        "peg_kernel_size": 3,  # positional encoding generator kernel size
        "dropout": tune.choice([0.0, 0.1, 0.2]),  # dropout
        "heads": tune.randint(2, 6),
    },
}


#  tune.sample_from(lambda _: np.random.randint(2, 6)), # tune.randint
# tune.sample_from(lambda _: np.random.randint(1, 3))
