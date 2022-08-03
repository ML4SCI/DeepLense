CROSSVIT_CONFIG = {
    "network_type": "CrossViT",
    "pretrained": False,
    "image_size": 256,
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
        "depth": 4,  # number of multi-scale encoding blocks
        "sm_dim": 128,  # high res dimension
        "sm_patch_size": 16,  # high res patch size (should be smaller than lg_patch_size)
        "sm_enc_depth": 4,  # high res depth
        "sm_enc_heads": 2,  # high res heads
        "sm_enc_mlp_dim": 128,  # high res feedforward dimension
        "lg_dim": 64,  # low res dimension
        "lg_patch_size": 64,  # low res patch size
        "lg_enc_depth": 4,  # low res depth
        "lg_enc_heads": 2,  # low res heads
        "lg_enc_mlp_dim": 128,  # low res feedforward dimensions
        "cross_attn_depth": 2,  # cross attention rounds
        "cross_attn_heads": 4,  # cross attention heads
        "dropout": 0.1,
        "emb_dropout": 0.1,
    },
}
