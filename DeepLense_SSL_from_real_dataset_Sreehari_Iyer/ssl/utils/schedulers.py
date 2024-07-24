# as is in https://github.com/facebookresearch/dino/blob/main/utils.py - cosine_scheduler

import numpy as np

def cosine_scheduler(
        init_val: int, 
        final_val: int, 
        epochs: int, 
        steps_per_epoch: int,
        warmup_epochs: int = 0,
        start_warmup_value: float = 0
    ):
    warmup_schedule = np.array([])
    warmup_steps = warmup_epochs * steps_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, init_val, warmup_steps)

    steps = np.arange(epochs * steps_per_epoch - warmup_steps)
    schedule = final_val + 0.5 * (init_val - final_val) * (1 + np.cos(np.pi * steps / len(steps)))

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * steps_per_epoch, f"len(schedule): {len(schedule)}, total_steps: {epochs*steps_per_epoch}, warmup_steps: {warmup_epochs*steps_per_epoch}, warmup_epochs: {warmup_epochs}, epochs: {epochs}"
    
    return schedule 

