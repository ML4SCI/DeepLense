import torch.optim as optim
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Creates a Cosine Learning Rate Schedule with a warmup phase.
    Crucial for training large Vision Transformers on lensing data.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    print("DeepLense Scheduler Utility Initialized.")
