import torch
import random
import numpy as np
import os

def set_seed(seed):
    #Sets the seed for Reproducibility
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class Pretraining_HPAMS:
    def __init__(self):
        #Pretraining_model_hyperparameters
        self.pretraining_epochs = 7
        self.pretraining_warmup_epochs = 3
        self.pretraining_learning_rate = 1e-4
        self.pretraining_weight_decay = 1e-5


class ADDA_HPAMS:
    def __init__(self):
        # ADDA_MODEL_HYperparameters
        self.adversarial_epochs = 5
        self.adversarial_warmup_epochs = 2
        self.discriminator_learning_rate = 1e-4
        self.discriminator_weight_decay = 1e-5
        self.target_learning_rate = 1e-6
        self.targetweight_decay = 1e-5

class SE_HPAMS:
    def __init__(self):
        # SE_MODEL_HYperparameters
        self.epochs = 5
        self.warmup_epochs = 2
        self.source_learning_rate = 1e-4
        self.source_weight_decay = 1e-5
        self.target_learning_rate = 1e-4
        self.target_weight_decay = 1e-5

class Adamatch_HPAMS:
    def __init__(self):
        # Adamatch_MODEL_HYperparameters
        self.tau = 0.9
        self.epochs = 5
        self.warmup_epochs = 1
        self.learning_rate = 3e-4
        self.weight_decay = 1e-5

