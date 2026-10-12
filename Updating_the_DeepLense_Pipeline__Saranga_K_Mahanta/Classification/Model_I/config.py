EPOCHS = 30
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
LOAD_PRETRAINED_MODEL = False
SAVE_MODEL = False

import os

# Base directory of Model_I
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Expected dataset structure:
# data/
#   Model_I/
#       axion/
#       cdm/
#       no_sub/
#   Model_I_test/
#       axion/
#       cdm/
#       no_sub/

DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "Model_I")
TEST_DATA_PATH = os.path.join(DATA_DIR, "Model_I_test")

MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "model.pth")
