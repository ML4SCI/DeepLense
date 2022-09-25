## Configuration (before training)

Make your required changes in config.py (placeholders provided)

__(all fields are required)__

*EPOCHS* = No. of training iterations (INT)

*LEARNING_RATE* = Learning rate of the optimizer (FLOAT)

*BATCH_SIZE* = Batch size for mini-batch gradient descent (INT)

*LOAD_PRETRAINED_MODEL* = If you want to load a pre-trained model (residing in your MODEL_PATH) and train it further (True/False)

*SAVE_MODEL* = If you want to save your model post training (True/False)

<br>

*MODEL_PATH* = 'Path to save model (if you do not have a .pth file already in the directory, choose a name for your model with .pth extension)'

*TRAIN_DATA_PATH* = 'Path to your training data folder ('\\\*\\\*' is necessary at the end because the dataloader is using the glob package to access all files)'

*TEST_DATA_PATH* = 'Path to your testing data folder ('\\\*\\\*' is necessary at the end because the dataloader is using the glob package to access all files)'

<br>

## Training

```
cd to_particular_model_directory_having_train.py
python train.py
```
<br>

## Testing 

```
cd to_particular_model_directory_having_test.py
python test.py
```