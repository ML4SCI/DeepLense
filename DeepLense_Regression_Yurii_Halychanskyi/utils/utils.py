import os
# Applies z-standartization to make the dataset mean=0 and std=1
def standardize(element,STD,MEAN):
    return (element - MEAN) / STD
# Cancel the effect of z-standartization
def inv_standardize(element,STD,MEAN):
    return element * STD + MEAN
# Checks whether a path contains a file
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

# Checks whether a path is a correct directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
