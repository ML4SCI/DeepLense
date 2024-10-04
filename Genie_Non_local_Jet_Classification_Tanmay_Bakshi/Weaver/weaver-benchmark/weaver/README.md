# Weaver

`Weaver` aims at providing a streamlined yet flexible machine learning R&D framework for high energy physics (HEP) applications. It puts particular emphases on:

- handling common HEP dataset formats (ROOT, HDF5, [awkward array](https://github.com/scikit-hep/awkward-array)) efficiently, up to terabyte-level
- providing a simple way to perform input processing _on-the-fly_ (e.g., sample selections, new variable definition, inputs transformation/standardization, sample reweighting, etc.)
- bridging the gap between development and production: neural networks are trained with [PyTorch](https://pytorch.org/) and exported to the [ONNX](http://onnx.ai/) format for fast inference (e.g., using [ONNXRuntime](https://github.com/microsoft/onnxruntime))

> Compared to its predecessor [NNTools](https://github.com/hqucms/NNTools/), `Weaver` simplifies the data processing pipeline by running all the pre-processing on-the-fly, without the necessity of creating an intermediate transformed dataset (though it still supports that). The neural network training now uses the more widely adopted `PyTorch` instead of `Apache MXNet`.

<!-- TOC -->

- [Weaver](#weaver)
    - [Set up your environment](#set-up-your-environment)
        - [Install Miniconda (if you don't already have it)](#install-miniconda-if-you-dont-already-have-it)
        - [Set up a conda environment and install the required packages](#set-up-a-conda-environment-and-install-the-required-packages)
    - [Prepare your configuration files](#prepare-your-configuration-files)
        - [Data configuration file](#data-configuration-file)
        - [Model configuration file](#model-configuration-file)
    - [Start running!](#start-running)
        - [Training](#training)
        - [Prediction/Inference](#predictioninference)
        - [Model exportation](#model-exportation)
    - [More about data loading and processing](#more-about-data-loading-and-processing)
        - [Training mode](#training-mode)
        - [Prediction/Inference mode](#predictioninference-mode)
    - [Performance consideration](#performance-consideration)

<!-- /TOC -->

## Set up your environment

The `Weaver` package requires Python 3.7+ and a number of packages like `numpy`, `scikit-learn`, `PyTorch`, etc.
To run the neural network training, a Nvidia GPU with [CUDA](https://developer.nvidia.com/cuda-downloads) support is needed.

To manage the Python environment, we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Below are the instructions to set up Miniconda and install the required packages. This needs to be done only once.

### Install Miniconda (if you don't already have it)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the instructions to finish the installation

# Make sure to choose `yes` for the following one to let the installer initialize Miniconda3
# > Do you wish the installer to initialize Miniconda3
# > by running conda init? [yes|no]

# disable auto activation of the base environment
conda config --set auto_activate_base false
```

Verify the installation is successful by running `conda info` and check if the paths are pointing to your Miniconda installation.

If you cannot run `conda` command, check if you need to add the conda path to your `PATH` variable in your bashrc/zshrc file, e.g.,

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
```

### Set up a conda environment and install the required packages

```bash
# create a new conda environment
conda create -n weaver python=3.7

# activate the environment
conda activate weaver

# install the necessary python packages
pip install numpy pandas scikit-learn scipy matplotlib tqdm PyYAML

# install uproot for reading/writing ROOT files
pip install uproot3 awkward0 lz4 xxhash

# install PyTables if using HDF5 files
pip install tables

# install onnxruntime if needs to run inference w/ ONNX models
pip install onnxruntime-gpu

# install tensorboard for visualization
pip install tensorboard

# install pytorch, follow instructions for your OS/CUDA version at:
# https://pytorch.org/get-started
# pip install torch
```

## Prepare your configuration files

To train a neural network using `Weaver`, you need to prepare:

- A YAML _data configuration file_ describing how to process the input data.
- A python _model configuration file_ providing the neural network module and the loss function.

### Data configuration file

The data configuration file is a [YAML](https://yaml.org/) format file describing how to process the input data. It needs the following sections:

- `selection` (optional): event selection for training
- `test_time_selection` (optional): event selection for testing; if not specified, use the same as `selection`
- `new_variables` (optional): new variable definition
- `inputs` (required): input groups, variables for each group, variable transformation (mean/scale/min/max for standardization, length/pad values for padding/clipping, etc.)
- `labels` (required): label definition
- `observers` (optional): additional variables that are not used in the training, but will be added to the output files when running prediction
- `weights` (optional): instance weight information for sampling the instances during the training

An example of the data configuration file is [data/ak15_points_pf_sv.yaml](data/ak15_points_pf_sv.yaml).
For more details, check [utils/data/config.py](utils/data/config.py) and [utils/dataset.py](utils/dataset.py).

### Model configuration file

The model configuration file specifies the neural network model and the loss function. It needs to implement the `get_model` function (required) and the `get_loss` fucntion (optional, default is `torch.nn.CrossEntropyLoss()`) in the following signatures:

```python
def get_model(data_config, **kwargs):
    model = ... # instance of PyTorch Module
    model_info = {
        'input_names':...,
        'input_shapes':...,
        'output_names':...,
        'dynamic_axes':...,
        }
    return model, model_info

# `get_loss` is optional
# if not provided, fallback to use `torch.nn.CrossEntropyLoss()`
def get_loss(data_config, **kwargs):
    loss_func = ...
    return loss_func
```

An example of the model configuration file is [networks/particle_net_pf_sv.py](networks/particle_net_pf_sv.py).

## Start running!

The [train.py](train.py) script is the top-level script to run for training a neural net, getting prediction from trained models, and exporting trained models to ONNX for production.
To check all the command-line options for `train.py`, run `python train.py -h`. Examples for training, inference and model exportation are shown below:

### Training

```bash
python train.py --data-train '/path/to/train_files/*/*/*/*/output_*.root' \
 --data-test '/path/to/train_files/*/*/*/*/output_*.root' \
 --data-config data/ak15_points_pf_sv.yaml \
 --network-config networks/particle_net_pf_sv.py \
 --model-prefix /path/to/models/prefix \
 --gpus 0,1,2,3 --batch-size 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
 --log logs/train.log
```

Note:

- `--data-train` and `--data-test` supports providing multiple entries, e.g., `--data-train /path/to/A /path/to/B /path/to/C`, and each entry also supports wildcards (`*`, `?`, etc. -- The python `glob` package is used to parse the paths).
- `--data-test` is optional: if specified, the performance on the testing dataset will be automatcially evaluated after the training, using the epoch giving the best performance on the validation set. The prediction output can be saved if `--predict-output` is also set.
- for training, `--model-prefix` sets the _prefix_ part of the paths to save model snapshots.
  At the end of each epoch, the model parameters will be saved to `/path/to/models/prefix_epoch-%d_state.pt`,
  and the optimizer states will be saved to `/path/to/models/prefix_epoch-%d_optimizer.pt` in case the training is interrupted and needed to be resumed from a certain epoch.
  - One can also uses a auto-generated path by including `{auto}` as part of the `--model-prefix`, then `{auto}` will be replaced by a string based on timestamp and the hash of the network configuration.
- for small datasets, it's more efficient to use `--in-memory` to load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run.
- when training on remote files (e.g., from EOS filesystem), one could consider adding `--copy-inputs` so the files are copied to the local workdir to speed up data loading.
- training can be resumed by adding `--load-epoch [last_epoch]`: with this option, the training will continue at `last_epoch + 1`, and the optimizer states and the learning rate will be properly restored.

### Prediction/Inference

Once you have a trained model, you can load it to run prediction and test its performance, e.g.,

```bash
python train.py --predict --data-test '/path/to/test_files/*/*/*/*/output_*.root' \
 --data-config data/ak15_points_pf_sv.yaml \
 --network-config networks/particle_net_pf_sv.py \
 --model-prefix /path/to/models/prefix_best_epoch_state.pt \
 --gpus 0,1,2,3 --batch-size 512 \
 --predict-output /path/to/output.root
```

Note:

- `--data-test` supports providing multiple entries, e.g., `--data-test /path/to/A /path/to/B /path/to/C`, and each entry also supports wildcards
  (`*`, `?`, etc. -- The python `glob` package is used to parse the paths).
- for inference, one can use a data configuration file with different `selection` / `observers` sections, but the `inputs` and `labels` sections must remain unchanged.
- for inference, one can specify the full path of the model parameters in `--model-prefix`.
- `--predict-output` sets the path for the output file. It can either be the full path (if `/` is contained in the path), or just the file name part (e.g., `output.root`) so that the output will be written under the directory of the `--model-prefix`, i.e., `{model_prefix_dir}/predict_output/{predict_output}`. Currently support saving to ROOT files (use `.root` extension) or awkward arrays (use `.awkd` extension).

### Model exportation

When you are satisfied with the trained model, you could export it from PyTorch to ONNX format for inference (e.g., using [ONNXRuntime](https://github.com/microsoft/onnxruntime)):

```bash
python train.py -c data/ak15_points_pf_sv.yaml -n networks/particle_net_pf_sv.py -m /path/to/models/prefix_best_epoch_state.pt --export-onnx model.onnx
```

## More about data loading and processing

To cope with large datasets, the data loader in `Weaver` does not read all input files into memory, but rather load the input events incrementally. The implementation follows the `PyTorch` [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) interface. To speed up the data loading process, [multi-process data loading](https://pytorch.org/docs/stable/data.html#multi-process-data-loading) is also implemented.

**[Note]** For small dataset that actually fits into the memory, use `--in-memory` to load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run.

The [data loader](utils/dataset.py) in `Weaver` operates in different ways for training and prediction/inference.

### Training mode

For training, properly mixing events of different types (e.g., signal/background processes, different kinematic phase space, etc.) and random shuffling are often helpful for improving the performance and stability of the training.

To achieve this efficiently, `Weaver` divides all input files randomly into `N` groups and will load them concurrently with `N` worker threads (`N` is set by `--num-workers`). Then, two data loading strategies are available at the worker thread level:

- [**Default**] The "event-based" strategy attempts to read all the input files (assigned to this worker thread) at each step in order to "maximally" mix events. To meet the memory constraint, for every step, only a small chunk of events is loaded from each input file, and then randomly shuffled before being fed to the training pipeline. The chunk size is set by `--fetch-step` (default is 0.01), corresponding to the fraction (i.e., 10% by default) of events to be loaded from each file in every step. This is the default strategy as, for typical HEP datasets, each individual input file originates from a specific physics process, thus contains events of only a particular type / limited phase space. Note that while this approach ensures a good mixing of events, it requires a high reading throughput of the data storage (thus a fast SSD is preferred), otherwise data loading can become a bottleneck in the training speed.

  - Note: consider setting a smaller `--fetch-step` if the memory limit is exceeded.

- An alternative approach is the "file-based" strategy, which can be enabled with `--fetch-by-files`. This approach will instead read all events from every file for each step, and it will read `m` input files (`m` is set by `--fetch-step`) before mixing and shuffling the loaded events. This strategy is more suitable when each input file is already a mixture of all types of events (e.g., pre-processed with [NNTools](https://github.com/hqucms/NNTools/)), otherwise it may lead to suboptimal training performance. However, a higher data loading speed can generally be achieved with this approach.

**[Note]** If the dataset is stored on EOS and the size is not very large, it may be more efficient to transfer the files to the local storage of the worker node when submitting batch jobs. This can be done on-the-fly by adding the `--copy-inputs` option.

### Prediction/Inference mode

Contrary to training, for prediction/inference, the events are not mixed/shuffled. Instead, the order of the events are preserved, exactly as in the input files. Therefore, only the "file-based" strategy described above is supported, and the `--fetch-step` is fixed to 1.

For more details of the data loader, please check [utils/dataset.py](utils/dataset.py).

## Performance consideration

Loading data from disk can often become a bottleneck. Here are a few tips to get better data loading performance:

- When using ROOT files as inputs, prepare the files w/ `LZ4` compression:

```C++
f.SetCompressionAlgorithm(ROOT::kLZ4);
f.SetCompressionLevel(4);
```

- Copy files to a faster disk (e.g., SSD) if possible.
- Enable multiprocessing for data loading. Setting `--num-workers` to 2 or 3 generally gives a good performance. Setting this value too high could overload the disk and degrade the performance.
  - Note that the memory usage also increases with the number of workers. So if you are getting any memory-related errors, try reducing `--num-workers`.
  - Note that the workload splitting is file-based, so make sure the number of input files is not too small (i.e., make sure each worker is able to load several files to get samples _from all classes_).
    - **e.g., if each (signal/background) class is present in only one input file, please use `--num-workers 1` so that they are properly mixed for the training.**
