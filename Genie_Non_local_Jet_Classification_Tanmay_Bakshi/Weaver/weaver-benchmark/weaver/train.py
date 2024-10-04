#!/usr/bin/env python

import os
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch

from torch.utils.data import DataLoader
from importlib import import_module
import ast
from utils.logger import _logger, _configLogger
from utils.dataset import SimpleIterDataset

parser = argparse.ArgumentParser()
parser.add_argument('--regression-mode', action='store_true', default=False,
                    help='run in regression mode if this flag is set; otherwise run in classification mode')
parser.add_argument('-c', '--data-config', type=str, default='data/ak15_points_pf_sv_v0.yaml',
                    help='data config YAML file')
parser.add_argument('-i', '--data-train', nargs='*', default=[],
                    help='training files')
parser.add_argument('-l', '--data-val', nargs='*', default=[],
                    help='validation files; when not set, will use training files and split by `--train-val-split`')
parser.add_argument('-t', '--data-test', nargs='*', default=[],
                    help='testing files; supported syntax:'
                         ' (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;'
                         ' (b) keyword-based, `--data-test: a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;'
                         ' (c) split output per N input files, `--data-test: a%10:/path/to/a/*`, will split per 10 input files')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
parser.add_argument('--file-fraction', type=float, default=1,
                    help='fraction of files to load; for training, the files are randomly selected for each epoch')
parser.add_argument('--fetch-by-files', action='store_true', default=False,
                    help='When enabled, will load all events from a small number (set by ``--fetch-step``) of files for each data fetching. '
                         'Otherwise (default), load a small fraction of events from all files each time, which helps reduce variations in the sample composition.')
parser.add_argument('--fetch-step', type=float, default=0.01,
                    help='fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); '
                         'Or: number of files to load each time (when ``--fetch-by-files`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.')
parser.add_argument('--in-memory', action='store_true', default=False,
                    help='load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run')
parser.add_argument('--train-val-split', type=float, default=0.8,
                    help='training/validation split fraction')
parser.add_argument('--demo', action='store_true', default=False,
                    help='quickly test the setup by running over only a small number of events')
parser.add_argument('--lr-finder', type=str, default=None,
                    help='run learning rate finder instead of the actual training; format: ``start_lr, end_lr, num_iters``')
parser.add_argument('--tensorboard', type=str, default=None,
                    help='create a tensorboard summary writer with the given comment')
parser.add_argument('--tensorboard-custom-fn', type=str, default=None,
                    help='the path of the python script containing a user-specified function `get_tensorboard_custom_fn`, '
                         'to display custom information per mini-batch or per epoch, during the training, validation or test.')
parser.add_argument('-n', '--network-config', type=str, default='networks/particle_net_pfcand_sv.py',
                    help='network architecture configuration file; the path must be relative to the current dir')
parser.add_argument('-o', '--network-option', nargs=2, action='append', default=[],
                    help='options to pass to the model class constructor, e.g., `--network-option use_counts False`')
parser.add_argument('-m', '--model-prefix', type=str, default='models/{auto}/network',
                    help='path to save or load the model; for training, this will be used as a prefix, so model snapshots '
                         'will saved to `{model_prefix}_epoch-%d_state.pt` after each epoch, and the one with the best '
                         'validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path '
                         'including the suffix, otherwise the one with the best validation metric will be used; '
                         'for training, `{auto}` can be used as part of the path to auto-generate a name, '
                         'based on the timestamp and network configuration')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--steps-per-epoch', type=int, default=None,
                    help='number of steps (iterations) per epochs; if not set, each epoch will run over all loaded samples')
parser.add_argument('--steps-per-epoch-val', type=int, default=None,
                    help='number of steps (iterations) per epochs for validation; if not set, each epoch will run over all loaded samples')
parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'adamW', 'ranger'],  # TODO: add more
                    help='optimizer for the training')
parser.add_argument('--optimizer-option', nargs=2, action='append', default=[],
                    help='options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`')
parser.add_argument('--lr-scheduler', type=str, default='flat+decay',
                    choices=['none', 'steps', 'flat+decay', 'flat+linear', 'flat+cos', 'one-cycle'],
                    help='learning rate scheduler')
parser.add_argument('--load-epoch', type=int, default=None,
                    help='used to resume interrupted training, load model and optimizer state saved in the `epoch-%d_state.pt` and `epoch-%d_optimizer.pt` files')
parser.add_argument('--start-lr', type=float, default=5e-3,
                    help='start learning rate')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='use mixed precision training (fp16); NOT WORKING YET')
parser.add_argument('--gpus', type=str, default='0',
                    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--predict', action='store_true', default=False,
                    help='run prediction instead of training')
parser.add_argument('--predict-output', type=str,
                    help='path to save the prediction output, support `.root` and `.awkd` format')
parser.add_argument('--export-onnx', type=str, default=None,
                    help='export the PyTorch model to ONNX model and save it at the given path (path must ends w/ .onnx); '
                         'needs to set `--data-config`, `--network-config`, and `--model-prefix` (requires the full model path)')
parser.add_argument('--io-test', action='store_true', default=False,
                    help='test throughput of the dataloader')
parser.add_argument('--copy-inputs', action='store_true', default=False,
                    help='copy input files to the current dir (can help to speed up dataloading when running over remote files, e.g., from EOS)')
parser.add_argument('--log', type=str, default='',
                    help='path to the log file; `{auto}` can be used as part of the path to auto-generate a name, based on the timestamp and network configuration')
parser.add_argument('--print', action='store_true', default=False,
                    help='do not run training/prediction but only print model information, e.g., FLOPs and number of parameters of a model')
parser.add_argument('--profile', action='store_true', default=False,
                    help='run the profiler')


def to_filelist(args, mode='train'):
    if mode == 'train':
        flist = args.data_train
    elif mode == 'val':
        flist = args.data_val
    else:
        raise NotImplementedError('Invalid mode %s' % mode)

    filelist = sum([glob.glob(f) for f in flist], [])
    np.random.shuffle(filelist)

    if args.copy_inputs:
        import tempfile
        tmpdir = tempfile.mkdtemp()
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        new_filelist = []
        for src in filelist:
            dest = os.path.join(tmpdir, src.lstrip('/'))
            if not os.path.exists(os.path.dirname(dest)):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            _logger.info('Copied file %s to %s' % (src, dest))
            new_filelist.append(dest)
        filelist = new_filelist

    return filelist


def train_load(args):
    """
    Loads the training data.
    :param args:
    :return: train_loader, val_loader, data_config, train_inputs
    """

    train_files = to_filelist(args, 'train')
    if args.data_val:
        val_files = to_filelist(args, 'val')
        train_range = val_range = (0, 1)
    else:
        val_files = train_files
        train_range = (0, args.train_val_split)
        val_range = (args.train_val_split, 1)
    _logger.info('Using %d files for training, range: %s' % (len(train_files), str(train_range)))
    _logger.info('Using %d files for validation, range: %s' % (len(val_files), str(val_range)))

    if args.demo:
        train_files = train_files[:20]
        val_files = val_files[:20]
        _logger.info(train_files)
        _logger.info(val_files)
        args.data_fraction = 0.1
        args.fetch_step = 0.002

    if args.in_memory and (args.steps_per_epoch is None or args.steps_per_epoch_val is None):
        raise RuntimeError('Must set --steps-per-epoch when using --in-memory!')

    train_data = SimpleIterDataset(train_files, args.data_config, for_training=True,
                                   load_range_and_fraction=(train_range, args.data_fraction),
                                   file_fraction=args.file_fraction,
                                   fetch_by_files=args.fetch_by_files,
                                   fetch_step=args.fetch_step,
                                   infinity_mode=args.steps_per_epoch is not None,
                                   in_memory=args.in_memory)
    val_data = SimpleIterDataset(val_files, args.data_config, for_training=True,
                                 load_range_and_fraction=(val_range, args.data_fraction),
                                 file_fraction=args.file_fraction,
                                 fetch_by_files=args.fetch_by_files,
                                 fetch_step=args.fetch_step,
                                 infinity_mode=args.steps_per_epoch_val is not None,
                                 in_memory=args.in_memory)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                              num_workers=min(args.num_workers, int(len(train_files) * args.file_fraction)),
                              persistent_workers=args.num_workers > 0 and args.steps_per_epoch is not None)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                            num_workers=min(args.num_workers, int(len(val_files) * args.file_fraction)),
                            persistent_workers=args.num_workers > 0 and args.steps_per_epoch_val is not None)
    data_config = train_data.config
    train_input_names = train_data.config.input_names
    train_label_names = train_data.config.label_names

    return train_loader, val_loader, data_config, train_input_names, train_label_names


def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}
    for f in args.data_test:
        if ':' in f:
            name, fp = f.split(':')
            if '%' in name:
                name, split = name.split('%')
                split_dict[name] = int(split)
        else:
            name, fp = '', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]

    def get_test_loader(name):
        filelist = file_dict[name]
        _logger.info('Running on test file group %s with %d files:\n...%s', name, len(filelist), '\n...'.join(filelist))
        num_workers = min(args.num_workers, len(filelist))
        test_data = SimpleIterDataset(filelist, args.data_config, for_training=False,
                                      load_range_and_fraction=((0, 1), args.data_fraction),
                                      fetch_by_files=True, fetch_step=1)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=False,
                                 pin_memory=True)
        return test_loader

    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset([], args.data_config, for_training=False).config
    return test_loaders, data_config


def onnx(args, model, data_config, model_info):
    """
    Saving model as ONNX.
    :param args:
    :param model:
    :param data_config:
    :param model_info:
    :return:
    """
    assert (args.export_onnx.endswith('.onnx'))
    model_path = args.model_prefix
    _logger.info('Exporting model %s to ONNX' % model_path)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.cpu()
    model.eval()

    os.makedirs(os.path.dirname(args.export_onnx), exist_ok=True)
    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])
    torch.onnx.export(model, inputs, args.export_onnx,
                      input_names=model_info['input_names'],
                      output_names=model_info['output_names'],
                      dynamic_axes=model_info.get('dynamic_axes', None),
                      opset_version=13)
    _logger.info('ONNX model saved to %s', args.export_onnx)

    preprocessing_json = os.path.join(os.path.dirname(args.export_onnx), 'preprocess.json')
    data_config.export_json(preprocessing_json)
    _logger.info('Preprocessing parameters saved to %s', preprocessing_json)


def flops(model, model_info):
    """
    Count FLOPs and params.
    :param args:
    :param model:
    :param model_info:
    :return:
    """
    from utils.flops_counter import get_model_complexity_info
    import copy

    model = copy.deepcopy(model).cpu()
    model.eval()

    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])

    macs, params = get_model_complexity_info(model, inputs, as_strings=True, print_per_layer_stat=True, verbose=True)
    _logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    _logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))


def profile(args, model, model_info, device):
    """
    Profile.
    :param model:
    :param model_info:
    :return:
    """
    import copy
    from torch.profiler import profile, record_function, ProfilerActivity

    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()

    inputs = tuple(torch.ones((args.batch_size,) + model_info['input_shapes'][k][1:], dtype=torch.float32).to(device) for k in model_info['input_names'])
    for x in inputs:
        print(x.shape, x.device)

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=2),
        on_trace_ready=trace_handler
    ) as p:
        for idx in range(100):
            model(*inputs)
            p.step()


def optim(args, model, device):
    """
    Optimizer and scheduler.
    :param args:
    :param model:
    :return:
    """
    optimizer_options = {k: ast.literal_eval(v) for k, v in args.optimizer_option}
    _logger.info('Optimizer options: %s' % str(optimizer_options))
    if args.optimizer == 'ranger':
        from utils.nn.optimizer.ranger import Ranger
        opt = Ranger(model.parameters(), lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adamW':
        opt = torch.optim.AdamW(model.parameters(), lr=args.start_lr, **optimizer_options)

    # load previous training and resume if `--load-epoch` is set
    if args.load_epoch is not None:
        _logger.info('Resume training from epoch %d' % args.load_epoch)
        model_state = torch.load(args.model_prefix + '_epoch-%d_state.pt' % args.load_epoch, map_location=device)
        model.load_state_dict(model_state)
        opt_state = torch.load(args.model_prefix + '_epoch-%d_optimizer.pt' % args.load_epoch, map_location=device)
        opt.load_state_dict(opt_state)

    scheduler = None
    if args.lr_finder is None:
        if args.lr_scheduler == 'steps':
            lr_step = round(args.num_epochs / 3)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=[lr_step, 2 * lr_step], gamma=0.1,
                last_epoch=-1 if args.load_epoch is None else args.load_epoch)
        elif args.lr_scheduler == 'flat+decay':
            lr_decay_epochs = max(1, int(args.num_epochs * 0.3))
            lr_decay_rate = 0.01 ** (1. / lr_decay_epochs)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(
                range(args.num_epochs - lr_decay_epochs, args.num_epochs)), gamma=lr_decay_rate,
                last_epoch=-1 if args.load_epoch is None else args.load_epoch)
        elif args.lr_scheduler == 'flat+linear' or args.lr_scheduler == 'flat+cos':
            total_steps = args.num_epochs * args.steps_per_epoch
            flat_steps = total_steps * 0.7 - 1
            min_factor = 0.001

            def lr_fn(step_num):
                if step_num > total_steps:
                    raise ValueError(
                        "Tried to step {} times. The specified number of total steps is {}".format(
                            step_num + 1, total_steps))
                if step_num <= flat_steps:
                    return 1.0
                pct = (step_num - flat_steps) / (total_steps - flat_steps)
                if args.lr_scheduler == 'flat+linear':
                    return max(min_factor, 1 - pct)
                else:
                    return max(min_factor, 0.5 * (math.cos(math.pi * pct) + 1))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_fn, last_epoch=-1 if args.load_epoch is None else args.load_epoch * args.steps_per_epoch)
            scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
        elif args.lr_scheduler == 'one-cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.start_lr, epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch, pct_start=0.3,
                anneal_strategy='cos', div_factor=25.0, last_epoch=-1 if args.load_epoch is None else args.load_epoch)
            scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
    return opt, scheduler


def model_setup(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config.replace('.py', '').replace('/', '.'))
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    _logger.info('Network options: %s' % str(network_options))
    if args.export_onnx:
        network_options['for_inference'] = True
    if args.use_amp:
        network_options['use_amp'] = True
    model, model_info = network_module.get_model(data_config, **network_options)
    # _logger.info(model)
    flops(model, model_info)
    return model, model_info, network_module, network_options


def iotest(args, data_loader):
    """
    Io test
    :param args:
    :param data_loader:
    :return:
    """
    from tqdm.auto import tqdm
    from collections import defaultdict
    from utils.data.tools import _concat
    _logger.info('Start running IO test')
    monitor_info = defaultdict(list)

    for X, y, Z in tqdm(data_loader):
        for k, v in Z.items():
            monitor_info[k].append(v.cpu().numpy())
    monitor_info = {k: _concat(v) for k, v in monitor_info.items()}
    if monitor_info:
        monitor_output_path = 'weaver_monitor_info.pkl'
        import pickle
        with open(monitor_output_path, 'wb') as f:
            pickle.dump(monitor_info, f)
        _logger.info('Monitor info written to %s' % monitor_output_path)


def save_root(args, output_path, data_config, scores, labels, observers):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param labels
    :param observers
    :return:
    """
    from utils.data.fileio import _write_root
    output = {}
    if args.regression_mode:
        output[data_config.label_names[0]] = labels[data_config.label_names[0]]
        output['output'] = scores
    else:
        for idx, label_name in enumerate(data_config.label_value):
            output[label_name] = (labels[data_config.label_names[0]] == idx)
            output['score_' + label_name] = scores[:, idx]
    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v
    for k, v in observers.items():
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v
    _write_root(output_path, output)


def save_awk(args, output_path, scores, labels, observers):
    """
    Saves as .awkd
    :param scores:
    :param labels:
    :param observers:
    :return:
    """
    from utils.data.tools import awkward
    output = {'scores': scores}
    output.update(labels)
    output.update(observers)

    name_remap = {}
    arraynames = list(output)
    for i in range(len(arraynames)):
        for j in range(i + 1, len(arraynames)):
            if arraynames[i].startswith(arraynames[j]):
                name_remap[arraynames[j]] = '%s_%d' % (arraynames[j], len(name_remap))
            if arraynames[j].startswith(arraynames[i]):
                name_remap[arraynames[i]] = '%s_%d' % (arraynames[i], len(name_remap))
    _logger.info('Renamed the following variables in the output file: %s', str(name_remap))
    output = {name_remap[k] if k in name_remap else k: v for k, v in output.items()}

    awkward.save(output_path, output, mode='w')


def main(args):
    _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))

    if args.file_fraction < 1:
        _logger.warning('Use of `file-fraction` is not recommended in general -- prefer using `data-fraction` instead.')

    # classification/regression mode
    if args.regression_mode:
        _logger.info('Running in regression mode')
        from utils.nn.tools import train_regression as train
        from utils.nn.tools import evaluate_regression as evaluate
    else:
        _logger.info('Running in classification mode')
        from utils.nn.tools import train_classification as train
        from utils.nn.tools import evaluate_classification as evaluate

    # training/testing mode
    training_mode = not args.predict

    # device
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(',')]
        dev = torch.device(gpus[0])
    else:
        gpus = None
        dev = torch.device('cpu')

    # load data
    if training_mode:
        train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)

    if args.io_test:
        data_loader = train_loader if training_mode else list(test_loaders.values())[0]()
        iotest(args, data_loader)
        return

    model, model_info, network_module, network_options = model_setup(args, data_config)

    if args.print:
        return

    if args.profile:
        profile(args, model, model_info, device=dev)
        return

    # export to ONNX
    if args.export_onnx:
        onnx(args, model, data_config, model_info)
        return

    if args.tensorboard:
        from utils.nn.tools import TensorboardHelper
        tb = TensorboardHelper(tb_comment=args.tensorboard, tb_custom_fn=args.tensorboard_custom_fn)
    else:
        tb = None

    # note: we should always save/load the state_dict of the original model, not the one wrapped by nn.DataParallel
    # so we do not convert it to nn.DataParallel now
    orig_model = model

    if training_mode:
        model = orig_model.to(dev)
        # loss function
        try:
            loss_func = network_module.get_loss(data_config, **network_options)
            _logger.info('Using loss function %s with options %s' % (loss_func, network_options))
        except AttributeError:
            loss_func = torch.nn.CrossEntropyLoss()
            _logger.warning('Loss function not defined in %s. Will use `torch.nn.CrossEntropyLoss()` by default.',
                            args.network_config)

        # optimizer & learning rate
        opt, scheduler = optim(args, model, dev)

        # multi-gpu
        if gpus is not None and len(gpus) > 1:
            # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
            model = torch.nn.DataParallel(model, device_ids=gpus)
        model = model.to(dev)

        # lr finder: keep it after all other setups
        if args.lr_finder is not None:
            start_lr, end_lr, num_iter = args.lr_finder.replace(' ', '').split(',')
            from utils.lr_finder import LRFinder
            lr_finder = LRFinder(model, opt, loss_func, device=dev, input_names=train_input_names,
                                 label_names=train_label_names)
            lr_finder.range_test(train_loader, start_lr=float(start_lr), end_lr=float(end_lr), num_iter=int(num_iter))
            lr_finder.plot(output='lr_finder.png')  # to inspect the loss-learning rate graph
            return

        if args.use_amp:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        else:
            scaler = None

        # training loop
        best_valid_metric = np.inf if args.regression_mode else 0
        for epoch in range(args.num_epochs):
            if args.load_epoch is not None:
                if epoch <= args.load_epoch:
                    continue
            print('-' * 50)
            _logger.info('Epoch #%d training' % epoch)
            train(model, loss_func, opt, scheduler, train_loader, dev, epoch,
                  steps_per_epoch=args.steps_per_epoch, grad_scaler=scaler, tb_helper=tb)
            if args.model_prefix:
                dirname = os.path.dirname(args.model_prefix)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
                torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

            _logger.info('Epoch #%d validating' % epoch)
            valid_metric = evaluate(model, val_loader, dev, epoch, loss_func=loss_func,
                                    steps_per_epoch=args.steps_per_epoch_val, tb_helper=tb)
            is_best_epoch = (
                valid_metric < best_valid_metric) if args.regression_mode else(
                valid_metric > best_valid_metric)
            if is_best_epoch:
                best_valid_metric = valid_metric
                if args.model_prefix:
                    shutil.copy2(args.model_prefix + '_epoch-%d_state.pt' %
                                 epoch, args.model_prefix + '_best_epoch_state.pt')
                    torch.save(model, args.model_prefix + '_best_epoch_full.pt')
            _logger.info('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
                         (epoch, valid_metric, best_valid_metric), color='bold')

    if args.data_test:
        if training_mode:
            del train_loader, val_loader
            test_loaders, data_config = test_load(args)

        if not args.model_prefix.endswith('.onnx'):
            model = orig_model.to(dev)
            model_path = args.model_prefix if args.model_prefix.endswith(
                '.pt') else args.model_prefix + '_best_epoch_state.pt'
            _logger.info('Loading model %s for eval' % model_path)
            model.load_state_dict(torch.load(model_path, map_location=dev))
            if gpus is not None and len(gpus) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpus)
            model = model.to(dev)

        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            # run prediction
            if args.model_prefix.endswith('.onnx'):
                _logger.info('Loading model %s for eval' % args.model_prefix)
                from utils.nn.tools import evaluate_onnx
                test_metric, scores, labels, observers = evaluate_onnx(args.model_prefix, test_loader)
            else:
                test_metric, scores, labels, observers = evaluate(model, test_loader, dev, epoch=None, for_training=False, tb_helper=tb)
            _logger.info('Test metric %.5f' % test_metric, color='bold')
            del test_loader

            if args.predict_output:
                if '/' not in args.predict_output:
                    args.predict_output = os.path.join(
                        os.path.dirname(args.model_prefix),
                        'predict_output', args.predict_output)
                os.makedirs(os.path.dirname(args.predict_output), exist_ok=True)
                if name == '':
                    output_path = args.predict_output
                else:
                    base, ext = os.path.splitext(args.predict_output)
                    output_path = base + '_' + name + ext
                if output_path.endswith('.root'):
                    save_root(args, output_path, data_config, scores, labels, observers)
                else:
                    save_awk(args, output_path, scores, labels, observers)
                _logger.info('Written output to %s' % output_path, color='bold')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split)
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None

    if '{auto}' in args.model_prefix or '{auto}' in args.log:
        import hashlib
        import time
        model_name = time.strftime('%Y%m%d-%H%M%S') + "_" + os.path.basename(args.network_config).replace('.py', '')
        if len(args.network_option):
            model_name = model_name + "_" + hashlib.md5(str(args.network_option).encode('utf-8')).hexdigest()
        model_name += '_{optim}_lr{lr}_batch{batch}'.format(lr=args.start_lr,
                                                            optim=args.optimizer, batch=args.batch_size)
        args._auto_model_name = model_name
        args.model_prefix = args.model_prefix.replace('{auto}', model_name)
        args.log = args.log.replace('{auto}', model_name)
        print('Using auto-generated model prefix %s' % args.model_prefix)

    _configLogger('weaver', filename=args.log)
    main(args)
