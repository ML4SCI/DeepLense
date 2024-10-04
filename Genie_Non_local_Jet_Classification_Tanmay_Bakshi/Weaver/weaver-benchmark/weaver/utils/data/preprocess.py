import time
import glob
import copy
import numpy as np

from ..logger import _logger
from .tools import _get_variable_names, _eval_expr
from .fileio import _read_files


def _apply_selection(table, selection):
    if selection is None:
        return
    selected = _eval_expr(selection, table).astype('bool')
    for k in table.keys():
        table[k] = table[k][selected]
    return selected.sum()


def _build_new_variables(table, funcs):
    if funcs is None:
        return
    for k, expr in funcs.items():
        if k in table:
            continue
        table[k] = _eval_expr(expr, table)


def _clean_up(table, drop_branches):
    for k in drop_branches:
        del table[k]


class AutoStandardizer(object):
    r"""AutoStandardizer.

    Class to compute the variable standardization information.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config (DataConfig): object containing data format information.
    """

    def __init__(self, filelist, data_config):
        self._filelist = filelist if isinstance(
            filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()
        self.load_range = (0, data_config.preprocess.get('data_fraction', 0.1))

    def read_file(self, filelist):
        self.keep_branches = set()
        self.load_branches = set()
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] == 'auto':
                self.keep_branches.add(k)
                if k in self._data_config.var_funcs:
                    expr = self._data_config.var_funcs[k]
                    self.load_branches.update(_get_variable_names(expr))
                else:
                    self.load_branches.add(k)
        if self._data_config.selection:
            self.load_branches.update(_get_variable_names(self._data_config.selection))
        _logger.debug('[AutoStandardizer] keep_branches:\n  %s', ','.join(self.keep_branches))
        _logger.debug('[AutoStandardizer] load_branches:\n  %s', ','.join(self.load_branches))
        table = _read_files(filelist, self.load_branches, self.load_range,
                            show_progressbar=True, treename=self._data_config.treename)
        _apply_selection(table, self._data_config.selection)
        _build_new_variables(table, {k: v for k, v in self._data_config.var_funcs.items() if k in self.keep_branches})
        _clean_up(table, self.load_branches - self.keep_branches)
        return table

    def make_preprocess_params(self, table):
        _logger.info('Using %d events to calculate standardization info', len(table[list(table.keys())[0]]))
        preprocess_params = copy.deepcopy(self._data_config.preprocess_params)
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] == 'auto':
                if k.endswith('_mask'):
                    params['center'] = None
                else:
                    a = table[k]
                    try:
                        a = a.content
                    except AttributeError:
                        pass
                    low, center, high = np.percentile(a, [16, 50, 84])
                    scale = max(high - center, center - low)
                    scale = 1 if scale == 0 else 1. / scale
                    params['center'] = float(center)
                    params['scale'] = float(scale)
                preprocess_params[k] = params
                _logger.info('[AutoStandardizer] %s low=%s, center=%s, high=%s, scale=%s', k, low, center, high, scale)
        return preprocess_params

    def produce(self, output=None):
        table = self.read_file(self._filelist)
        preprocess_params = self.make_preprocess_params(table)
        self._data_config.preprocess_params = preprocess_params
        # must also propogate the changes to `data_config.options` so it can be persisted
        self._data_config.options['preprocess']['params'] = preprocess_params
        if output:
            _logger.info(
                'Writing YAML file w/ auto-generated preprocessing info to %s' % output)
            self._data_config.dump(output)
        return self._data_config


class WeightMaker(object):
    r"""WeightMaker.

    Class to make reweighting information.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config (DataConfig): object containing data format information.
    """

    def __init__(self, filelist, data_config):
        self._filelist = filelist if isinstance(filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()

    def read_file(self, filelist):
        self.keep_branches = set(self._data_config.reweight_branches + self._data_config.reweight_classes)
        self.load_branches = set()
        for k in self.keep_branches:
            if k in self._data_config.var_funcs:
                expr = self._data_config.var_funcs[k]
                self.load_branches.update(_get_variable_names(expr))
            else:
                self.load_branches.add(k)
        if self._data_config.selection:
            self.load_branches.update(_get_variable_names(self._data_config.selection))
        _logger.debug('[WeightMaker] keep_branches:\n  %s', ','.join(self.keep_branches))
        _logger.debug('[WeightMaker] load_branches:\n  %s', ','.join(self.load_branches))
        table = _read_files(filelist, self.load_branches, show_progressbar=True, treename=self._data_config.treename)
        _apply_selection(table, self._data_config.selection)
        _build_new_variables(table, {k: v for k, v in self._data_config.var_funcs.items() if k in self.keep_branches})
        _clean_up(table, self.load_branches - self.keep_branches)
        return table

    def make_weights(self, table):
        x_var, y_var = self._data_config.reweight_branches
        x_bins, y_bins = self._data_config.reweight_bins
        if not self._data_config.reweight_discard_under_overflow:
            # clip variables to be within bin ranges
            x_min, x_max = min(x_bins), max(x_bins)
            y_min, y_max = min(y_bins), max(y_bins)
            _logger.info(f'Clipping `{x_var}` to [{x_min}, {x_max}] to compute the shapes for reweighting.')
            _logger.info(f'Clipping `{y_var}` to [{y_min}, {y_max}] to compute the shapes for reweighting.')
            table[x_var] = np.clip(table[x_var], min(x_bins), max(x_bins))
            table[y_var] = np.clip(table[y_var], min(y_bins), max(y_bins))

        _logger.info('Using %d events to make weights', len(table[x_var]))

        sum_evts = 0
        max_weight = 0.9
        raw_hists = {}
        class_events = {}
        result = {}
        for label in self._data_config.reweight_classes:
            pos = (table[label] == 1)
            x = table[x_var][pos]
            y = table[y_var][pos]
            hist, _, _ = np.histogram2d(x, y, bins=self._data_config.reweight_bins)
            _logger.info('%s:\n %s', label, str(hist.astype('int64')))
            sum_evts += hist.sum()
            raw_hists[label] = hist.astype('float32')
            result[label] = hist.astype('float32')
        if sum_evts != len(table[x_var]):
            _logger.warning(
                'Only %d (out of %d) events actually used in the reweighting. '
                'Check consistency between `selection` and `reweight_classes` definition, or with the `reweight_vars` binnings '
                '(under- and overflow bins are discarded by default, unless `reweight_discard_under_overflow` is set to `False` in the `weights` section).',
                sum_evts, len(table[x_var]))
            time.sleep(10)

        if self._data_config.reweight_method == 'flat':
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                hist = result[label]
                threshold_ = np.median(hist[hist > 0]) * 0.01
                nonzero_vals = hist[hist > threshold_]
                min_val, med_val = np.min(nonzero_vals), np.median(hist)  # not really used
                ref_val = np.percentile(nonzero_vals, self._data_config.reweight_threshold)
                _logger.debug('label:%s, median=%f, min=%f, ref=%f, ref/min=%f' %
                              (label, med_val, min_val, ref_val, ref_val / min_val))
                # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
                wgt = np.clip(np.nan_to_num(ref_val / hist, posinf=0), 0, 1)
                result[label] = wgt
                # divide by classwgt here will effective increase the weight later
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
        elif self._data_config.reweight_method == 'ref':
            # use class 0 as the reference
            hist_ref = raw_hists[self._data_config.reweight_classes[0]]
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
                ratio = np.nan_to_num(hist_ref / result[label], posinf=0)
                upper = np.percentile(ratio[ratio > 0], 100 - self._data_config.reweight_threshold)
                wgt = np.clip(ratio / upper, 0, 1)  # -> [0,1]
                result[label] = wgt
                # divide by classwgt here will effective increase the weight later
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
        # ''equalize'' all classes
        # multiply by max_weight (<1) to add some randomness in the sampling
        min_nevt = min(class_events.values()) * max_weight
        for label in self._data_config.reweight_classes:
            class_wgt = float(min_nevt) / class_events[label]
            result[label] *= class_wgt

        _logger.info('weights:')
        for label in self._data_config.reweight_classes:
            _logger.info('%s:\n %s', label, str(result[label]))

        _logger.info('Raw hist * weights:')
        for label in self._data_config.reweight_classes:
            _logger.info('%s:\n %s', label, str((raw_hists[label] * result[label]).astype('int32')))

        return result

    def produce(self, output=None):
        table = self.read_file(self._filelist)
        wgts = self.make_weights(table)
        self._data_config.reweight_hists = wgts
        # must also propogate the changes to `data_config.options` so it can be persisted
        self._data_config.options['weights']['reweight_hists'] = {k: v.tolist() for k, v in wgts.items()}
        if output:
            _logger.info('Writing YAML file w/ reweighting info to %s' % output)
            self._data_config.dump(output)
        return self._data_config
