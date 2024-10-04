import numpy as np
import math

try:
    import awkward0 as awkward
except ImportError:
    import awkward
    if awkward.__version__[0] == '1':
        raise ImportError('Please install awkward0 with `pip install awkward0`.')


def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return awkward.concatenate(arrays, axis=axis)


def _stack(arrays, axis=1):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.stack(arrays, axis=axis)
    else:
        content = np.stack([a.content for a in arrays], axis=axis)
        return awkward.JaggedArray.fromcounts(arrays[0].counts, content)


def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, awkward.JaggedArray):
        return a.pad(maxlen, clip=True).fillna(value).regular().astype(dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x


def _repeat_pad(a, maxlen, shuffle=False, dtype='float32'):
    x = a.flatten()
    x = np.tile(x, int(np.ceil(len(a) * maxlen / len(x))))
    if shuffle:
        np.random.shuffle(x)
    x = x[:len(a) * maxlen].reshape((len(a), maxlen))
    mask = _pad(awkward.JaggedArray.zeros_like(a), maxlen, value=1)
    x = _pad(a, maxlen) + mask * x
    return x.astype(dtype)


def _clip(a, a_min, a_max):
    if isinstance(a, np.ndarray):
        return np.clip(a, a_min, a_max)
    else:
        return awkward.JaggedArray.fromcounts(a.counts, np.clip(a.content, a_min, a_max))


def _knn(support, query, k, n_jobs=1):
    from scipy.spatial import cKDTree
    kdtree = cKDTree(support)
    d, idx = kdtree.query(query, k, n_jobs=n_jobs)
    return idx


def _batch_knn(supports, queries, k, maxlen_s, maxlen_q=None, n_jobs=1):
    assert (len(supports) == len(queries))
    if maxlen_q is None:
        maxlen_q = maxlen_s
    batch_knn_idx = np.ones((len(supports), maxlen_q, k), dtype='int32') * (maxlen_s - 1)
    for i, (s, q) in enumerate(zip(supports, queries)):
        batch_knn_idx[i, :len(q[:maxlen_q]), :] = _knn(
            s[:maxlen_s], q[:maxlen_q], k, n_jobs=n_jobs).reshape((-1, k))  # (len(q), k)
    return batch_knn_idx


def _batch_permute_indices(array, maxlen):
    batch_permute_idx = np.tile(np.arange(maxlen), (len(array), 1))
    for i, a in enumerate(array):
        batch_permute_idx[i, :len(a)] = np.random.permutation(len(a[:maxlen]))
    return batch_permute_idx


def _batch_argsort(array, maxlen):
    batch_argsort_idx = np.tile(np.arange(maxlen), (len(array), 1))
    for i, a in enumerate(array):
        batch_argsort_idx[i, :len(a)] = np.argsort(a[:maxlen])
    return batch_argsort_idx


def _batch_gather(array, indices):
    out = array.zeros_like()
    for i, (a, idx) in enumerate(zip(array, indices)):
        maxlen = min(len(a), len(idx))
        out[i][:maxlen] = a[idx[:maxlen]]
    return out


def _get_variable_names(expr, exclude=['awkward', 'np', 'numpy', 'math']):
    import ast
    root = ast.parse(expr)
    return sorted({node.id for node in ast.walk(root) if isinstance(
        node, ast.Name) and not node.id.startswith('_')} - set(exclude))


def _eval_expr(expr, table):
    tmp = {k: table[k] for k in _get_variable_names(expr)}
    tmp.update(
        {'math': math, 'np': np, 'awkward': awkward, '_concat': _concat, '_stack': _stack, '_pad': _pad,
         '_repeat_pad': _repeat_pad, '_clip': _clip, '_batch_knn': _batch_knn,
         '_batch_permute_indices': _batch_permute_indices, '_batch_argsort': _batch_argsort,
         '_batch_gather': _batch_gather})
    return eval(expr, tmp)
