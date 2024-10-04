from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import math
from typing import Callable, Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.table import Table
from rich.highlighter import ReprHighlighter
from rich import box
from tabulate import tabulate

class RemoveSelfLoops(BaseTransform):
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.edge_index, _ = remove_self_loops(data.edge_index)
        if hasattr(data, 'adj_t'):
            data.adj_t = data.adj_t.remove_diag()
        return data

class RemoveIsolatedNodes(BaseTransform):
    def __call__(self, data: Data) -> Data:
        mask = data.y.new_zeros(data.num_nodes, dtype=bool)
        mask[data.edge_index[0]] = True
        mask[data.edge_index[1]] = True
        data = data.subgraph(mask)
        return data

def dict2table(input_dict: dict, num_cols: int = 4, title: Optional[str] = None) -> Table:
    num_items = len(input_dict)
    num_rows = math.ceil(num_items / num_cols)
    col = 0
    data = {}
    keys = []
    vals = []

    for i, (key, val) in enumerate(input_dict.items()):
        keys.append(f'{key}:')

        vals.append(val)
        if (i + 1) % num_rows == 0:
            data[col] = keys
            data[col+1] = vals
            keys = []
            vals = []
            col += 2

    data[col] = keys
    data[col+1] = vals

    highlighter = ReprHighlighter()
    message = tabulate(data, tablefmt='plain')
    table = Table(title=title, show_header=False, box=box.HORIZONTALS)
    table.add_row(highlighter(message))
    return table

