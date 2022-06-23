from collections.abc import Sequence

import torch
import numpy as np
from torch import Tensor
from .data import Data
from e3nn.o3 import Irreps


class Batch(Data):
    """
    A collection of tensors that can be indexed as (or initialized from) a list or a dict.
    """

    def __init__(self, attrs={}, **tensors):
        """
        This function is directly called only if initializing from a file.
        Call from_data_list if initializing from a list of data dicts.
        """
        super().__init__(attrs, **tensors)
        self.computeCumsums()

    def computeCumsums(self):
        self.computeSums()
        if "_n_nodes" in self.data:
            self.n_graphs = self.data["_n_nodes"].shape[0]
            self.node_cumsum = [0]
            for i in range(self.n_graphs):
                self.node_cumsum.append(self.node_cumsum[-1] + self.data["_n_nodes"][i])
        if "_n_edges" in self.data:
            self.n_graphs = self.data["_n_edges"].shape[0]
            self.edge_cumsum = [0]
            for i in range(self.n_graphs):
                self.edge_cumsum.append(self.edge_cumsum[-1] + self.data["_n_edges"][i])

    @classmethod
    def from_data_list(cls, lst, attrs={}):
        """
        This function can be used for creating datasets.
        It works with list of dict of tensors or numpy arrays,
        which will be properly shaped into (cat_dim, irreps_and_channels).
        """
        data = {}
        
        node_key = None
        edge_key = None
        for key in lst[0].keys():
            if key in attrs:
                if attrs[key][0] == "node":
                    node_key = key
                elif attrs[key][0] == "edge":
                    edge_key = key

        if node_key:
            attrs["_n_nodes"] = ("graph", "1x0e")
            data["_n_nodes"] = torch.zeros((len(lst),), dtype=torch.long)
            for i, item in enumerate(lst):
                data["_n_nodes"][i] = item[node_key].shape[0]

        if edge_key:
            attrs["_n_edges"] = ("graph", "1x0e")
            data["_n_edges"] = torch.zeros((len(lst),), dtype=torch.long)
            for i, item in enumerate(lst):
                data["_n_edges"][i] = item[edge_key].shape[0]
                
        for key in lst[0].keys():
            if key == "edge_index" or key == "face":
                to_cat = []
                cnt = 0
                for i, item in enumerate(lst):
                    if not isinstance(item[key], torch.Tensor):
                        item[key] = torch.Tensor(item[key])
                    to_cat.append(item[key] + cnt)
                    cnt += data['_n_nodes'][i]
                data[key] = torch.cat(to_cat, dim=-1)
            else:
                items = [
                    torch.tensor(item[key])
                    if not isinstance(item[key], torch.Tensor)
                    else item[key]
                    for item in lst
                ]
                if key in attrs:
                    irreps = attrs[key][1]
                    if (
                        isinstance(irreps, int)
                        or isinstance(irreps, str)
                        and irreps.isdigit()
                    ):
                        dim = int(irreps)
                    elif isinstance(irreps, str) or isinstance(irreps, Irreps):
                        dim = Irreps(irreps).dim
                    items = [item.reshape(-1, dim) for item in items]
                data[key] = torch.cat(items, dim=0)

        return cls(attrs, **data)

    def get(self, idx):
        dic = {}
        for key, value in self.data.items():
            if not key in self.attrs:
                continue
            if self.attrs[key][0] == "graph":
                start, end = idx, idx + 1
            else:
                if self.attrs[key][0] == "node":
                    cumsum = self.node_cumsum
                elif self.attrs[key][0] == "edge":
                    cumsum = self.edge_cumsum
                start, end = cumsum[idx], cumsum[idx + 1]
            if key == "edge_index":
                dic[key] = value[start:end] - cumsum[idx]
            else:
                dic[key] = value[start:end]
        return Data(self.attrs, **dic)

    def index_select(self, idx):
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            idx = idx.flatten().tolist()

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            idx = idx.flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            idx = idx.flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass

        else:
            raise IndexError(
                f"Only integers, slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        lst = [self.get(i) for i in idx]
        attrs = self.get(idx[0]).attrs
        result = Batch.from_data_list(lst, attrs)
        return result

    def nodeSegment(self):
        if not "_node_segment" in self.data:
            batch = []
            for i, n in enumerate(self["_n_nodes"]):
                batch += [i] * n
            batch = torch.tensor(batch)
            self.data["_node_segment"] = batch
        return self.data["_node_segment"]
      
    def edgeSegment(self):
        if not "_edge_segment" in self.data:
            batch = []
            for i, n in enumerate(self["_n_edges"]):
                batch += [i] * n
            batch = torch.tensor(batch)
            self.data["_edge_segment"] = batch
        return self.data["_edge_segment"]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.data[idx]
        elif isinstance(idx, (int, np.integer)):
            return self.get(idx)
        else:
            return self.index_select(idx)

    def __setitem__(self, idx, item):
        if isinstance(idx, int):
            raise UnimplementedError(
                "Setting item with an integer index is not supported for Batch."
            )
        else:
            self.data[idx] = item
            if idx in ('_n_nodes', '_n_edges'):
                self.computeCumsums()

    def update(self, other):
        for key, value in other.items():
            self[key] = value
        if '_n_nodes' in other or '_n_edges' in other:
            self.computeCumsums()

    def __len__(self):
        return self.n_graphs
