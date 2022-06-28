"""AtomicData: neighbor graphs in (periodic) real space.

Authors: Albert Musaelian
Modified by Hangrui Bi
"""

from typing import Tuple, Union


import torch
import e3nn.o3

_TORCH_INTEGER_DTYPES = (torch.int, torch.long)

# A type representing ASE-style periodic boundary condtions, which can be partial (the tuple case)
PBC = Union[bool, Tuple[bool, bool, bool]]

import re
import copy

import torch
import h5py

__num_nodes_warn_msg__ = (
    "The number of nodes in your data object can only be inferred by its {} "
    "indices, and hence may result in unexpected batch-wise behavior, e.g., "
    "in case there exists isolated nodes. Please consider explicitly setting "
    "the number of nodes for this data object by assigning it to "
    "data.num_nodes."
)


class Data(object):
    def __init__(
        self,
        attrs={},
        **tensors,
    ):
        """
        Do not inherit dict or torch pin_memory will not be overriden.
        It is recommended to describe each tensor if possible, e.g. attrs['position'] = ('node', '1x1o').
        A described tensor helps to infer the number of node and edges, to perform equivariance tests, to reshape the tensor, and to select one item in a batch.
        The later term should be a string describing the irreps or an int for the number of dimensions.
        Each tensor should be of shape (cat_dim, irreps_and_channels).
        Keys that do not start with an underscore are normal ones that appears in the data list.
        Those start with an underscore describes the auxillary tensors, such as '_n_nodes' and '_n_edges'.
        Both of them are scalars defined per graph, so the attrs for these keys are ('graph', '1x0e').
        """
        self.attrs = attrs
        self.data = tensors
        for key, value in tensors.items():
            self[key] = value

    def computeSums(self):
        node_key = None
        edge_key = None
        graph_key = None
        attrs = self.attrs
        keys = list(iter(self.keys()))
        for key in keys:
            if not key in attrs:
                continue
            if attrs[key][0] == "node":
                node_key = key
            elif attrs[key][0] == "edge":
                edge_key = key
            elif attrs[key][0] == "graph":
                graph_key = key
        if node_key:
            self.n_nodes = self.data[node_key].shape[self.__cat_dim__(key)]
        if edge_key:
            self.n_edges = self.data[edge_key].shape[self.__cat_dim__(key)]
        if graph_key:
            self.n_graphs = self.data[graph_key].shape[self.__cat_dim__(key)]

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys())

    def __getitem__(self, idx):
        return self.data[idx]

    def keys(self):
        return self.data.keys()

    def __contains__(self, key):
        return key in self.data

    def items(self):
        return [(key, value) for key, value in self.data.items()]

    def __setitem__(self, key, item):
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item)
        if key in self.attrs:
            irreps = self.attrs[key][1]
            if (
                isinstance(irreps, int)
                or isinstance(irreps, str)
                and irreps.isdigit()
            ):
                dim = int(irreps)
            elif isinstance(irreps, str) or isinstance(irreps, e3nn.o3.Irreps):
                dim = e3nn.o3.Irreps(irreps).dim
            else:
                dim = None
            if not dim is None:
                if not (len(item.shape)==2 and item.shape[-1] == dim):
                    item = item.view(-1, dim)
              
        self.data[key] = item
        if key == "_n_nodes" or key == "_n_edges":
            self.computeSums()

    def update(self, other):
        for key, value in other.items():
            self[key] = value
        if "_n_nodes" in other or "_n_edges" in other:
            self.computeSums()

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys()) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        if bool(re.search("(index|face)", key)):
            return -1
        return 0

    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph."""
        return self.attrs["__n_nodes"]

    @property
    def num_edges(self):
        """
        Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges.
        """
        return self["edge_index"].shape[-1]

    @property
    def num_faces(self):
        r"""Returns the number of faces in the mesh."""
        return self["face"].shape[-1]

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key in keys:
            self.data[key] = self.__apply__(self.data[key], func)
        return self

    def contiguous(self):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *self.keys())

    def to(self, device, **kwargs):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device, **kwargs), *self.keys())

    def cpu(self):
        r"""Copies all attributes :obj:`*keys` to CPU memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.cpu(), *self.keys())

    def cuda(self, device=None, non_blocking=False):
        r"""Copies all attributes :obj:`*keys` to CUDA memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(
            lambda x: x.cuda(device=device, non_blocking=non_blocking), *self.keys()
        )

    def clone(self):
        r"""Performs a deep-copy of the data object."""
        return self.__class__(
            copy.deepcopy(self.attrs),
            **{
                k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in self.data.items()
            },
        )

    def pin_memory(self):
        r"""Copies all attributes :obj:`*keys` to pinned memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.pin_memory(), *self.keys())

    def __repr__(self):
        attrs = self.attrs
        data = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                data[key] = (value.shape, value.dtype)
            else:
                data[key] = type(value)
        return f"attrs:{attrs}\n tensors:{data}"

    def dumpHDF5(self, path):
        with h5py.File(path, "w") as f:
            for key in self.keys():
                print(f"Writing tensor {key}.")
                f[key] = self[key].to('cpu')

            for key in self.attrs.keys():
                print(f"Writing attr {key}={self.attrs[key]}")
                f.attrs[key] = self.attrs[key]
