import numpy as np

import ase

import torch
from torch_runstats.scatter import scatter_std, scatter_mean

from .batch import Batch
from ..utils import bincount, solver, keyMap
import h5py
from tqdm import trange

import numpy as np

import torch.utils.data
from e3nn import o3
import os
import re
import logging

from inspect import signature

class CondensedDataset(Batch):
    """
    A e3_layers.data.Batch with preprocessing, statistics and key mapping.
    """

    def __init__(self, path=None, data={}, attrs={}, key_map={}, type_names=None, preprocess=[], **kwargs):
        """
        If path is provided, loads the dataset from file(s). The path parameter may specify a file, a list of files, a directory, or a directory and a regular_expression separated by ':'.
        """
        if not path is None:
            data, attrs = CondensedDataset.load(path)
            if isinstance(data, list):
                data = Batch.from_data_list(data, attrs).data
        super().__init__(attrs, **data)
        self.data = keyMap(self.data, key_map)
        self.attrs = keyMap(self.attrs, key_map)
        self.attrs = {key:(value[0], value[1]) for key, value in self.attrs.items()}
        # converts from object array to tuple
             
        if type_names == None:
            type_names = ase.atom.atomic_numbers.keys()
        self.type_names = list(type_names)
        self.preprocess = preprocess
        self.kwargs = kwargs
        
    @staticmethod
    def load(path):
        def loadFile(file):
            logging.info(f'Loading {file}')
            data = {}
            attrs = {}
            with h5py.File(file, "r") as file:
                for key in file.keys():
                    item = torch.tensor(file[key][:])
                    if item.dtype == torch.int32:
                        item = item.long()
                    elif item.dtype == torch.float64:
                        item = item.float()
                    data[key] = item
                for key in file.attrs.keys():
                    attrs[key] = file.attrs[key]
                logging.info(f'Loaded {file}')
            return data, attrs
          
        if isinstance(path, str):
            path = path.split(':')
            if len(path) == 2:
                path, regexp = path
                regexp = re.compile(regexp)
            else:
                path = path[0]
                regexp = None

            if os.path.isdir(path):
                data = []
                attrs = {}
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file = os.path.join(root, file)
                        if not regexp is None and regexp.match(file) is None:
                            continue
                        _data, _attrs = loadFile(file)
                        data.append(_data)
                        attrs.update(_attrs)
            else:
                data, attrs = loadFile(path)
        else: # is a list
            data = []
            attrs = {}
            for item in path:
                x, y = CondensedDataset.load(item)
                if isinstance(x, list):
                    data += x
                else:
                    data.append(x)
                attrs.update(y)
                
        if len(data) == 0:
            logging.warning(f'No dataset file is found in {path}.')
        return data, attrs
            
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.data[idx]
        elif isinstance(idx, (int, np.integer)):
            data = self.get(idx).clone()
            for func in self.preprocess:
                sig = signature(func)
                if len(sig.parameters) == 1:
                    data = func(data)
                else:
                    tensors, attrs = data.data, data.attrs
                    tensors, attrs = func(tensors, attrs)
                    data.data, data.attrs = tensors, attrs
            return data
        else:
            dataset = self.index_select(idx)
            return dataset

    def equivarianceTest(self, size, idx=0):
        matrices = o3.rand_matrix(size)
        self.length = size
        self.data["_rotation_matrix"] = matrices
        for i in range(size):
            for key, value in self.data.items():
                if key in self.attrs:
                    transform = self.attrs[key][1]
                    if isinstance(transform, str) or isinstance(transform, o3.Irreps):
                        mat = transform.D_from_matrix(matrices[i])
                        self.data[key][i] = (
                            mat @ value[idx].transpose(-1, -2)
                        ).transpose(-1, -2)
                    else:
                        self.data[key][i] = transform(matrices[i], value[idx])

    def statistics(
        self,
        fields,
        stride: int = 1,
        unbiased: bool = True,
    ):
        n_samples = len(self) // stride
        pbar = trange(n_samples)
        pbar.set_description("loading samples for computing statistics")
        lst = [self[i * stride] for i in pbar]
        data_transformed = Batch.from_data_list(lst)
        out: list = []
        for field in fields:
            key = field.split("-")[0]
            ana_mode = field[len(key) + 1 :]
            arr = data_transformed[key]
            is_per = self.attrs[key][0]
            if not isinstance(arr, torch.Tensor):
                if np.issubdtype(arr.dtype, np.floating):
                    arr = torch.as_tensor(arr, dtype=torch.get_default_dtype())
                else:
                    arr = torch.as_tensor(arr)

            # compute statistics
            if ana_mode == "count":
                # count integers
                uniq, counts = torch.unique(
                    torch.flatten(arr), return_counts=True, sorted=True
                )
                out.append((uniq, counts))
            elif ana_mode == "rms":
                # root-mean-square
                out.append((torch.sqrt(torch.mean(arr * arr)),))

            elif ana_mode == "mean_std":
                # mean and std
                mean = torch.mean(arr, dim=0)
                std = torch.std(arr, dim=0, unbiased=unbiased)
                out.append((mean, std))

            elif ana_mode.startswith("per-node-"):
                # per-atom
                # only makes sense for a per-graph quantity
                if is_per != "graph":
                    raise ValueError(
                        f"It doesn't make sense to ask for `{ana_mode}` since `{field}` is not per-graph"
                    )
                ana_mode = ana_mode[len("per-node-") :]
                results = self._per_node_statistics(
                    ana_mode=ana_mode,
                    arr=arr,
                    batch=data_transformed.nodeSegment(),
                    unbiased=unbiased,
                )
                out.append(results)

            elif ana_mode.startswith("per-"):
                _, key, ana_mode = ana_mode.split("-")
                # per-species

                atom_types = data_transformed[key]
                results = self._per_species_statistics(
                    ana_mode,
                    arr,
                    is_per=is_per,
                    batch=data_transformed.nodeSegment(),
                    atom_types=atom_types,
                    unbiased=unbiased,
                )
                out.append(results)

            else:
                raise NotImplementedError(f"Cannot handle statistics mode {ana_mode}")

        return out

    def index_select(self, idx):
        batch = super().index_select(idx)
        dataset = CondensedDataset(type_names=self.type_names,
                                   preprocess=self.preprocess, data=batch.data, attrs=batch.attrs)
        return dataset

    def _per_node_statistics(
        self,
        ana_mode: str,
        arr: torch.Tensor,
        batch: torch.Tensor,
        unbiased: bool = True,
    ):
        """Compute "per-atom" statistics that are normalized by the number of atoms in the system.

        Only makes sense for a graph-level quantity
        """
        # using unique_consecutive handles the non-contiguous selected batch index
        _, N = torch.unique_consecutive(batch, return_counts=True)
        N = N.unsqueeze(-1)
        assert N.ndim == 2
        assert N.shape == (len(arr), 1)
        assert arr.ndim >= 2
        data_dim = arr.shape[1:]
        arr = arr / N
        assert arr.shape == (len(N),) + data_dim
        if ana_mode == "mean_std":
            mean = torch.mean(arr, dim=0)
            std = torch.std(arr, unbiased=unbiased, dim=0)
            return mean, std
        elif ana_mode == "rms":
            return (torch.sqrt(torch.mean(arr.square())),)
        else:
            raise NotImplementedError(
                f"{ana_mode} for per-atom analysis is not implemented"
            )

    def _per_species_statistics(
        self,
        ana_mode: str,
        arr: torch.Tensor,
        is_per: str,
        atom_types: torch.Tensor,
        batch: torch.Tensor,
        unbiased: bool = True,
    ):
        """Compute "per-species" statistics.

        For a graph-level quantity, models it as a linear combintation of the number of atoms of different types in the graph.

        For a per-node quantity, computes the expected statistic but for each type instead of over all nodes.
        """
        N = bincount(atom_types.squeeze(-1), batch, minlength=len(self.type_names))
        assert N.ndim == 2  # [batch, n_type]
        N = N[(N > 0).any(dim=1)]  # deal with non-contiguous batch indexes
        assert arr.ndim >= 2
        if is_per == "graph":

            if ana_mode != "mean_std":
                raise NotImplementedError(
                    f"{ana_mode} for per species analysis is not implemented for shape {arr.shape}"
                )

            N = N.type(torch.get_default_dtype())

            return solver(N, arr)

        elif is_per == "node":
            arr = arr.type(torch.get_default_dtype())

            if ana_mode == "mean_std":
                mean = scatter_mean(arr, atom_types, dim=0)
                assert mean.shape[1:] == arr.shape[1:]  # [N, dims] -> [type, dims]
                assert len(mean) == N.shape[1]
                std = scatter_std(arr, atom_types, dim=0, unbiased=unbiased)
                assert std.shape == mean.shape
                return mean, std
            elif ana_mode == "rms":
                square = scatter_mean(arr.square(), atom_types, dim=0)
                assert square.shape[1:] == arr.shape[1:]  # [N, dims] -> [type, dims]
                assert len(square) == N.shape[1]
                dims = len(square.shape) - 1
                for i in range(dims):
                    square = square.mean(axis=-1)
                return (torch.sqrt(square),)

        else:
            raise NotImplementedError
