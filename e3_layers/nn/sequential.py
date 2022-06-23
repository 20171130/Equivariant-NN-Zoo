from collections import OrderedDict
from ml_collections.config_dict import ConfigDict

import torch

from e3nn.o3 import Irreps
from ..data import Data
from ..utils import build, keyMap
from torch.profiler import record_function


class Module(torch.nn.Module):
    def init_irreps(self, output_keys=[], **kwargs):
        if isinstance(output_keys, str):
            output_keys = [output_keys]
        self.irreps_in = {}
        self.irreps_out = {}
        self.input_key_mapping = {}
        self.output_key_mapping = {}
        for key, value in kwargs.items():
            if isinstance(value, str) or isinstance(value, Irreps):
                irreps, custom_keys = value, key
            elif isinstance(value, list) or isinstance(value, tuple):
                irreps, custom_keys = value[0], value[1:]
            if key in output_keys:
                self.irreps_out[key] = Irreps(irreps)
                self.output_key_mapping[key] = custom_keys
            else:
                self.irreps_in[key] = Irreps(irreps)
                for custom_key in custom_keys:
                    self.input_key_mapping[custom_key] = key

    def inputKeyMap(self, input):
        return keyMap(input, self.input_key_mapping)

    def outputKeyMap(self, output):
        return keyMap(output, self.output_key_mapping)


class SequentialGraphNetwork(torch.nn.Sequential):
    r"""
    layers can be a callable, besides instance of nn.Module
    """

    def __init__(self, **config):
        layer_configs = OrderedDict(config["layers"])
        self.layers = []
        self.modules = {}
        for i, (key, value) in enumerate(layer_configs.items()):
            if isinstance(value, ConfigDict) or isinstance(value, dict):
                module = build(value, **config)
                self.modules[key] = module
                self.layers += [(key, module)]
            elif callable(value):
                self.layers += [(key, value)]
            else:
                raise TypeError("invalid config node")

        modules = OrderedDict(self.modules)
        super().__init__(modules)

    def forward(self, batch):
        for i, (key, module) in enumerate(self.layers):
            with record_function(key):
                batch = module(batch)
            assert isinstance(
                batch, Data
            ), f"The return of {module} is not an instance of Data"
        return batch
