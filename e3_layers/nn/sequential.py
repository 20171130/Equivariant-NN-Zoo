from collections import OrderedDict
from ml_collections.config_dict import ConfigDict

import torch

from e3nn.o3 import Irreps
from ..data import Data, Batch
from ..utils import build, keyMap
from torch.profiler import record_function
from e3nn.util.jit import script, trace

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
                irreps, custom_key = value, key
            elif isinstance(value, list) or isinstance(value, tuple):
                assert len(value)==2
                irreps, custom_key = value[0], value[1]
            else: # value is None
                continue 
            if key in output_keys:
                self.irreps_out[key] = irreps
                self.output_key_mapping[key] = custom_key
            else:
                self.irreps_in[key] = irreps
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
        layer_configs = config["layers"]
        self.layers = []
        self.layer_configs = layer_configs
        modules = {}
        for i, (key, value) in enumerate(layer_configs):
            if isinstance(value, ConfigDict) or isinstance(value, dict):
                module = build(value)
                modules[key] = module
                if 'jit' in config and config['jit']:
                    module = (module, script(module))
                else:
                    modules[key] = module
                self.layers += [(key, module)]
            elif callable(value):
                if 'jit' in config and config['jit']:
                    value = torch.jit.script(value)
                self.layers += [(key, value)]
            else:
                raise TypeError("invalid config node")

        modules = OrderedDict(modules)
        super().__init__(modules)

    def forward(self, batch):
        batch.nodeSegment()
        data, attrs = batch.data, batch.attrs
        for i, (key, module) in enumerate(self.layers):
            with record_function(key):
                _data, _attrs = data, attrs
                if isinstance(module, tuple):
                    module, forward = module
                else:
                    forward = module
                if isinstance(module, Module):
                    _data = module.inputKeyMap(_data)
                    _attrs = module.inputKeyMap(_attrs)
                _data, _attrs = forward(_data, _attrs)
                if isinstance(module, Module):
                    _data = module.outputKeyMap(_data)
                    _attrs = module.outputKeyMap(_attrs)
                data.update(_data)
                attrs.update(_attrs)
        return Batch(attrs, **data)
