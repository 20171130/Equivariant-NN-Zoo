from ml_collections.config_dict import ConfigDict
import inspect
from e3nn import o3
import torch

import math

def insertAfter(lst, key, item):
    for i, layer in enumerate(lst):
        if layer[0] == key:
            lst = lst[:i+1] + [item] + lst[i+1:]
            return lst
    raise ValueError(f'Key {key} not found.')
    
def replace(lst, key, item):
    for i, layer in enumerate(lst):
        if layer[0] == key:
            lst = lst[:i] + [item] + lst[i+1:]
            return lst
    raise ValueError(f'Key {key} not found.')

def transformMatrix(irreps_l, irreps_r, mat, H):
    D_l = irreps_l.D_from_matrix(mat)
    D_r = irreps_r.D_from_matrix(mat)
    return D_l @ H @ D_r.transpose(-1, -2)


def tanhlu(x):
    return torch.tanh(x) * torch.abs(x)


@torch.jit.script
def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)


activations = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
    "tanhlu": tanhlu,
}


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def build(node, **kwargs):
    """Instantiates a layer using its config node"""
    if isinstance(node, dict) or isinstance(node, ConfigDict):
        func = node["module"]
        kwargs.update(**node)

    elif isinstance(node, list) or isinstance(node, tuple):
        func = node[0]
        args = node[1:]

    else:
        func = node

    args = []
    if "module" in kwargs:
        kwargs.pop("module")
    kwargs = pruneArgs(func, **kwargs)
    return func(*args, **kwargs)


def pruneArgs(_func=None, prefix="", **kwargs):
    if not prefix == "":
        args = {}
        for key, value in kwargs.items():
            if key.startswith(prefix):
                key = key[len(prefix) + 1 :]
                args[key] = value
    else:
        args = kwargs

    if not _func is None:
        arg_spec = inspect.getfullargspec(_func)
        if arg_spec.varkw:
            return args
        else:
            pnames = inspect.signature(_func).parameters
            return {key: args[key] for key in args if key in pnames}
    return args


def keyMap(dic, key_mapping):
    if isinstance(dic, dict):
        result = {}
        for key, value in dic.items():
            if key in key_mapping:
                new_key = key_mapping[key]
                if isinstance(new_key, str):
                    result[new_key] = value
                else:
                    for item in new_key:
                        result[item] = value
            else:
                result[key] = value
        return result
    else:  # an instance of data.Data
        attrs = keyMap(dic.attrs, key_mapping)
        data = keyMap(dic.data, key_mapping)
        return type(dic)(attrs, **data)


def _countParameters(module):
    return sum([param.numel() for param in module.parameters() if param.requires_grad])


def countParameters(model):
    for name, module in model.named_modules():
        cnt = _countParameters(module)
        if cnt > 0:
            print(name, cnt)
