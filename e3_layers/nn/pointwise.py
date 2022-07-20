import torch
import torch.nn.functional
from e3nn.o3 import Linear, Irreps, TensorProduct

from .sequential import Module
from ..utils import activations
from e3nn.nn import NormActivation

from torch import Tensor
from e3nn.util.jit import compile_mode
from typing import Optional, Dict, Tuple


class PointwiseLinear(Module):
    def __init__(self, irreps_in, irreps_out, biases=True, **kwargs):
        super().__init__()
        self.init_irreps(input=irreps_in, output=irreps_out, output_keys=["output"])
        self.linear = Linear(
            irreps_in=self.irreps_in["input"],
            irreps_out=self.irreps_out["output"],
            biases=biases,
        )

    def forward(self, data: Dict[str, Tensor], attrs:Dict[str, Tuple[str, str]]):
        input = data["input"]
        output = self.linear(input)

        attrs = {"output": (attrs["input"][0], self.irreps_out["output"])}
        data = {"output": output}
        return data, attrs
      
class LayerNormalization(Module):
    def __init__(self, irreps_in, irreps_out, **kwargs):
        super().__init__()
        self.init_irreps(input=irreps_in, output=irreps_out, output_keys=["output"])
        assert irreps_in == irreps_out
        self.muls = [irrep.mul for irrep in Irreps(irreps_in)]
        self.slices = [(slice.start, slice.stop) for slice in Irreps(irreps_in).slices()]
        self.std = torch.nn.Parameter(torch.ones(len(self.slices)))

    def forward(self, data: Dict[str, Tensor], attrs:Dict[str, Tuple[str, str]]):
        input = data["input"]
        output = torch.zeros(input.shape).to(input.device)
        for i, (start, end) in enumerate(self.slices):
            tmp = input[:, start:end]
            norm = (tmp*tmp).sum(dim=-1, keepdim=True)
            norm = (norm/self.muls[i] + 1e-6)**0.5 
            tmp = tmp/norm
            output[:, start:end] = tmp * self.std[i]
        data = {"output": output}
        return data, attrs


class TensorProductExpansion(Module):
    def __init__(
        self, left, right, output, instruction="uvu", internal_weight=True, **kwargs
    ):
        super().__init__()
        self.init_irreps(left=left, right=right, output=output, output_keys=["output"])

        irreps_mid = []
        instructions = []
        for i, (mul, left) in enumerate(Irreps(self.irreps_in["left"])):
            for j, (_, right) in enumerate(Irreps(self.irreps_in["right"])):
                for ir_out in left * right:
                    if ir_out in self.irreps_out["output"]:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, instruction, True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            Irreps(self.irreps_in["left"]),
            Irreps(self.irreps_in["right"]),
            irreps_mid,
            instructions,
            shared_weights=internal_weight,
            internal_weights=internal_weight,
        )
        self.internal_weight = internal_weight
        self.linear = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=self.irreps_out["output"],
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, left=None, right=None, weight=None):
        if self.internal_weight:
            output = self.tp(left, right)
        else:
            output = self.tp(left, right, weight)
        output = self.linear(output)
        return output


class ResBlock(Module):
    def __init__(self, irreps_in, irreps_out, activation="silu", biases=True, **kwargs):
        super().__init__()
        self.init_irreps(input=irreps_in, output=irreps_out, output_keys=["output"])
        self.linear_1 = Linear(irreps_in=irreps_in, irreps_out=irreps_in, biases=biases)
        if not irreps_in == irreps_out:
            self.linear_2 = Linear(
                irreps_in=irreps_in, irreps_out=irreps_out, biases=biases
            )
        self.act = NormActivation(irreps_in, activations[activation])

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            input = data
        else:
            input = self.inputKeyMap(data)["input"]
        output = self.linear_1(self.act(input))
        output = input + output
        if not self.irreps_in["input"] == self.irreps_out["output"]:
            output = self.linear_2(output)

        if isinstance(data, torch.Tensor):
            return output
        else:
            is_per = self.inputKeyMap(data.attrs)["input"][0]
            data.attrs.update(
                self.outputKeyMap({"output": (is_per, self.irreps_out["output"])})
            )
            data.update(self.outputKeyMap({"output": output}))
            return data

class Concat(Module):
    def __init__(self, irreps_out, **irreps_in):
        super().__init__()
        self.init_irreps(**irreps_in, output=irreps_out, output_keys=["output"])
        lst = [Irreps(value) for value in self.irreps_in.values()]
        irreps_in = lst[0]
        for i in range(1, len(lst)):
            irreps_in += lst[i]
        self.linear = Linear(irreps_in=irreps_in, irreps_out=Irreps(self.irreps_out['output']), biases=True)
        
    def forward(self, data: Dict[str, Tensor], attrs:Dict[str, Tuple[str, str]]):
        input = torch.cat([data[key] for key in self.irreps_in.keys()], dim = 1)
        output = self.linear(input)

        key = list(self.irreps_in.keys())[0]
        is_per = attrs[key][0]
        attrs = {"output": (is_per, self.irreps_out["output"])}
        data = {"output": output}
        return data, attrs

class Split(Module):
    def __init__(self, irreps_in, **irreps_out):
        super().__init__()
        self.init_irreps(input = irreps_in, **irreps_out, output_keys=[key for key in irreps_out])
        lst = [Irreps(value) for value in self.irreps_out.values()]
        irreps_out = lst[0]
        for i in range(1, len(lst)):
            irreps_out += lst[i]
        self.linear = Linear(irreps_in=Irreps(self.irreps_in['input']), irreps_out=irreps_out, biases=True)
        
    def forward(self, data: Dict[str, Tensor], attrs:Dict[str, Tuple[str, str]]):
        input = data['input']
        result = self.linear(input)
        output = {}
        cnt = 0
        for key, value in self.irreps_out.items():
            inc = value.dim
            output[key] = result[cnt:cnt+inc]
            cnt += inc
        is_per = attrs['input'][0]
        attrs = {key: (is_per, value) for key in self.irreps_out.items()}
        return output, attrs