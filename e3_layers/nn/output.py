import torch

from e3nn.o3 import Irreps, wigner_3j
from e3nn.util.jit import compile_mode

from .sequential import Module
from ml_collections.config_dict import ConfigDict
from ..utils import build, tp_path_exists
from .message_passing import FactorizedConvolution
from .pointwise import PointwiseLinear, ResBlock, TensorProductExpansion
from torch_runstats.scatter import scatter
from copy import copy

from torch import Tensor
from e3nn.util.jit import compile_mode
from typing import Optional, Dict, Tuple

@compile_mode("script")
class GradientOutput(Module):
    def __init__(self, func, x, y, gradients, sign: float = 1.0, **kwargs):
        super().__init__()
        sign = float(sign)
        assert sign in (1.0, -1.0)
        self.sign = sign
        self.init_irreps(x=x, y=y, gradients=gradients, output_keys=['gradients'])
        assert Irreps(self.irreps_in["y"]).lmax == 0
        if isinstance(func, dict) or isinstance(func, ConfigDict):
            func = build(func, **kwargs)
        self.func = func

    def forward(self, data):
        input = self.inputKeyMap(data)
        wrt_tensor = input["x"]

        old_requires_grad = wrt_tensor.requires_grad
        wrt_tensor.requires_grad_(True)

        output = self.func(data)
        grad = torch.autograd.grad(
            self.inputKeyMap(output)["y"].sum(),
            wrt_tensor,
            create_graph=self.training  # needed to allow gradients of this output during training
        )

        grad = self.sign * grad[0]
        wrt_tensor.requires_grad_(old_requires_grad)

        is_per = self.inputKeyMap(data.attrs)["x"][0]
        data.attrs.update(
            self.outputKeyMap({"gradients": (is_per, self.irreps_out["gradients"])})
        )
        output.update(self.outputKeyMap({"gradients": grad}))
        return output


class Pooling(Module, torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, reduce):
        """
        Currently, only supports pooling node features into graph features
        """
        super().__init__()
        self.init_irreps(input=irreps_in, output=irreps_out, output_keys=["output"])
        self.reduce = reduce
        assert reduce in ("sum", "mean")

    def forward(self, data: Dict[str, Tensor], attrs:Dict[str, Tuple[str, str]]):
        input = data['input']
        batch = data['_node_segment'].to(input.device)
        output = scatter(input, batch, dim=0, reduce=self.reduce)

        is_per = "graph"
        attrs = {"output": (is_per, self.irreps_out["output"])}
        data = {"output": output}
        return data, attrs


class Pairwise(Module, torch.nn.Module):
    def __init__(
        self,
        node_features,
        edge_radial,
        edge_spherical,
        diagonal,
        off_diagonal,
        invariant_layers=2,
        invariant_neurons=16,
        conv=None,
    ):
        """
        This module constructs pairwise features using node features and their relative positions.
        Roughly speaking (linear and residual blocks ommited),
        f_ii = f_i + tensor_product(f_i, f_i)
        f_ij = f_i + tensor_product(f_i, messagePassing(f_j, Ylm(r_ij)))
        The message passing module does not reduce the messages for each destination node, and returns results for every edge instead of every node. It is essentially a tensor product of f_j and Ylm.
        Notice that the tensor products are decomposed into irreducible representations.
        """
        super().__init__()
        self.init_irreps(
            node_features=node_features,
            edge_radial=edge_radial,
            edge_spherical=edge_spherical,
            diagonal=diagonal,
            off_diagonal=off_diagonal,
            output_keys=["diagonal", "off_diagonal"],
        )
        irreps_in = self.irreps_in["node_features"]
        if conv == "auto":
            dic = self.inputKeyMap(self.irreps_in)
            dic["input_features"] = (dic.pop("node_features"), "node_features")
            dic["node_attrs"] = None
            self.conv = FactorizedConvolution(
                output_features=irreps_in,
                invariant_layers=2,
                invariant_neurons=32,
                avg_num_neighbors=1,
                use_sc=False,
                reduce=False,
                **dic,
            )
        else:
            self.conv = None

        irreps_out = self.irreps_out["diagonal"]
        self.tp = TensorProductExpansion(irreps_in, irreps_in, irreps_out, "uvu")
        self.res_center = ResBlock(irreps_in, irreps_in)
        self.res_pair = ResBlock(irreps_out, irreps_out)
        self.res_res = ResBlock(irreps_in, irreps_out)

        self.tp_off = TensorProductExpansion(irreps_in, irreps_in, irreps_out, "uvu")
        self.res_center_off = ResBlock(irreps_in, irreps_in)
        self.res_pair_off = ResBlock(irreps_out, irreps_out)
        self.res_res_off = ResBlock(irreps_in, irreps_out)

    def forward(self, data):
        input = self.inputKeyMap(data)
        node_features = input["node_features"]
        input["edge_radial"]
        edge_index = input["edge_index"]
        input["edge_spherical"]

        src, dst = edge_index
        if self.conv is None:
            neighbor = node_features[src]
        else:
            neighbor = self.conv(input)["output_features"]
        center = node_features[dst]
        off_diagonal = self.tp_off(left=self.res_center_off(center), right=neighbor)
        off_diagonal = self.res_pair_off(off_diagonal)
        off_diagonal = self.res_res_off(center) + off_diagonal

        center = node_features
        neighbor = node_features
        diagonal = self.tp(left=self.res_center(center), right=neighbor)
        diagonal = self.res_pair(diagonal)
        diagonal = self.res_res(center) + diagonal

        attrs_diag = ("edge", self.irreps_out["diagonal"])
        attrs_off = ("edge", self.irreps_out["off_diagonal"])
        data.attrs.update(
            self.outputKeyMap({"diagonal": attrs_diag, "off_diagonal": attrs_off})
        )
        output = self.outputKeyMap({"diagonal": diagonal, "off_diagonal": off_diagonal})
        data.update(output)
        return data


from functools import lru_cache


@lru_cache(maxsize=None)
def get_clebsch_gordon(i: int, j: int, k: int, device):
    return wigner_3j(i, j, k, dtype=torch.float32, device=device)


class TensorProductContraction(Module, torch.nn.Module):
    """
    This module composes irreducible representations into tensor product representations tp_l*tp_r for each atom pair.
    For simplicity, it assumes all atoms use the same set of basis, which means the result is padded (using a larger basis then necessary) for simpler atoms.
    """

    def __init__(self, irreps_in, tp_l, tp_r):
        super().__init__()

        self.init_irreps(
            irreducible=irreps_in, tp_l=tp_l, tp_r=tp_r, output_keys=["tp_l", "tp_r"]
        )

        self.irreps_mul = {}
        # counts the mul of irreps needed to construct the tensor product tp_l*tp_r
        # for example, if tp_l = 1x0o+1x1e, tp_r = 2x2e, irreps_mul = {1e: 2, 2e: 2, 2o:2, 3e:2}
        for mul_l, (degree_l, parity_l) in self.irreps_out["tp_l"]:
            for mul_r, (degree_r, parity_r) in self.irreps_out["tp_r"]:
                parity = parity_l * parity_r
                parity = "e" if parity == 1 else "o"
                mul = mul_l * mul_r
                for degree in range(abs(degree_l - degree_r), degree_l + degree_r + 1):
                    key = f"{degree}{parity}"
                    if not key in self.irreps_mul:
                        self.irreps_mul[key] = mul
                    else:
                        self.irreps_mul[key] += mul
        self.irreps = "+".join(
            [f"{value}x{key}" for key, value in self.irreps_mul.items()]
        )
        self.irreps = Irreps(self.irreps)
        self.linear = PointwiseLinear(irreps_in, self.irreps)

    def forward(self, data):
        input = self.inputKeyMap(data)["irreducible"]
        input = self.linear(input)
        irreps_mul = copy(self.irreps_mul)
        tp = {}
        # fills the tensor product matrix with irreps
        for mul_l, (degree_l, parity_l) in self.irreps_out["tp_l"]:
            for mul_r, (degree_r, parity_r) in self.irreps_out["tp_r"]:
                p_l = "e" if parity_l == 1 else "o"
                p_r = "e" if parity_r == 1 else "o"
                tp_key = f"{mul_l}x{degree_l}{p_l}*{mul_r}x{degree_r}{p_r}"
                tp[tp_key] = 0
                mul = mul_l * mul_r
                for i, irrep in enumerate(self.irreps):
                    degree, parity = irrep[1]
                    parity = "e" if parity == 1 else "o"
                    if not tp_path_exists(
                        f"{degree_l}{p_l}", f"{degree_r}{p_r}", f"{degree}{parity}"
                    ):
                        continue
                    key = str(irrep.ir)
                    # fetch irreps from the tail
                    start = self.irreps.slices()[i].start
                    stop = start + irreps_mul[key] * (degree * 2 + 1)
                    start = stop - mul * (degree * 2 + 1)
                    basis = get_clebsch_gordon(degree_l, degree_r, degree, input.device)
                    a = input[:, start:stop].reshape(-1, mul_l, mul_r, degree * 2 + 1)
                    tp[tp_key] = tp[tp_key] + torch.einsum("bmni,lri->bmlnr", a, basis)
                    irreps_mul[key] -= mul
        for key, value in irreps_mul.items():
            assert value == 0

        # the outputs are not irreps, they are dict of tensors instead of tensors
        output = self.outputKeyMap({"tp_l": tp})
        data.update(output)
        return data
