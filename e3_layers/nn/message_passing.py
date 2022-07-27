import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.nn import Gate, NormActivation
from e3nn.o3 import Irreps
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Linear, FullyConnectedTensorProduct

from .pointwise import TensorProductExpansion, LayerNormalization
from .sequential import Module
from ..utils import build, tp_path_exists, activations
from e3nn.util.jit import compile_mode

from torch import Tensor
from e3nn.util.jit import compile_mode
from typing import Optional, Dict, Tuple, Callable
from typing_extensions import Final

@compile_mode("script")
class FactorizedConvolution(Module, torch.nn.Module):
    avg_num_neighbors: Optional[float]
    use_sc: bool

    def __init__(
        self,
        input_features,
        output_features,
        node_attrs,
        edge_radial,
        edge_spherical,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=None,
        use_sc=True,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
        reduce=True,
    ) -> None:
        super().__init__()

        self.init_irreps(
            input_features=input_features,
            output_features=output_features,
            node_attrs=node_attrs,
            edge_radial=edge_radial,
            edge_spherical=edge_spherical,
            output_keys=["output_features"],
        )

        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc

        feature_irreps_in = self.irreps_in["input_features"]
        feature_irreps_out = self.irreps_out["output_features"]
        irreps_edge_attr = self.irreps_in["edge_spherical"]

        # - Build modules -
        self.linear_1 = Linear(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        self.tp = TensorProductExpansion(
            feature_irreps_in,
            (irreps_edge_attr, "edge_spherical"),
            (feature_irreps_out, "edge_features"),
            "uvu",
            internal_weight=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [Irreps(self.irreps_in["edge_radial"]).num_irreps]
            + invariant_layers * [invariant_neurons]
            + [self.tp.tp.weight_numel],
            activations["ssp"],
        )

        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(
                feature_irreps_in,
                Irreps(self.irreps_in["node_attrs"]),
                feature_irreps_out,
            )

        self.reduce = reduce

    def forward(self, data: Dict[str, Tensor], attrs:Dict[str, Tuple[str, str]]):
        input = data
        weight = self.fc(input["edge_radial"])

        x = input["input_features"]
        edge_src = input["edge_index"][0]
        edge_dst = input["edge_index"][1]

        if self.sc is not None:
            sc = self.sc(x, input["node_attrs"])

        x = self.linear_1(x)
        
        edge_features = self.tp(
            left=x[edge_src], right=input["edge_spherical"], weight=weight
        )
        if self.reduce:
            # [edges, feature_dim], [edges, sh_dim], [edges, weight_numel]
            x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

            # Necessary to get TorchScript to be able to type infer when its not None
            avg_num_neigh: Optional[float] = self.avg_num_neighbors
            if avg_num_neigh is not None:
                x = x.div(avg_num_neigh ** 0.5)

            if self.sc is not None:
                x = x + sc
        else:
            x = edge_features

        is_per = attrs["input_features"][0]
        attrs = {"output_features": (is_per, self.irreps_out["output_features"])}
        data = {"output_features": x}
        return data, attrs

@compile_mode("script")
class MessagePassing(Module, torch.nn.Module):
    normalize: Final[bool]
    def __init__(
        self,
        input_features,
        output_features,
        node_attrs,
        edge_radial,
        edge_spherical,
        convolution,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
        normalize=False
    ):
        """Convolution with nonlinearity and residual connection"""
        super().__init__()
        # initialization
        self.init_irreps(
            input_features=input_features,
            output_features=output_features,
            node_attrs=node_attrs,
            edge_radial=edge_radial,
            edge_spherical=edge_spherical,
            output_keys=["output_features"],
        )

        assert nonlinearity_type in ("gate", "norm")
        # make the nonlin dicts from parity ints instead of convinience strs
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }
        self.resnet = resnet

        edge_attr_irreps = Irreps(self.irreps_in["edge_spherical"])
        irreps_layer_out_prev = Irreps(self.irreps_in["input_features"])
        self.feature_irreps_hidden = Irreps(self.irreps_out["output_features"])

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0
                and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir)
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0
                and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir)
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = "0e"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    activations[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[
                    activations[nonlinearity_gates[ir.p]] for _, ir in irreps_gates
                ],
                irreps_gated=irreps_gated,
            )

            conv_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else:
            conv_irreps_out = irreps_layer_out.simplify()

            equivariant_nonlin = NormActivation(
                irreps_in=conv_irreps_out,
                # norm is an even scalar, so use nonlinearity_scalars[1]
                scalar_nonlinearity=activations[nonlinearity_scalars[1]],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        self.equivariant_nonlin = equivariant_nonlin

        # TODO: partial resnet?
        if irreps_layer_out == irreps_layer_out_prev and resnet:
            # We are doing resnet updates and can for this layer
            self.resnet = True
        else:
            self.resnet = False

        self.conv = build(
            convolution,
            input_features=input_features,
            output_features=conv_irreps_out,
            node_attrs=node_attrs,
            edge_radial=edge_radial,
            edge_spherical=edge_spherical,
        )
        self.normalize = normalize
        if self.normalize:
            self.norm = LayerNormalization(self.irreps_out["output_features"], self.irreps_out["output_features"])

    def forward(self, data: Dict[str, Tensor], attrs:Dict[str, Tuple[str, str]]):
        # save old features for resnet
        old_x = data["input_features"]
        # run convolution
        data, _ = self.conv(data, attrs)
        output = data["output_features"]
        # do nonlinearity
        output = self.equivariant_nonlin(output)
        # do resnet
        #   output = output + self.tp(output, self.linear_tp(output))
        if self.resnet:
            output = old_x + output
            
        if self.normalize:
            tmp = self.norm({'input':output}, attrs)[0]
            output = tmp['output']

        is_per = attrs["input_features"][0]
        attrs = {"output_features": (is_per, self.irreps_out["output_features"])}
        data = {"output_features": output}
        return data, attrs
