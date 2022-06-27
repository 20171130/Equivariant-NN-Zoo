import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from .sequential import Module
from .pointwise import PointwiseLinear
import torch
import torch.nn.functional

from e3nn.util.jit import compile_mode

from typing import Optional
import math

import torch

from torch import nn

from e3nn.math import soft_one_hot_linspace
from e3nn.util.jit import compile_mode


@torch.jit.script
def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)


class PolynomialCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        return _poly_cutoff(x, self._factor, p=self.p)


@compile_mode("trace")
class e3nn_basis(nn.Module):
    r_max: float
    r_min: float
    e3nn_basis_name: str
    num_basis: int

    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = None,
        e3nn_basis_name: str = "gaussian",
        num_basis: int = 8,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min if r_min is not None else 0.0
        self.e3nn_basis_name = e3nn_basis_name
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return soft_one_hot_linspace(
            x,
            start=self.r_min,
            end=self.r_max,
            number=self.num_basis,
            basis=self.e3nn_basis_name,
            cutoff=True,
        )

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(5, 1),)} for _ in range(n)]


class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True, one_over_r=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
            
        one_over_r:
            Set to true if the value should explode at x = 0, e.g. when x is the interatomic distance.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max
        self.one_over_r = one_over_r

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        result = self.prefactor * numerator
        if self.one_over_r:
            result = result/x.unsqueeze(-1)
        return  result


@compile_mode("script")
class SphericalEncoding(Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_out,
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in="1x1o",
    ):
        super().__init__()
        self.init_irreps(
            vectors=irreps_in,
            spherical_harmonics=irreps_out,
            output_keys=["spherical_harmonics"],
        )
        self.mul = self.irreps_in["vectors"][0].mul
        irreps = []
        for irrep in self.irreps_out["spherical_harmonics"]:
            assert irrep.mul == self.mul
            irreps += [str(irrep.ir)]
        self.sh = o3.SphericalHarmonics(
            "+".join(irreps), edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, data):
        vectors = self.inputKeyMap(data)["vectors"]
        cat, _ = vectors.shape
        edge_sh = self.sh(vectors.view(cat, self.mul, 3)).view(cat, -1)
        data.attrs.update(
            self.outputKeyMap(
                {
                    "spherical_harmonics": (
                        "edge",
                        self.irreps_out["spherical_harmonics"],
                    )
                }
            )
        )
        data.update(self.outputKeyMap({"spherical_harmonics": edge_sh}))
        return data


@compile_mode("script")
class RadialBasisEncoding(Module):
    def __init__(
        self,
        r_max,
        polynomial_degree,
        trainable,
        irreps_out,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        real="1x0e",
        one_over_r = True,
        input_features=None,
    ):
        """
        Radial basis embedding of a positive real number. Optionally composes the embedding with input features.
        """
        super().__init__()
        self.init_irreps(
            real=real,
            input_features=input_features,
            radial_embedding=irreps_out,
            output_keys=["radial_embedding"],
        )
        num_basis = self.irreps_out["radial_embedding"]
        num_basis = num_basis[0].mul
        self.basis = basis(r_max, num_basis, trainable, one_over_r=one_over_r)
        self.cutoff = cutoff(r_max, p=polynomial_degree)

    def forward(self, data):
        input = self.inputKeyMap(data)
        input_features = None
        real = input['real']
        if "input_features" in input:
            input_features = input["input_features"]
        embedded = (
            self.basis(real) * self.cutoff(real)[:, None]
        )
        if not input_features is None:
            embedded = embedded * input_features
        is_per = input.attrs['real'][0]
        data.attrs.update(
            self.outputKeyMap(
                {"radial_embedding": (is_per, self.irreps_out["radial_embedding"])}
            )
        )
        data.update(self.outputKeyMap({"radial_embedding": embedded}))
        return data
      
@compile_mode("script")
class GraphFeatureEmbedding(Module):
    def __init__(
        self, graph=None, edge_in=None, node_in=None, edge_out=None, node_out=None
    ):
        """
        Broadcast graph features into edge or node features.
        """
        super().__init__()
        self.init_irreps(
            graph = graph,
            edge_in = edge_in, edge_out=edge_out,
            node_in = node_in, node_out=node_out,
            output_keys=["edge_out", "node_out"],
        )
        if 'edge_out' in self.irreps_out:
            irreps_in = self.irreps_in['edge_in'] + self.irreps_in['graph']
            self.linear_edge = PointwiseLinear(irreps_in, self.irreps_out['edge_out'])
        if 'node_out' in self.irreps_out:
            irreps_in = self.irreps_in['node_in'] + self.irreps_in['graph']
            self.linear_node = PointwiseLinear(irreps_in, self.irreps_out['node_out'])

    def forward(self, data):
        input = self.inputKeyMap(data)
        
        if 'edge_out' in self.irreps_out:
            device = input['graph'].device
            feature = input['graph'][data.edgeSegment()]
            feature = feature/feature.shape[-1]**0.5
            if 'edge_in' in self.irreps_in:
                feature = torch.cat([input['edge_in'], feature], dim=1)
            feature = self.linear_edge(feature)
            data.attrs.update(
                self.outputKeyMap(
                    {"edge_out": ("edge", self.irreps_out["edge_out"])}
                )
            )
            data.update(self.outputKeyMap({"edge_out": feature}))
          
        if 'node_out' in self.irreps_out:
            device = input['graph'].device
            feature = input['graph'][data.nodeSegment()]
            if 'node_in' in self.irreps_in:
                feature = torch.cat([input['node_in'], feature], dim=1)
            feature = self.linear_node(feature)
            data.attrs.update(
                self.outputKeyMap(
                    {"node_out": ("node", self.irreps_out["node_out"])}
                )
            )
            data.update(self.outputKeyMap({"node_out": feature}))
        
        return data

@compile_mode("script")
class OneHotEncoding(Module):
    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        irreps_out,
        set_features: bool = True,
        irreps_in="0x0e",
    ):
        super().__init__()
        self.num_types = num_types
        self.init_irreps(input=irreps_in, one_hot=irreps_out, output_keys="one_hot")

    def forward(self, data):
        input = self.inputKeyMap(data)["input"]
        type_numbers = input.squeeze(-1)

        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=torch.float)

        tmp = self.inputKeyMap(data.attrs)
        data.attrs.update(
            self.outputKeyMap(
                {"one_hot": (tmp["input"][0], self.irreps_out["one_hot"])}
            )
        )
        result = {"one_hot": one_hot}
        data.update(self.outputKeyMap(result))
        return data
