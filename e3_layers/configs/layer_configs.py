from functools import partial
from ml_collections.config_dict import ConfigDict
from ..nn import *
from ..utils import tp_path_exists, insertAfter, replace
from e3nn.o3 import Irreps
from copy import deepcopy


def featureModel(
    n_dim,
    l_max,
    edge_radial,
    num_types,
    num_layers,
    node_attrs,
    r_max,
    edge_spherical=None
):
    config = ConfigDict()

    config.n_dim = n_dim
    config.l_max = l_max
    config.edge_spherical = edge_spherical
    config.edge_radial = edge_radial
    config.num_types = num_types
    config.num_layers = num_layers
    config.r_max = r_max
    config.module = SequentialGraphNetwork
    node_features = "+".join([f"{n_dim}x{n}e+{n_dim}x{n}o" for n in range(l_max + 1)])
    if edge_spherical is None:
        edge_spherical = "+".join(
            [f"1x{n}e" if n % 2 == 0 else f"1x{n}o" for n in range(l_max + 1)]
        )
    config.node_features = node_features
    config.edge_spherical = edge_spherical
    config.node_attrs = node_attrs

    layers = {}
    layers["one_hot"] = {
        "module": OneHotEncoding,
        "irreps_out": (f"{num_types}x0e", "one_hot"),
        "irreps_in": ("1x0e", "atom_types"),
    }
    layers["spharm_edges"] = {
        "module": SphericalEncoding,
        "irreps_out": (edge_spherical, "edge_spherical"),
        "irreps_in": ("1x1o", "edge_vector"),
    }
    layers["radial_basis"] = {
        "module": RadialBasisEdgeEncoding,
        "r_max": r_max,
        "trainable": True,
        "polynomial_degree": 6,
        "irreps_out": (edge_radial, "edge_radial"),
    }
    layers["chemical_embedding"] = {
        "module": PointwiseLinear,
        "irreps_in": (f"{num_types}x0e", "one_hot"),
        "irreps_out": (f"{n_dim}x0e", "node_features"),
    }
    layers["node_attrs"] = {
        "module": PointwiseLinear,
        "irreps_in": (f"{num_types}x0e", "one_hot"),
        "irreps_out": (node_attrs, "node_attrs"),
    }
    irreps = {
        "node_attrs": node_attrs,
        "input_features": [node_features, "node_features"],
        "edge_radial": edge_radial,
        "edge_spherical": edge_spherical,
        "output_features": [node_features, "node_features"],
    }
    conv = {
        "module": FactorizedConvolution,
        "avg_num_neighbors": 10,
        "use_sc": True,
        "invariant_layers": 3,
        "invariant_neurons": n_dim,
    }
    mp = {
        "module": MessagePassing,
        "resnet": False,
        "convolution": conv,
        "nonlinearity_type": "gate",
        "nonlinearity_scalars": {"e": "silu", "o": "tanhlu"},
        "nonlinearity_gates": {"e": "silu", "o": "tanhlu"},
        **irreps,
    }
    cur_node_features = Irreps(f"{n_dim}x0e")
    node_features = Irreps(node_features)
    for layer_i in range(num_layers):
        cur = deepcopy(mp)
        cur["input_features"][0] = cur_node_features
        cur_node_features = [
            (mul, ir)
            for mul, ir in node_features
            if tp_path_exists(cur_node_features, edge_spherical, ir)
        ]
        cur_node_features = Irreps(cur_node_features)
        cur["output_features"][0] = cur_node_features
        layers[f"layer{layer_i}"] = cur

    config.layers = list(layers.items())
    return config


def addEdgeEmbedding(config, num_bond_types):
    layers = {}
    config.num_bond_types = num_bond_types
    n_dim, r_max, edge_radial = config.n_dim, config.r_max, config.edge_radial
    layers["edge_onehot"] = {
        "module": OneHotEncoding,
        "irreps_out": (f"{num_bond_types}x0e", "edge_onehot"),
        "irreps_in": ("1x0e", "bond_type"),
    }
    layers["edge_embedding"] = {
        "module": PointwiseLinear,
        "irreps_in": (f"{num_bond_types}x0e", "edge_onehot"),
        "irreps_out": (f"{n_dim}x0e", "edge_embedding"),
    }
    layers = list(layers.items())
    layers = layers + config.layers
    
    layer = {
        "module": RadialBasisEdgeEncoding,
        "r_max": r_max,
        "trainable": True,
        "polynomial_degree": 6,
        "edge_attr": (f"{n_dim}x0e", "edge_embedding"),
        "irreps_out": (edge_radial, "edge_radial"),
    }
    replace(layers, 'radial_basis', ('radial_basis', layer))

    return config


def addEnergyOutput(config, shifts):
    layers = {}
    layers["output_linear"] = {
        "module": PointwiseLinear,
        "irreps_in": (config.node_features, "node_features"),
        "irreps_out": ("1x0e", "energy"),
    }

    layers["rescale"] = {
        "module": PerTypeScaleShift,
        "num_types": config.num_types,
        "shifts": shifts,
        "scales": None,
        "irreps_in": ("1x0e", "energy"),
        "irreps_out": ("1x0e", "energy"),
        "species": ("1x0e", "atom_types"),
    }

    layers["reduce"] = {
        "module": Pooling,
        "reduce": "sum",
        "irreps_in": ("1x0e", "energy"),
        "irreps_out": ("1x0e", "total_energy"),
    }
    config.layers += layers.items()
    return config


def addForceOutput(config):
    config = ConfigDict()
    config.func = func_config
    config.update(
        {
            "module": GradientOutput,
            "x": ("1x1o", "pos"),
            "y": ("1x1o", "energy"),
            gradient: ("1x0e", "forces"),
        }
    )
    return config


def addMatrixOutput(config, tp_l, tp_r):
    layers = {}
    layers["pairwise"] = dict(
        module=Pairwise,
        node_features=config.node_features,
        edge_radial=config.edge_radial,
        edge_spherical=config.edge_spherical,
        diagonal=config.node_features,
        off_diagonal=config.node_features,
        conv="auto",
    )
    layers["irreps2tp_diagonal"] = dict(
        module=TensorProductContraction,
        irreps_in=(config.node_features, "diagonal"),
        tp_l=(tp_l, "hamiltonian_diagonal"),
        tp_r=(tp_r, "hamiltonian_diagonal"),  # O: 3s 2p 1d
    )
    layers["irreps2tp_off"] = dict(
        module=TensorProductContraction,
        irreps_in=(config.node_features, "off_diagonal"),
        tp_l=(tp_l, "hamiltonian_off"),
        tp_r=(tp_r, "hamiltonian_off"),  # O: 3s 2p 1d
    )
    config.layers += layers.items()
    return config
