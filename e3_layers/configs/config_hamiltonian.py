from functools import partial
from ..data import computeEdgeIndex
from ml_collections.config_dict import ConfigDict
import ase
from .layer_configs import featureModel, addMatrixOutput
import e3nn
import torch


def transform(result):
    """
    It transforms the hamiltonian from e3nn basis to ORCA basis.
    """
    S = torch.ones(1, 1)
    P = torch.tensor([[0, 1.0, 0], [0, 0, 1], [1, 0, 0]])
    D = torch.tensor(
        [
            [0, 1, 0, 0, 0.0],
            [0, 0, 0, 0, 1],
            [-0.5, 0, 0, -((3 / 4) ** 0.5), 0],
            [0, 0, 1, 0, 0],
            [((3 / 4) ** 0.5), 0, 0, -0.5, 0],
        ]
    )
    M = e3nn.math.direct_sum(S, S, S, P, P, D, S, S, P, S, S, P).to(result.device)
    result = M.T @ result @ M
    # from e3nn to ORCA
    return result


def contractBasis(data):
    """
    This function fills the molecular hamiltonian with atompair hamiltonians (removing padding basis).
    It also removes the padding basis.
    It is just a reshape, and does not have any trainable parameters.
    Notice that this function only works for H2O, but it is not hard to generalize it.
    """

    batch = data.n_graphs
    diagonal = data["hamiltonian_diagonal"]
    off = data["hamiltonian_off"]
    device = list(diagonal.values())[0].device

    result = torch.zeros((batch, 24, 24)).to(device)
    orbitals = [
        (0, 0, 3),
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 1, 1),
        (2, 0, 2),
        (2, 1, 1),
    ]
    dic = {(0, 1): 0, (0, 2): 1, (1, 0): 2, (1, 2): 3, (2, 0): 4, (2, 1): 5}
    # atom_index, degree, mul
    full = [3, 2, 1]  # padded basis
    i_cnt = 0
    for i, degree_i, mul_i in orbitals:
        j_cnt = 0
        parity_l = "e" if degree_i % 2 == 0 else "o"
        irreps_l = e3nn.o3.Irreps(f"{mul_i}x{degree_i}{parity_l}")
        _irreps_l = e3nn.o3.Irreps(
            f"{full[degree_i]}x{degree_i}{parity_l}"
        )  # the padded basis
        dim_l = irreps_l.dim

        for j, degree_j, mul_j in orbitals:
            parity_r = "e" if degree_j % 2 == 0 else "o"
            irreps_r = e3nn.o3.Irreps(f"{mul_j}x{degree_j}{parity_r}")
            _irreps_r = e3nn.o3.Irreps(
                f"{full[degree_j]}x{degree_j}{parity_r}"
            )  # the padded basis
            dim_r = irreps_r.dim

            key = f"{_irreps_l}*{_irreps_r}"
            if i == j:
                H = diagonal[key].reshape(batch, 3, _irreps_l.dim, _irreps_r.dim)
                H = H[:, i, :dim_l, :dim_r]
            else:
                H = off[key].reshape(batch, 6, _irreps_l.dim, _irreps_r.dim)
                idx = dic[(i, j)]
                H = H[:, idx, :dim_l, :dim_r]
            result[:, i_cnt : i_cnt + dim_l, j_cnt : j_cnt + dim_r] = H
            j_cnt += dim_r
        assert j_cnt == 24
        i_cnt += dim_l
    assert i_cnt == 24
    result = (result + result.transpose(2, 1)) / 2
    result = transform(result)
    result = result.view(batch, -1)
    data.update({"hamiltonian": result})
    return data


def get_config(spec=None):
    config = ConfigDict()
    data, model = ConfigDict(), ConfigDict()
    config.data_config = data
    config.model_config = model

    config.epoch_subdivision = 1
    config.learning_rate = 1e-2
    config.batch_size = 16

    config.use_ema = True
    config.ema_decay = 0.99
    config.ema_use_num_updates = True
    config.metric_key = "validation_loss"  # saves the best model according to this

    config.max_epochs = int(1e6)
    config.early_stopping_patiences = {"validation_loss": 20}
    config.early_stopping_lower_bounds = {"LR": 1e-6}

    config.loss_coeffs = {"hamiltonian": [1e5, "MSELoss"]}
    config.metrics_components = {"hamiltonian": ["mae"]}
    config.optimizer_name = "Adam"
    config.lr_scheduler_name = "ReduceLROnPlateau"
    config.lr_scheduler_patience = 8
    config.lr_scheduler_factor = 0.8

    model.n_dim = 64
    model.l_max = 4
    model.r_max = 4.0
    model.num_layers = 5
    model.node_attrs = "8x0e"
    num_types = 9

    data.n_train = 500
    data.n_val = 500
    data.train_val_split = "random"
    data.shuffle = True
    data.path = "h2o.hdf5"
    data.type_names = list(ase.atom.atomic_numbers.keys())[:num_types]
    data.preprocess = [partial(computeEdgeIndex, r_max=model.r_max)]

    features = "+".join(
        [f"{model.n_dim}x{n}e+{model.n_dim}x{n}o" for n in range(model.l_max + 1)]
    )
    irreps_in = {
        "pos": "1x1o",
        "node_features": features,
        "diagonal": features,
        "off_diagonal": features,
    }

    edge_spherical = "1x0e+1x1o+1x2e+1x3o"
    layer_configs = featureModel(
        n_dim=model.n_dim,
        l_max=model.l_max,
        edge_spherical=edge_spherical,
        node_attrs=model.node_attrs,
        edge_radial="8x0e",
        num_types=num_types,
        num_layers=model.num_layers,
        r_max=model.r_max,
    )
    layer_configs = addMatrixOutput(layer_configs, "3x0e+2x1o+1x2e", "3x0e+2x1o+1x2e")
    layer_configs.layers.append(("hamiltonian", contractBasis))

    model.update(layer_configs)

    return config
