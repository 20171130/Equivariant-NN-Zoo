from functools import partial
from ..data import computeEdges
from ml_collections.config_dict import ConfigDict
import ase
from .layer_configs import featureModel, addEnergyOutput


def get_config(spec=None):
    config = ConfigDict()
    data, model = ConfigDict(), ConfigDict()
    config.data_config = data
    config.model_config = model

    config.epoch_subdivision = 1
    config.learning_rate = 1e-2
    config.batch_size = 64

    config.use_ema = True
    config.ema_decay = 0.99
    config.ema_use_num_updates = True
    config.metric_key = "validation_loss"  # saves the best model according to this

    config.max_epochs = int(1e6)
    config.early_stopping_patiences = {"validation_loss": 20}
    config.early_stopping_lower_bounds = {"LR": 1e-6}

    config.loss_coeffs = {"total_energy": [1e3, "PerAtomMSELoss"]}
    config.metrics_components = {"total_energy": ["mae"]}
    config.optimizer_name = "Adam"
    config.lr_scheduler_name = "ReduceLROnPlateau"
    config.lr_scheduler_patience = 1
    config.lr_scheduler_factor = 0.8

    model.n_dim = 64
    model.l_max = 3
    model.r_max = 4.0
    model.num_layers = 4
    model.node_attrs = "8x0e"
    num_types = 10

    data.n_train = 120000
    data.n_val = 10831
    data.train_val_split = "random"
    data.shuffle = True
    data.path = "qm9.hdf5"
    data.type_names = list(ase.atom.atomic_numbers.keys())[:num_types]
    data.key_map = {"Z": "atom_types", "R": "pos", "U": "total_energy"}
    data.preprocess = [partial(computeEdges, r_max=model.r_max)]

    "+".join([f"{model.n_dim}x{n}e+{model.n_dim}x{n}o" for n in range(model.l_max + 1)])

    edge_spherical = "1x0e+1x1o+1x2e"
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
    shifts = [
        -620.4502,
        -16.4435,
        -620.4502,
        -620.4502,
        -620.4502,
        -620.4502,
        -1036.0271,
        -1489.8005,
        -2046.9702,
        -2717.4263,
    ]
    # can be computed by calling  dataset.statistics(['total_energy-per-atom_types-mean_std'], stride=stride )
    # refer to data.ipynb
    layer_configs = addEnergyOutput(layer_configs, shifts)
    model.update(layer_configs)

    return config
