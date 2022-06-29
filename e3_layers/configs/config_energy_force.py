from functools import partial
from ..data import computeEdgeIndex
from ml_collections.config_dict import ConfigDict
import ase
from .layer_configs import featureModel, addEnergyOutput, addForceOutput


def get_config(spec=None):
    """
    Jointly predicting energy and forces.
    """
    config = ConfigDict()
    data, model = ConfigDict(), ConfigDict()
    config.data_config = data
    config.model_config = model

    config.epoch_subdivision = 20
    config.learning_rate = 1e-2
    config.batch_size = 128

    config.use_ema = True
    config.ema_decay = 0.99
    config.ema_use_num_updates = True
    config.metric_key = "validation_loss"  # saves the best model according to this

    config.max_epochs = int(1e6)
    config.early_stopping_patiences = {"validation_loss": 20}
    config.early_stopping_lower_bounds = {"LR": 1e-6}

    config.loss_coeffs = {"energy": [1e3, "MSELoss"], "forces": [1e3, "MSELoss"]}
    config.metrics_components = {"energy": ["mae"], "forces": ["mae"]}
    config.optimizer_name = "Adam"
    config.lr_scheduler_name = "ReduceLROnPlateau"
    config.lr_scheduler_patience = 1
    config.lr_scheduler_factor = 0.8

    model.n_dim = 32
    model.l_max = 2
    model.r_max = 4.0
    model.num_layers = 4
    model.node_attrs = "8x0e"
    num_types = 20

    data.n_train = 2560000
    data.n_val = 171180
    data.train_val_split = "random"
    data.shuffle = True
    data.path = "/opt/shared-data/proteindata_cz/protein_E_and_F.hdf5"
    data.type_names = list(ase.atom.atomic_numbers.keys())[:num_types]
    data.preprocess = [partial(computeEdgeIndex, r_max=model.r_max)]
    
    override = eval(spec)
    config.update_from_flattened_dict(override)

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
    # cannot compute edge vectors during preprocessing since we need the gradients
    shifts = [-3.7204, -2.2483, -3.7204, -3.7204, -3.7204, -3.7204, -7.6108, -4.0182,
        -5.2651, -3.7204, -3.7204, -3.7204, -3.7204, -3.7204, -3.7204, -3.7204,
        -3.2213, -3.7204, -3.7204, -3.7204]
    # can be computed by calling  dataset.statistics(['energy-per-atom_types-mean_std'], stride=stride )
    # refer to data.ipynb
    layer_configs = addEnergyOutput(layer_configs, shifts, output_key='energy')
    layer_configs = addForceOutput(layer_configs)
    model.update(layer_configs)
    

    return config
