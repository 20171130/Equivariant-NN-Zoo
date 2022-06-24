from functools import partial
from ..data import computeEdgeIndex
from ml_collections.config_dict import ConfigDict
import ase
from .layer_configs import featureModel
from ..nn import PointwiseLinear, RadialBasisEncoding, GraphFeatureEmbedding
from ..utils import insertAfter


def get_config(spec=''):
    config = ConfigDict()
    data, model = ConfigDict(), ConfigDict()
    config.data_config = data
    config.model_config = model

    config.learning_rate = 1e-2
    config.batch_size = 128

    config.use_ema = True
    config.ema_decay = 0.99
    config.ema_use_num_updates = True
    config.metric_key = "validation_loss"  # saves the best model according to this

    config.max_epochs = int(1e6)
    config.early_stopping_patiences = {"validation_loss": 20}
    config.early_stopping_lower_bounds = {"LR": 1e-6}

    config.loss_coeffs = {"dipole": [1e3, "MSELoss"]}
    config.metrics_components = {"dipole": ["mae"]}
    config.optimizer_name = "Adam"
    config.lr_scheduler_name = "ReduceLROnPlateau"
    config.lr_scheduler_patience = 2
    config.lr_scheduler_factor = 0.8

    model.n_dim = 32
    model.l_max = 2
    model.r_max = 5.0
    model.num_layers = 5
    model.edge_radial = '8x0e'
    model.node_attrs = "16x0e"
    num_types = 18

    data.n_train = 120000
    data.n_val = 10831
    data.std = 2.4208
    data.train_val_split = "random"
    data.shuffle = True
    data.path = "qm9.hdf5"
    data.type_names = list(ase.atom.atomic_numbers.keys())[:num_types]
    data.preprocess = [partial(computeEdgeIndex, r_max=model.r_max)]
    data.key_map = {"Z": "atom_types", "R": "pos", "U": "total_energy"}
    
    if spec and 'profiling' in spec:
        data.n_train = 2048
        data.n_val = 256

    features = "+".join(
        [f"{model.n_dim}x{n}e+{model.n_dim}x{n}o" for n in range(model.l_max + 1)]
    )

    edge_spherical = "1x0e+1x1o+1x2e"
    layer_configs = featureModel(
        n_dim=model.n_dim,
        l_max=model.l_max,
        edge_spherical=edge_spherical,
        node_attrs=model.node_attrs,
        edge_radial=model.edge_radial,
        num_types=num_types,
        num_layers=model.num_layers,
        r_max=model.r_max,
    )

    if 'embed_time_in_nodes' in spec:
        time_encoding = ('time_encoding', {
            "module": RadialBasisEncoding,
            "r_max": 1.0,
            "trainable": True,
            "polynomial_degree": 6,
            "real": ("1x0e", "t"),
            'one_over_r': False,
            "irreps_out": (f"{model.n_dim}x0e", "time_encoding"),
        })
        layer_configs.layers = insertAfter(layer_configs.layers, 'chemical_embedding', time_encoding)
        time_embedding = ('time_embedding', {'module': GraphFeatureEmbedding,
                                             'graph': (f"{model.n_dim}x0e", 'time_encoding'),
                                             'node_in': (f"{model.n_dim}x0e", 'node_features'),
                                             'node_out': (f"{model.n_dim}x0e", 'node_features')
                                            })
        
    else:
        time_encoding = ('time_encoding', {
            "module": RadialBasisEncoding,
            "r_max": 1.0,
            "trainable": True,
            "polynomial_degree": 6,
            "real": ("1x0e", "t"),
            'one_over_r': False,
            "irreps_out": (model.edge_radial, "time_encoding"),
        })
        layer_configs.layers = insertAfter(layer_configs.layers, 'radial_basis', time_encoding)
        time_embedding = ('time_embedding', {'module': GraphFeatureEmbedding,
                                         'graph': (model.edge_radial, 'time_encoding'),
                                         'edge_in': (model.edge_radial, 'edge_radial'),
                                         'edge_out': (model.edge_radial, 'edge_radial')
                                        })
    
    layer_configs.layers = insertAfter(layer_configs.layers, 'time_encoding', time_embedding)
  
    layer_configs.layers.append(
        (
            "score_output",
            {
                "module": PointwiseLinear,
                "irreps_in": (features, "node_features"),
                "irreps_out": (f"1x1o", "score"),
            },
        )
    )
    model.update(layer_configs)

    return config
