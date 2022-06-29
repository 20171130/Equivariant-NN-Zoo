from functools import partial
from ..data import computeEdgeIndex
from ml_collections.config_dict import ConfigDict
import ase
from .layer_configs import featureModel, addEdgeEmbedding, addForceOutput, addEnergyOutput
from ..nn import PointwiseLinear, RadialBasisEncoding, GraphFeatureEmbedding
from ..utils import insertAfter, saveProtein


def get_config(spec=''):
    config = ConfigDict()
    data, model = ConfigDict(), ConfigDict()
    config.data_config = data
    config.model_config = model

    config.learning_rate = 1e-2
    config.batch_size = 128

    config.use_ema = True
    config.ema_decay = 0.99
    config.config_spec = spec
    config.ema_use_num_updates = True
  #  config.metric_key = "validation_loss"  # saves the best model according to this

  #  config.max_epochs = int(1e6)
  #  config.early_stopping_patiences = {"validation_loss": 20}
  #  config.early_stopping_lower_bounds = {"LR": 1e-6}

  #  config.loss_coeffs = {"dipole": [1e3, "MSELoss"]}
  #  config.metrics_components = {"dipole": ["mae"]}
    config.optimizer_name = "Adam"
    config.lr_scheduler_name = "ReduceLROnPlateau"
    config.lr_scheduler_patience = 15 # number of eval steps
    config.lr_scheduler_factor = 0.8
    config.grad_clid_norm = 1.0
    config.saveMol = saveProtein

    model.n_dim = 32
    model.l_max = 2
    model.r_max = 5.0 
    model.num_layers = 4
    model.edge_radial = '8x0e'
    model.node_attrs = "16x0e"
    num_types = 21

    data.n_train = 50000
    data.n_val = 3009
    data.std = 17.6163 
    data.train_val_split = "random"
    data.shuffle = True
    data.path = "tmp.hdf5"
    data.type_names = list(ase.atom.atomic_numbers.keys())[:num_types]
    data.preprocess = []
    if not 'gcn' in spec:
        data.preprocess = [partial(computeEdgeIndex, r_max=model.r_max)]
    if 'sidechain_agnostic' in spec:
        def func(batch):
            batch['atom_types'] = batch['atom_types']*0
        data.preprocess.append(func)
    data.key_map = {"aa_type": "species", "R": "pos", "edge_attr": "bond_type"}
    
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
    layer_configs = addEdgeEmbedding(layer_configs, num_bond_types=3)

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
        layer_configs.layers = insertAfter(layer_configs.layers, 'node_attrs', time_encoding)
        time_embedding = ('time_embedding', {'module': GraphFeatureEmbedding,
                                             'graph': (f"{model.n_dim}x0e", 'time_encoding'),
                                             'node_in': (f"{model.node_attrs}", 'node_attrs'),
                                             'node_out': (f"{model.node_attrs}", 'node_attrs')
                                            })
    elif 'embed_time_in_edges' in spec:
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
    else:
        print('WARINING: Not using time embedding!')
        time_embedding = None
    if not time_embedding is None:
        layer_configs.layers = insertAfter(layer_configs.layers, 'time_encoding', time_embedding)
    
    if 'nll' in spec:
        layer_configs = addEnergyOutput(layer_configs, shifts=None, output_key='nll')
        layer_configs = addForceOutput(layer_configs, y='nll', gradients='score')
    else: # predict scores directly
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
    # the gradients are in fact -score, sign will be reversed in score_fn
    model.update(layer_configs)

    return config
