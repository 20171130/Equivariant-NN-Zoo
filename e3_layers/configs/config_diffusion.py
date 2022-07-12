from functools import partial
from ..data import computeEdgeIndex
from ml_collections.config_dict import ConfigDict
import ase
from .layer_configs import featureModel, addForceOutput, addEnergyOutput
from ..nn import PointwiseLinear, RadialBasisEncoding, Broadcast, OneHotEncoding, Concat
from ..utils import insertAfter, saveMol


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

    config.optimizer_name = "Adam"
    config.lr_scheduler_name = "ReduceLROnPlateau"
    config.lr_scheduler_patience = 1
    config.lr_scheduler_factor = 0.8
    config.grad_clid_norm = 1.0
    config.grad_acc = 1
    config.saveMol = saveMol

    model.n_dim = 32
    model.l_max = 2
    model.r_max = 5.0 
    model.num_layers = 4
    model.edge_radial = '8x0e'
    model.node_attrs = "16x0e"
    num_types = 18

    data.n_train = 120000
    data.n_val = 10831
    data.std = 1.4 # such that the variance of pos is roughly 3
    data.train_val_split = "random"
    data.shuffle = True
    data.path = "qm9_edge.hdf5"
    data.type_names = list(ase.atom.atomic_numbers.keys())[:num_types]
    if not 'gcn' in spec:
        data.preprocess = [partial(computeEdgeIndex, r_max=9999)]
    data.key_map = {"Z": "species", "R": "pos", "U": "total_energy", "edge_attr": "bond_type"}
    
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
    
    
    bond_onehot = ('bond_onehot', {'module': OneHotEncoding,
              'num_types': 4,
              'irreps_in':('1x0e', "bond_type"),
              'irreps_out':('4x0e', "bonde_type_onehot")})
    concat = ('concat1', {'module': Concat,
              'bondtype':('4x0e', "bonde_type_onehot"),
              'edge_radial':(model.edge_radial, "edge_radial"),
              'irreps_out' : (model.edge_radial, "edge_radial")})
    layer_configs.layers = insertAfter(layer_configs.layers, 'radial_basis', bond_onehot)
    layer_configs.layers = insertAfter(layer_configs.layers, 'bond_onehot', concat)

    time_encoding = ('time_encoding', {
        "module": RadialBasisEncoding,
        "r_max": 1.0,
        "trainable": True,
        "polynomial_degree": 6,
        "real": ("1x0e", "t"),
        'one_over_r': False,
        "irreps_out": (f"{model.n_dim}x0e", "time_encoding"),
    })
    layer_configs.layers = insertAfter(layer_configs.layers, 'node_features', time_encoding)
    graph2node = ('graph2node', {'module': Broadcast, 
                                 'irreps_in': (f"{model.n_dim}x0e", "time_encoding"),
                                 'irreps_out': (f"{model.n_dim}x0e", "time_encoding"),
                                 'to': 'node'})
    layer_configs.layers = insertAfter(layer_configs.layers, 'time_encoding', graph2node)
    concat = ('concat2', {'module': Concat,
              'node_attrs':(model.node_attrs, "node_attrs"),
              'time_encoding':(f"{model.n_dim}x0e", "time_encoding"),
              'irreps_out':(model.node_attrs, "node_attrs")})
    layer_configs.layers = insertAfter(layer_configs.layers, 'graph2node', concat)
    
    
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
