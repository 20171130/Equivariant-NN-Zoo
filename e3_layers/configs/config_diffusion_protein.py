from functools import partial
from ..data import computeEdgeIndex
from ml_collections.config_dict import ConfigDict
import ase
from .layer_configs import featureModel, addForceOutput, addEnergyOutput
from ..nn import PointwiseLinear, RadialBasisEncoding, Broadcast, RelativePositionEncoding, Concat, symmetricCutoff
from ..utils import insertAfter, saveProtein
import torch

def posMask(batch):
    batch['pos'] = batch['pos'] + batch['pos_mask']*torch.randn(batch['pos'].shape) # a cluster at the center
    batch['species'] = batch['species'] + batch['pos_mask']*23
    return batch
def maskSpecies(batch):
    batch['species'] = batch['species']*0
    return batch
def criteria(data, edge_index):
    mask = (data['chain_id'][edge_index[0]] == data['chain_id'][edge_index[1]]).view(-1)
    mask = torch.logical_and(mask, abs(edge_index[0]-edge_index[1])<5)
    
    tmp = torch.rand((edge_index.shape[1],)).to(mask.device)
    mask = torch.logical_or(mask, tmp<0.03)
    return mask

def get_config(spec=''):
    config = ConfigDict()
    data, model = ConfigDict(), ConfigDict()
    config.data_config = data
    config.model_config = model

    config.learning_rate = 1e-2
    config.batch_size = 4
    config.grad_acc = 4
    
    config.use_ema = True
    config.ema_decay = 0.99
    config.config_spec = spec
    config.ema_use_num_updates = True

    config.optimizer_name = "Adam"
    config.lr_scheduler_name = "ReduceLROnPlateau"
    config.lr_scheduler_patience = 1
    config.lr_scheduler_factor = 0.8
    config.grad_clid_norm = 1.
    config.saveMol = saveProtein
    
    model.n_dim = 64
    model.l_max = 2
    model.r_max = 5.0 
    model.num_layers = 8
    model.edge_radial = '32x0e'
    model.node_attrs = "32x0e"
    model.jit = True
    num_types = 23*2

    data.n_train = 0.9
    data.n_val = 0.1
    data.std = 25.83
    data.train_val_split = "random"
    data.shuffle = True
    data.path = [f'/mnt/vepfs/hb/protein_small/{i}' for i in range(7)]
    data.preprocess = []
    if 'sidechain_agnostic' in spec:
        data.preprocess.append(maskSpecies)
        
    data.preprocess.append(posMask)
    
    data.key_map = {"aa_type": "species", "R": "pos"}

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
        avg_num_neighbors=100,
        normalize=True
    )
    
    relative_position = {
        "module": RadialBasisEncoding,
        "r_max": 150,
        "cutoff": symmetricCutoff,
        "trainable": True,
        'one_over_r': False,
    }
    relative_position = ('relative_position', {'module': RelativePositionEncoding,
                                              'segment': ('1x0e', 'chain_id'),
                                              'irreps_out':  (model.edge_radial, "rel_pos_embed"),
                                              'radial_encoding': relative_position})
    concat = ('concat1', {'module': Concat,
              'rel_pos':(model.edge_radial, "rel_pos_embed"),
              'edge_radial':(model.edge_radial, "edge_radial"),
              'irreps_out' : (model.edge_radial, "edge_radial")})
    layer_configs.layers = [relative_position] + layer_configs.layers
    layer_configs.layers = insertAfter(layer_configs.layers, 'radial_basis', concat)
    
    time_encoding = ('time_encoding', {
        "module": RadialBasisEncoding,
        "r_max": 1.0,
        "trainable": True,
        "irreps_in": ("1x0e", "t"),
        'one_over_r': False,
        "irreps_out": (f"{model.n_dim}x0e", "time_encoding"),
    })
    layer_configs.layers = insertAfter(layer_configs.layers, 'embedding', time_encoding)
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
    layer_configs.layers = [('edge_index', partial(computeEdgeIndex, r_max=8.0/data.std, criteria=criteria))] + layer_configs.layers
    model.update(layer_configs)

    return config
