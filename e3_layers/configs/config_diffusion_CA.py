from functools import partial
from ..data import computeEdgeIndex, computeEdgeVector, Batch
from ml_collections.config_dict import ConfigDict
import ase
from .layer_configs import featureModel, addForceOutput, addEnergyOutput
from ..nn import PointwiseLinear, RadialBasisEncoding, Broadcast, RelativePositionEncoding, Concat, symmetricCutoff
from ..utils import insertAfter, replace, saveProtein, getScaler
import torch
import numpy as np

def masked2indexed(batch):
    data = {}
    id = torch.arange(start=0, end=batch['_n_nodes'].item())
    mask = batch['mask'].view(-1).bool()
    data['id'] = id[mask]
    data['_n_nodes'] = sum(mask).view(-1, 1)
    data['species'] = batch['species'][mask]
    data['chain_id'] = batch['chain_id'][mask]

    attrs = {'id': ('node', '1x0e')}
    for atom in ['N', 'CA', 'C', 'O']:
        data[atom] = batch[atom][mask]
    attrs.update(batch.attrs)
    return Batch(attrs, **data)

def crop(data, attrs, max_nodes):
    if data['_n_nodes'] <= max_nodes:
        return data, attrs
    x = np.random.randint(data['_n_nodes'])
    distance = data['CA'] - data['CA'][x]
    distance = torch.linalg.norm(distance, dim=-1)
    
    def binarySearch(r_min, r_max):
        if r_max - r_min < 0.5:
            return r_min
        mask = distance < (r_min+r_max)/2
        if sum(mask) > max_nodes:
            return binarySearch(r_min, (r_min+r_max)/2)
        elif sum(mask) < max_nodes:
            return binarySearch((r_min+r_max)/2, r_max)
        else:
            return (r_min+r_max)/2
    
    r = binarySearch(20, 70)
    mask = distance < r
    mask = mask.view(-1).bool()
    data['_n_nodes'] = sum(mask).view(-1, 1)
    
    data['id'] = data['id'][mask]
    data['species'] = data['species'][mask]
    data['chain_id'] = data['chain_id'][mask]
    for atom in ['N', 'CA', 'C', 'O']:
        data[atom] = data[atom][mask]
        
    for key in ['N', 'C', 'O']:
        data.pop(key)
        attrs.pop(key)
    return data, attrs
  
def criteria(data, edge_index):
    mask = (data['chain_id'][edge_index[0]] == data['chain_id'][edge_index[1]]).view(-1)
    mask = torch.logical_and(mask, abs(edge_index[0]-edge_index[1])<5)
    
    tmp = torch.rand((edge_index.shape[1],)).to(mask.device)
    mask = torch.logical_or(mask, tmp<0.02)
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
    #config.diffusion_keys = {'CA':3, 'C':3, 'O':3, 'N':3}
    config.diffusion_keys = {'CA':3}#, 'C':3, 'O':3, 'N':3}
    
    model.n_dim = 64
    model.l_max = 2
    model.r_max = 5.0 # this does not control the number of edges 
    model.num_layers = 8
    model.edge_radial = '32x0e'
    model.node_attrs = "32x0e"
    model.jit = True
    num_types = 21

    data.n_train = 0.9
    data.n_val = 0.1
    data.std = 25.83
    data.scaler = getScaler([('CA', ('shift', 'mean')), ('CA', ('scale',  1/data.std))])
    data.inverse_scaler = getScaler([('CA', ('scale', data.std))])
    data.train_val_split = "random"
    data.shuffle = True
 #   data.path = [f'/mnt/vepfs/hb/protein_new/{i}' for i in range(8)]
    data.path = f'/mnt/vepfs/hb/protein_new/0/pdb_0.hdf5'
    data.preprocess = [masked2indexed, partial(crop, max_nodes=384)]
    data.key_map = {}

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
    layer_configs.layers = replace(layer_configs.layers, 'edge_vector', ('edge_vector', partial(computeEdgeVector, key='CA')))
        
    relative_position = {
        "module": RadialBasisEncoding,
        "r_max": 150,
        "cutoff": symmetricCutoff,
        "trainable": True,
        'one_over_r': False,
    }
    relative_position = ('relative_position', {'module': RelativePositionEncoding,
                                              'segment': ('1x0e', 'chain_id'),
                                               'id': ('1x0e', 'id'),
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
    
    """
    concat = ('concat3', {'module': Concat,
              'node_features':(layer_configs.node_features, "node_features"),
              'C':(f"1x1o", "C"),
              'N':(f"1x1o", "N"),
              'O':(f"1x1o", "O"),
              'irreps_out':(layer_configs.node_features, "node_features")})
    layer_configs.layers = insertAfter(layer_configs.layers, 'layer3', concat)
    """
    
    for key in config.diffusion_keys:
        layer_configs.layers.append(
            (
                f"score_{key}",
                {
                    "module": PointwiseLinear,
                    "irreps_in": (features, "node_features"),
                    "irreps_out": (f"1x1o", f"score_{key}"),
                },
            )
        )
    layer_configs.layers = [('edge_index', partial(computeEdgeIndex, r_max=8.0/data.std, key='CA',
                                                   criteria=criteria))] + layer_configs.layers
    model.update(layer_configs)

    return config
