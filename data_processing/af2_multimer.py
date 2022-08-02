import sys
import os
import json
sys.path.append('/home/hangrui/dpaie')
from unifold.fold.multimer_dataset import load_multimer_raw_feature
from tqdm import tqdm
import argparse
import numpy as np
from e3_layers.data import Batch

residue_atoms = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3',
            'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O',
            'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O'],
    'UNK': ['C', 'CA', 'N', 'O']
}


parser = argparse.ArgumentParser(description='')
parser.add_argument('--split', metavar='S', type=int,
                    help='')
parser.add_argument('--n_process', metavar='N', type=int,
                    help='')

args = parser.parse_args()
SPLIT = args.split
N_PROCESS = args.n_process
OUTPUT = '.'
print(f'{SPLIT}/{N_PROCESS}')

data_folder = '/mnt/data/data_0322'
file = f'{data_folder}/0512_multimer_pdb_info.json'
with open(file) as file:
    file = json.load(file)
    
failure = 0
lst = []
if SPLIT == 0:
    pbar = tqdm(file.keys())
else:
    pbar = file.keys()
for key in pbar:
    if not hash(key)%N_PROCESS == SPLIT:
        continue
    chains = [f'{key}_{item}' for item in file[key]['chains']]
    label_transforms = file[key]['opers']
    try:
        labels = load_multimer_raw_feature(chains, label_path=f'{data_folder}/pdb_labels', label_transform=label_transforms)
    except:
        failure += 1
        print(f'Failed {key}, number of failure cases {failure}.')
        continue
    
    data = {}
   # data['pdb_id'] = key
    for i, atom in enumerate(['N', 'CA', 'C', 'CB', 'O']):
        data[atom] = np.concatenate([item['all_atom_positions'][:, i] for item in labels], axis=0)
    data['species'] = np.concatenate([item['aatype_index'] for item in labels], axis=0)
    
    cnt = 0
    chain_id = np.zeros((data['species'].shape[0], 1), dtype=np.long)
    mask = np.zeros((data['species'].shape[0], 1), dtype=np.long)
    
    for i, item in enumerate(labels):
        chain_id[cnt: cnt+item['aatype_index'].shape[0]] = i
        tmp = np.logical_and(item['all_atom_mask'][:, 0:3].all(axis=1), item['all_atom_mask'][:, 4])
        mask[cnt: cnt+item['aatype_index'].shape[0], 0] = tmp
        cnt += item['aatype_index'].shape[0]
    
    data['chain_id'] = chain_id
    data['mask'] = mask
    lst.append(data)
    
path = os.path.join(OUTPUT, f'pdb_{SPLIT}.hdf5')
attrs = {}
for atom in ['N', 'CA', 'C', 'CB', 'O']:
    attrs[atom] = ('node', '1x1o')
attrs['mask'] = ('node', '1x0e')
attrs['species'] = ('node', '1x0e')
attrs['edge_attr'] = ('edge', '1x0e')
attrs['chain_id'] = ('node', '1x0e')
#attrs['pdb_id'] = ('graph', '1x0e')
attrs['_n_nodes'] = ('graph', '1x0e')
attrs['_n_edges'] = ('graph', '1x0e')
batch = Batch.from_data_list(lst, attrs)
batch.dumpHDF5(path)