import h5py
import os
import torch
import numpy as np
import ase
from e3nn import o3
from e3nn.o3 import Irreps
from e3_layers.data import Batch
import subprocess
import copy
from os.path import splitext
from pdbx.reader.PdbxReader import PdbxReader
from pdbx.reader.PdbxContainers import *

from tqdm import tqdm, trange

import argparse

INPUT = '/mnt/vepfs/hb/mmcif_gz'
CACHE = '/mnt/vepfs/hb/mmcif'
OUTPUT = '/mnt/vepfs/hb/protein_h5/'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--split', metavar='S', type=int,
                    help='')
parser.add_argument('--n_process', metavar='N', type=int,
                    help='')

args = parser.parse_args()
SPLIT = args.split
N_PROCESS = args.n_process
print(f'{SPLIT}/{N_PROCESS}')


codification = { "UNK" : 'X',
                 "ALA" : 'A',
                 "CYS" : 'C',
                 "ASP" : 'D',
                 "GLU" : 'E',
                 "PHE" : 'F',
                 "GLY" : 'G',
                 "HIS" : 'H',
                 "ILE" : 'I',
                 "LYS" : 'K',
                 "LEU" : 'L',
                 "MET" : 'M',
                 "ASN" : 'N',
                 "PYL" : 'O',
                 "PRO" : 'P',
                 "GLN" : 'Q',
                 "ARG" : 'R',
                 "SER" : 'S',
                 "THR" : 'T',
                 "SEC" : 'U',
                 "VAL" : 'V',
                 "TRP" : 'W',
                 "TYR" : 'Y' }

def sequence(block):
    # Retrieve the entity category table, which contains information that will be used in the FASTA header1
    entity = block.getObj("entity")
    # A container to hold each codified entity for optional comparison
    codifiedEntities = []

    # Track the current entity
    currentEntity = 1

    # Track the column width
    columnWidth = 0

    # Codification of the current entity
    codifiedEntity = ""

    # Holds non-mandatory entity attributes that could serve as FASTA header lines, ordered preferentially
    candidates = ["pdbx_description", "details", "type"]

    # Retrieve the entity_poly_seq category table, which containers the monomer sequences for entities2
    entity_poly_seq = block.getObj("entity_poly_seq")
    if entity_poly_seq is None:
        return None

    # Iterate over every row in entity_poly_seq, each containing an entity monomer
    for index in range(entity_poly_seq.getRowCount()) :

        # Obtain the ID of the entity monomer described by this row
        tempEntity = (int)(entity_poly_seq.getValue("entity_id", index))

        # If we are dealing with a new entity
        if currentEntity != tempEntity :

            # Store the current entity's FASTA codification
            codifiedEntities.append(codifiedEntity)

            # Set the new entity ID
            currentEntity = tempEntity

            columnWidth = 0
            codifiedEntity = ""

        # Retrieve the monomer stored in this row
        monomer = entity_poly_seq.getValue("mon_id", index)

        # If the monomer is an amino acid
        if len(monomer) == 3 :

            # If it's in the codification dictionary, add it
            if monomer in codification:
                codifiedEntity += codification[monomer]

            # Otherwise, use the default value "X"
            else :
                codifiedEntity += "X"

        # If it's a nucleic acid, there is nothing to codify
        else :
            codifiedEntity += monomer

        columnWidth += 1
    codifiedEntities.append(codifiedEntity)
    return codifiedEntities

def parseOperationExpression(expression) :
    operations = []
    stops = [ "," , "-" , ")" ]

    currentOp = ""
    i = 1

    # Iterate over the operation expression
    while i in range(1, len(expression) - 1):
        pos = i

        # Read an operation
        while expression[pos] not in stops and pos < len(expression) - 1 : 
            pos += 1    
        currentOp = expression[i : pos]

        # Handle single operations
        if expression[pos] != "-" :
            operations.append(currentOp)
            i = pos

        # Handle ranges
        if expression[pos] == "-" :
            pos += 1
            i = pos

            # Read in the range's end value
            while expression[pos] not in stops :
                pos += 1
            end = int(expression[i : pos])

            # Add all the operations in [currentOp, end]
            for val in range((int(currentOp)), end + 1) :
                operations.append(str(val))
            i = pos
        i += 1
    return operations

def prepareOperation(oper_list, op1index, op2index) :
    # Prepare matrices for operations 1 & 2
    op1 = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1]]
    op2 = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1]]

    # Fill the operation matrices for operations 1 & 2
    for i in range(3) :
        op1[i][3] = float(oper_list.getValue("vector[" + str(i + 1) + "]", op1index))

        if (op2index != -1) :
            op2[i][3] = float(oper_list.getValue("vector[" + str(i + 1) + "]", op2index))
        for j in range(3) :
            op1[i][j] = float(oper_list.getValue("matrix[" + str(i + 1) + "][" + str(j + 1) + "]", op1index))
            if (op2index != -1) :
                op2[i][j] = float(oper_list.getValue("matrix[" + str(i + 1) + "][" + str(j + 1) + "]", op2index))
    
    # Handles non-Cartesian product expressions
    if (op2index == -1) :
        return op1

    operation = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1]]

    # Handles Cartesian product expressions (4x4 matrix multiplication)
    sum = 0.0
    for row in range(4) :
        for col in range(4) :
            sum = 0.0
            for r in range(4) :
                sum += (op1[row][r] * op2[r][col])
            operation[row][col] = sum
    return operation

def structure(block):
    # Retrieve the atom_site category table, which delineates constituent atoms1
    atom_site = block.getObj("atom_site")

    # Make a reference copy of the atom_site category table
    atom_site_ref = copy.copy(atom_site)

    # Retrieve the pdbx_struct_assembly_gen category table, which details the generation of each macromolecular assembly2
    assembly_gen = block.getObj("pdbx_struct_assembly_gen")
    if assembly_gen is None:
        return None
    # Retrieve the pdbx_struct_oper_list category table, which details translation and rotation 
    # operations required to generate/transform assembly coordinates3
    oper_list = block.getObj("pdbx_struct_oper_list")

    attributes = atom_site_ref.getAttributeList()

    # Create a CIF file for every assembly specified in pdbx_struct_assembly_gen
    for index in range(assembly_gen.getRowCount()) :

        # Create a new atom_site category table for this assembly
        lst = [{}]

        # Lists to hold the individual operations
        oper = []
        oper2 = []

        # Keep track of the current atom and model number
        atomNum = 0
        modelNum = 0

        # Retrieve the assembly_id attribute value for this assembly
        assemblyId = assembly_gen.getValue("assembly_id", index)

        # Retrieve the operation expression for this assembly from the oper_expression attribute	
        oper_expression = assembly_gen.getValue("oper_expression", index)

        # Count the number of left parentheses in the operation expression
        parenCount = oper_expression.count("(")

        # Handles one operation assemblies (e.g., "1")
        if parenCount == 0 : oper.append(oper_expression)

        # Handles multiple operation assemblies, no Cartesian products (e.g., "(1-5)")
        if parenCount == 1 : oper.extend(parseOperationExpression(oper_expression))

        # Handles Cartesian product expressions (e.g., "(X0)(1-60)")
        if parenCount == 2 :
            # Break the expression into two parenthesized expressions and parse them
            temp = oper_expression.find(")")
            oper.extend(parseOperationExpression(oper_expression[0:temp+1]))
            oper2.extend(parseOperationExpression(oper_expression[temp+1:]))

        # Retrieve the asym_id_list, which indicates which atoms to apply the operations to
        asym_id_list = assembly_gen.getValue("asym_id_list", index)

        temp = (1 > len(oper2)) and 1 or len(oper2)

        # For every operation in the first parenthesized list
        for op1 in oper :
            # Find the index of the current operation in the oper_list category table
            op1index = 0
            for row in range(oper_list.getRowCount()) : 
                if oper_list.getValue("id", row) == op1 : 
                    op1index = row
                    break

            # For every operation in the second parenthesized list (if there is one)
            for i in range(temp) :
                # Find the index of the second operation in the oper_list category table
                op2index = -1
                if (oper2) :
                    for row in range(oper_list.getRowCount()) :
                        if oper_list.getValue("id", row) == oper2[i] :
                            op2index = row
                            break

                # Prepare the operation matrix
                operation = prepareOperation(oper_list, op1index, op2index)

                # Iterate over every atom in the atom_site reference table
                for r in range(atom_site_ref.getRowCount()) :

                    # If the asym_id of the current atom is not in the asym_id list, skip to the next atom
                    if (asym_id_list.find(atom_site_ref.getValue("label_asym_id", r)) == -1) :
                        continue

                    # Retrieve the atom's row from the atom_site reference table
                    atom = atom_site_ref.getFullRow(r)

                    # Add this row to the atom_site table for this assembly
                    for s in range(len(attributes)):
                        lst[atomNum][attributes[s]] = atom[s]

                    # Update the atom number and model number for this row
                    lst[atomNum]['id'] = str(atomNum)
                    lst[atomNum]["pdbx_PDB_model_num"] = str(modelNum)

                    # Determine and set the new coordinates
                    coords = [float(atom[10]), float(atom[11]), float(atom[12]), 1.0]
                    sum = 0.0
                    xyz = ['x', 'y', 'z']
                    for a in range(3) :
                        sum = 0.0
                        for b in range(4) :
                            sum += (operation[a][b] * coords[b])
                        lst[atomNum]["Cartn_" + xyz[a]] = "%.3f" % sum
                    atomNum += 1
                    lst.append({})
                modelNum += 1

        return lst[:-1]
      

aa_ids = {codification[key]:i for i, key in enumerate(codification.keys())}
def name2id(x):
    return aa_ids[x]

proteins = []
for root, dirs, files in os.walk(INPUT):
    files = tqdm(files, mininterval=60)
    files.set_description(str(SPLIT))
    for filename in files:
        tmp = filename.split('.')
        if not len(tmp) == 3:
            continue
        name, _, ext = tmp
        if not ext == 'gz':
            continue
        if not hash(name)%N_PROCESS == SPLIT:
            continue
        
        with open(os.path.join(CACHE, filename[:-3]), "w") as file:
            p = subprocess.run(["gunzip", "-c", os.path.join(root, filename)], stdout=file)
        
        try:
            with open(os.path.join(CACHE, filename[:-3])) as file:
                reader = PdbxReader(file)
                data = []
                reader.read(data)
                block = data[0]
                os.remove(os.path.join(CACHE, filename[:-3]))
        except:
            continue
            
        try:
            struct = structure(block)
        except:
            continue
        if struct is None:
            continue
        seq = sequence(block)
        if seq is None:
            continue
        struct = [item for item in struct if item['group_PDB'] =='ATOM'\
                  and 'label_atom_id' in item and item['label_atom_id'] == 'CA']
        
        cnt = 0
        aa_type = []
        cumsum = [0]
        
        try:  # exception: seq == '0' instead of a letter
            for i, chain in enumerate(seq):
                for j, res in enumerate(seq[i]):
                    aa_type.append(name2id(seq[i][j]))
                    cnt += 1
                cumsum.append(cnt)
        except:
            continue

        chain_id = np.zeros((cnt, 1), dtype='int64')
        for i in range(len(cumsum)-1):
            chain_id[cumsum[i]:cumsum[i+1]] = i
        
        pos = np.zeros((len(aa_type), 3), dtype='float32')
        pos_mask = [1]*len(aa_type)
        for i, ca in enumerate(struct):
            idx = cumsum[int(ca['label_entity_id'])-1] + int(ca['label_seq_id'])-1
            pos_mask[idx] = 0
            pos[idx] = float(ca['Cartn_x']), float(ca['Cartn_y']), float(ca['Cartn_z'])

        file = {'_n_nodes': cnt, 'aa_type': np.array(aa_type), 'pos': pos, 'pos_mask': np.array(pos_mask)}
        file.update({'chain_id': chain_id})
        proteins.append(file)
        
path = os.path.join(OUTPUT, f'pdb_{SPLIT}.hdf5')
attrs = {}
attrs['pos'] = ('node', '1x1o')
attrs['pos_mask'] = ('node', '1x0e')
attrs['aa_type'] = ('node', '1x0e')
attrs['_n_nodes'] = ('graph', '1x0e')
attrs['chain_id'] = ('node', '1x0e')
batch = Batch.from_data_list(proteins, attrs)
batch.dumpHDF5(path)