import sys, os, timeit, argparse
import pandas as pd
import numpy as np
import h5py
import tables
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from scipy import sparse

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdmolfiles, rdmolops
from rdkit.DataStructs import cDataStructs

import argparse

import itertools

def smiles_to_ecfp(product, size=2048):
    """Converts a single SMILES into an ECFP4

    Parameters:
        product (str): The SMILES string corresponing to the product/molecule of choice.
        size (int): Size (dimensions) of the ECFP4 vector to be calculated.
    
    Returns:
        ecfp4 (arr): An n dimensional ECFP4 vector of the molecule.
    """
    mol = Chem.MolFromSmiles(product)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    arr = np.zeros((0,), dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(ecfp, arr)
    return arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset with templates extracted into Train/Val/Test, and precompute fingerprints')
    parser.add_argument('-fp', '--fingerprint_size', type = int, default = 2048,
                        help = 'Specify the size of the ECFP4 fingerprint to compute (e.g. 2048)')
    parser.add_argument('-m', '--min_templates', type = int, default = 3,
                        help = 'Specify the number of times a template should occur to be included in the final dataset. Default is 3 times.')
    parser.add_argument('-o', '--out', type = str, default = None,
                        help = 'Specify the absolute path to the folder to which the results should be written \n' +
                        'if the folder does not exist, it will be created')
    parser.add_argument('-f', '--file', type = str, default = None,
                        help = 'Specify the filename for the output file')
    parser.add_argument('-i', '--input', type = str, default = None,
                        help = 'Specify the absolute path to the input file')

    args = parser.parse_args()
    data_source = args.input
    if os.path.exists(args.out):
       pass
    else: 
        os.mkdir(args.out)

FPSIZE = int(args.fingerprint_size)
MIN_TEMPLATES = int(args.min_templates)
output_name = args.file

data_path = args.input
out_dir = args.out

full_data = pd.read_csv(data_path, index_col=False, header=None, 
            names=["index", "ID", "reaction_hash", "reactants", "products", "classification", "retro_template", "template_hash", "selectivity", "outcomes"])
full_data = full_data.drop_duplicates(subset='reaction_hash')
template_group = full_data.groupby('template_hash')
template_group = template_group.size().sort_values(ascending=False)
min_N_templates = template_group[template_group >= MIN_TEMPLATES].index
dataset = full_data[full_data['template_hash'].isin(min_N_templates)]

template_labels = LabelEncoder()
dataset['template_code'] = template_labels.fit_transform(dataset['template_hash'])
dataset.to_csv(out_dir + output_name + '_template_library.csv', mode='w', header=False, index=False)

print('Dataset filtered, Generating Labels...')

# Compute labels 
lb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=True)
labels = lb.fit_transform(dataset['template_hash']) #CSR matrix of all vectorised labels

print('Labels Generated, Splitting...')

train_labels, test_labels = train_test_split(labels, test_size = 0.1, random_state = 42, shuffle = True)
val_labels, test_labels = train_test_split(test_labels, test_size = 0.5, random_state = 42, shuffle = True)
sparse.save_npz(out_dir + output_name + '_training_labels.npz', train_labels, compressed = True)
sparse.save_npz(out_dir + output_name + '_validation_labels.npz', val_labels, compressed = True)
sparse.save_npz(out_dir + output_name + '_testing_labels.npz', test_labels, compressed = True)

print('Labels Split, Generating Inputs...')

# Compute inputs
inputs = [x for x in dataset['products'].apply(smiles_to_ecfp, size=FPSIZE).values]
inputs = sparse.lil_matrix(inputs)
inputs = inputs.tocsr()

print('Inputs Generated, Splitting...')

train_inputs, test_inputs = train_test_split(inputs, test_size = 0.1, random_state = 42, shuffle = True)
val_inputs, test_inputs = train_test_split(test_inputs, test_size = 0.5, random_state = 42, shuffle = True)
sparse.save_npz(out_dir + output_name + '_training_inputs.npz', train_inputs, compressed = True)
sparse.save_npz(out_dir + output_name + '_validation_inputs.npz', val_inputs, compressed = True)
sparse.save_npz(out_dir + output_name + '_testing_inputs.npz', test_inputs, compressed = True)

print('Inputs Split, Splitting Full Dataset...')

train, test = train_test_split(dataset, test_size = 0.1, random_state = 42, shuffle = True)
val, test = train_test_split(test, test_size = 0.5, random_state = 42, shuffle = True)
train.to_csv(out_dir + output_name + '_training.csv', mode='w', header=False, index=False)
val.to_csv(out_dir + output_name + '_validation.csv', mode='w', header=False, index=False)
test.to_csv(out_dir + output_name + '_testing.csv', mode='w', header=False, index=False)

print('Complete')