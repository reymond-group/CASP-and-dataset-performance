"""
Developed for the purpose of template extraction and data curation from reaction datasets,
including:
1) Data filtering 
2) Data extraction
3) Template generation

Some components have been taken and modified from Coley and co-workers:
ACS Cent. Sci. 3, 5, 434-443
http://dx.doi.org/10.1021/acscentsci.7b00064
https://github.com/connorcoley/ochem_predict_nn

Authors:
Amol Thakkar, University of Bern and AstraZeneca
Esben Jannik Bjerrum, AstraZeneca
"""

import os
import sys
import timeit
import time
import random
import argparse
import faulthandler
import itertools
import pandas as pd
import numpy as np
from functools import partial

import multiprocessing 
from multiprocessing import Pool, Value, Lock, Manager

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdmolfiles, rdmolops

from template_utils.amol_utils_rdchiral import Reaction, Binarizer, Parsers

faulthandler.enable()

class Counter():
    """Threadsafe counter
    """
    def __init__(self, initval = 0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1
        
    def value(self):
        with self.lock:
            return self.val.value

def process_reaction_data(datafile, results_file, radius=1):
    """Processes reaction data from source, curates, filters, extracts templates, checks templates for validity,
    and assesses templates for selectivity.

    Filtered if:
        - More than three components in the reaction
        - More than one product in the reaction
        - The reaction record is incomplete
        - The reactants are products are equivalent
        - The reaction SMILES cannot be parsed by RDKit
        - The reaction is not atom-mapped
        - A reaction template (SMIRKS pattern) cannot be extracted
        - The template SMIRKS does not match a sub-structure in the from which it was extracted
        - The template SMIRKS cannot be validated in RDKit prior to application

    Parameters:
        datafile (str): absolute path to the csv file containing the raw data.
        results_file (str): absolute path to the csv file to output.
        radius (int): Specify the radius (number of atoms) away from the reaction centre to consider.
        stereochemistry (bool): Specify whether to consider stereochemistry.
    
    Returns:
        Writes sucessfully extracted reactions to a csv file with the following columns
        columns=["ID", "reaction_hash", "reactants", "products", "classification", "retro_template", "template_hash", "selectivity", "outcomes"]

        Writes unsucessful reactions to a csv file with the following columns
        columns=["ID", "rsmi", "reason"]

    ID represents all associated id's concatenated by ';'
    """
    p = Parsers()
    print('Parsing: {}'.format(datafile))
    reaction_data = p.import_USPTO_fast(datafile)

    def reaction_data_row_fn(row):
        try:
            reaction = Reaction(row["rsmi"], rid=row["ID"])
            if len(reaction.rsmi.split('>')) > 3:
                return pd.Series([row["ID"], None, row["rsmi"], None, None, None, None, None, None, True, "components > 3"], index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])
            elif len(reaction.product_list) > 1:
                return pd.Series([row["ID"], None, row["rsmi"], None, None, None, None, None, None, True, "products > 1"], index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])
            elif reaction.incomplete_reaction():
                return pd.Series([row["ID"], None, row["rsmi"], None, None, None, None, None, None, True, "incomplete"],
                                 index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])
            elif reaction.equivalent_reactant_product_set():
                return pd.Series([row["ID"], None, row["rsmi"], None, None, None, None, None, None, True, "reactants = products"],
                                 index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])
            elif reaction.generate_reaction_template(radius=radius) is None:
                print("template generation failure")
                return pd.Series([row["ID"], None, row["rsmi"], None, None, None, None, None, None, True, "template generation failure"],
                                 index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])
            elif reaction.validate_retro_template(reaction.retro_template) is None:
                print("template rdkit validation failed")
                return pd.Series([row["ID"], None, row["rsmi"], None, None, None, None, None, None, True, "template rdkit validation failed"],
                                 index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])
            elif reaction.check_retro_template_outcome(reaction.retro_template, reaction.products,
                                                       save_outcome=True) != 0:
                outcomes = len(reaction.retro_outcomes)
                assessment = reaction.assess_retro_template(reaction.retro_template, reaction.reactant_mol_list,
                                                            reaction.retro_outcomes)
                print("assessed")
                rinchi_hash = reaction.generate_concatenatedRInChI()
                row_list = [row["ID"],
                            rinchi_hash,
                            reaction.reactants,
                            reaction.products,
                            row["classification_id"],
                            reaction.retro_template,
                            reaction.hash_template(reaction.retro_template),
                            assessment,
                            outcomes,
                            False,
                            None]

                return pd.Series(row_list, index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])
            else:
                return pd.Series([None, None, None, None, None, None, None, None, None, True, "Unknown reason"],
                                 index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])
        except Exception as e:
            print(e)
            print('Template not extracted - Reaction is not suitable for processing or invalid')
            return pd.Series([row["ID"], None, row["rsmi"], None, None, None, None, None, None, True, "reaction invalid"],
                                 index=["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes", "failed", "reason"])

    dataset = reaction_data.apply(reaction_data_row_fn, axis=1)

    failed_df = dataset[dataset['failed']][["ID", "reactants", "reason"]].rename(columns={'reactants': 'rsmi'})
    dataset = dataset[~dataset['failed']][["ID", "reaction_hash", "reactants", "products", "classification",
                                                       "retro_template", "template_hash", "selectivity", "outcomes"]]
    total_extracted = len(dataset)
    invalid_template = len(failed_df[failed_df["reason"].isin(["template generation failure",
                                                         "template rdkit validation failed",
                                                         "reaction invalid"])])
    not_suitable = total_extracted - invalid_template

    print('creating dataframes.....')
    sys.stdout.flush()
    output = dataset.drop_duplicates(subset="reaction_hash")
    print('Dataframes created.....')
    sys.stdout.flush()

    w = False
    attempts = 10
    ctr = 0
    print('Writing {}'.format(datafile))
    print('Attempting write {}'.format(ctr))
    print("Total Reaction Count: " + str(total_reactions) + '\n')
    print("Total Reactions Extracted: " + str(total_extracted) + '\n')
    print("Reaction is not suitable for processing or invalid: " + str(not_suitable) + '\n')
    print("Template Validation Failure: " + str(invalid_template) + '\n')
    sys.stdout.flush()
    while ctr < attempts:
        ctr += 1
        try:
            # attempt to write
            print('Writing')
            output.to_csv(results_file + '.csv', mode='a', header=False)
            failed_df.to_csv(results_file + '_failed.csv', mode='a', header=False)
            print('Written, attempting to remove file')
            os.remove(datafile)
            print('File removed')
            w = True
            break
        except:
            print('Sleeping {}'.format(ctr - 1))
            time.sleep(float('.{}'.format(random.randint(1, 1000))))

    if not w:
        sys.stdout.flush()
        print("ERROR")
        output.to_csv(results_file + '_error.csv', mode='a', header=False)
        failed_df.to_csv(results_file + '_failed_error.csv', mode='a', header=False)
        os.remove(datafile)

    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract reaction templates, both canonical and retro, and validate')
    parser.add_argument('-d', '--data', type = str, default = None,
                        help = 'Specify the absolute path to the folder containing the datafiles')
    parser.add_argument('-o', '--out', type = str, default = None,
                        help = 'Specify the absolute path to the folder to which the results should be written \n' +
                        'if the folder does not exist, it will be created')
    parser.add_argument('-f', '--file', type = str, default = None,
                        help = 'Specify the filename for the output file')
    parser.add_argument('-r', '--radius', type = int, default = 1,
                        help = 'Specify the radius (number of atoms) away from the reaction centre to consider')
    # parser.add_argument('-s', '--stereo', type = bool, default = False,
    #                 help = 'Specify whether to consider stereochemistry')
    args = parser.parse_args()
    data_source = args.data
    if os.path.exists(args.out):
       pass
    else: 
        os.mkdir(args.out)

    data = ['/'.join([data_source, filename]) for filename in os.listdir(data_source)]
    output_file = '/'.join([args.out, args.file])
    cores = multiprocessing.cpu_count() 

    total_reactions = Counter(0)
    total_extracted = Counter(0)
    not_suitable = Counter(0)
    invalid_template = Counter(0)

    # df = pd.DataFrame(columns=["ID", "reaction_hash", "reactants", "products", "classification", "retro_template", "template_hash", "selectivity", "outcomes"])
    # df.to_csv(output_file + '.csv.gz', mode='a', compression="gzip", header=True)

    start = timeit.default_timer()

    with Pool(cores-2) as p:
         process_dummy = partial(process_reaction_data, results_file=output_file, radius=args.radius)
         p.map(process_dummy, data)


    stop = timeit.default_timer()
    total_time = stop - start
    # output running time in a nice format.
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)


    with open(output_file + '.txt', 'w') as stats:
        stats.write("-------- Rule Extraction Statistics --------" +'\n\n')
        stats.write("Total Reaction Count: " + str(total_reactions.value()) + '\n')
        stats.write("Total Reactions Extracted: " + str(total_extracted.value()) + '\n')
        stats.write("Reaction is not suitable for processing or invalid: " + str(not_suitable.value()) + '\n')
        stats.write("Template Validation Failure: " + str(invalid_template.value()) + '\n')
        stats.write("Number of cores: " + str(cores-2) + '\n')
        stats.write("Total Execution Time: %d:%d:%d.\n" % (hours, mins, secs))
    
    print("complete!")

