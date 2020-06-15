# Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain

Thank you for your interest in the source code complementing our publication in Chemical Science:
https://pubs.rsc.org/en/content/articlelanding/2019/sc/c9sc04944d#!divAbstract

The code released is that used in the above publication, and is an alpha version. 

**We have now refactored and released the AiZynthfinder package which supercedes this repository for running searches for synthetic routes. Templates are still extracted as per the code found in this repository and RDChiral.**
AiZynthfinder:
https://github.com/MolecularAI/aizynthfinder

## Install conda environment

The conda enivronment casp_env is available to install from a .yml or .txt requirements file

`conda create -f casp_env.yml`

or 

`conda create --name casp_env --file casp_env.txt`

## Install rdchiral 

Install or clone the rdchiral package by Coley et.al. from the following repository:

https://github.com/connorcoley/rdchiral

the corresponding publication can be found at:

https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00286

## Prepare data for Template Extraction
Split data into individual files and place into a new folder.
    
`mv DATAFILE.csv new_folder`

In the new folder split the file as in the example below:

`split -l 100 DATAFILE.csv`

generated files will be 100 lines. It can be increased but this is dependant on how much memory you have available.

## Template Extraction

Call help for template extraction:
`python Template_Extraction_and_Validation.py -h`

Extract Templates as in the example below:
`python Template_Extraction_and_Validation.py -d /path/to/data/folder -o /path/to/data/output_folder -f output_filename -r 1`

### Example Notebooks

The following notebooks allow you to experiment with the reaction processing class and extraction methods:

*ReactionClass_amol_utils_example.ipynb*

*TemplateExtraction_example.ipynb*

## Preprocess Data for Training Policy

`python Split_and_Precompute.py -fp 2048 -m 3 -o /path/to/data/output_folder -f filename -i /path/to/data/folder/filename.csv`

This will generate the template library as a .csv file and yield a series of .npz files containin the training, validation, and test data as sparse matrices containging precomputed fingerprints. .csv files of the training, validation, and test data will also be saved. 

## Train Policy
Train the policy using the bash script, ensuring the paths are changed to reflect those in your file system:

`sh ./policy_uspto_precomputed.sh`

The training script is entitled:

*policy_precomputed.py* 

## Convert template library to .hdf format
Run Jupyter notebook for conversion and specify the path to the file

*Templates_to_hdf.ipynb*

## Ensure you have a stock set of compounds
A full stock set of commercially available compounds has not been provided, however can be downloaded from the Zinc database.

http://zinc.docking.org/catalogs/

### Procedure for stock set file formatting

Conversion of the stock set to the required format is required and is illustrated in the example:

*convert_stock_example.ipynb*

# Tree Search 

To use the tree search you will need to have installed:
- the conda environment
- rdchiral
- aizynthfinder

aizynthfinder can be installed by navigating to the aizynthfinder directory and running.

`python setup.py install` 

You will additionally require:
- A pre-trained policy
- The template library as a .hdf file (see previous)
- A set of stock compounds as a .hdf file (see previous)

You should then be able to experiment with the notebook:

*Test_PolicyGuidedTreeSearch.ipynb*
