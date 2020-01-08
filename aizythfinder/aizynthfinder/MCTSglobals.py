import functools
import copy
import math
import time
import logging
import json


#from itertools import product
from io import StringIO
import numpy as np
import pandas as pd

#Grow tf as needed
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

import keras
from keras.models import load_model

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import networkx as nx


class Log(object):
    """Logger object, for when we need to store special things like the Nodes where actions fail"""
    __shared_state = {} #Borg Pattern, class level properties shared amoung instances
    
    def __init__(self):
        self.__dict__ = self.__shared_state
        if not hasattr(self, "initialized"):
            self.reinitialize()
            
    def reinitialize(self):
        self.log_stream = StringIO()    
        logging.basicConfig(stream=self.log_stream, level=logging.INFO)
        self.logger = logging.getLogger()
        self.runaway_nodes = []
        self.initialized = True

    def log(self):
        print(self.log_stream.getvalue())
        
    def set_level(self, level="DEBUG"):
        self.logger.setLevel(level)


class Stock(object):
    __shared_state = {} #Borg Pattern, class level properties shared amoung instances
    
    def __init__(self):
        self.__dict__ = self.__shared_state
        
    def load_stock(self, stock_name, overwrite=False):
        if overwrite or not hasattr(self, "stock_df"):
            self.stock_df = pd.read_hdf(stock_name, 'table')
            self.stock_inchikeys = set(self.stock_df.inchi_key.values)
        else:
            df = pd.read_hdf(stock_name, 'table')
            self.stock_df = pd.concat([self.stock_df, df], sort=False)
            self.stock_inchikeys = set(self.stock_df.inchi_key.values)
            
    def reinitialize(self):
        del(self.stock_df)
        del(self.stock_inchikeys)
            
    #Simple check if compound is in stock
    def mol_in_stock(self, mol):
        inchi_key = Chem.MolToInchiKey( mol )
        return inchi_key in self.stock_inchikeys

    def smiles_in_stock(self, smiles):
        return self.mol_in_stock( Chem.MolFromSmiles(smiles))

    
class Policy(object):
    __shared_state = {} #Borg, class level properties shared among all instances
    
    def __init__(self):
        self.__dict__ = self.__shared_state
        if not hasattr(self, "initialized"):
            self.initialize()
            
    def initialize(self):
        self.cutoff_cumulative = 0.95
        self.cutoff_number = 10
        self.C = 1.4 #Exploitation/exploration balance, used by the Nodes
        self.max_transforms = 5 #Max direct transforms, used by states
        self.max_depth = 20 #Max depth in tree search, used by states
        self.default_prior = 0.5 #Default prior set by nodes, This should more probably be set for the expected average score of the terminal solutions in the tree
        self.use_prior = True #If true, will use the predictions from the NN as priors
        self.initialized = True
        
    def reinitialize(self):
        del(self.policy)
        del(self.templates_df)
        del(self.templates)
        self.initialize()
        
    def load_policy(self, modelfile):
        """
        Loads the policy network
        """
        top10_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=10)
        top10_acc.__name__ = 'top10_acc'

        top50_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=50)
        top50_acc.__name__ = 'top50_acc'

        self.policy = load_model(modelfile, custom_objects={'top10_acc': top10_acc, 'top50_acc': top50_acc})
        
        self.input_dims = int(self.policy.input.shape[1])

    def load_templates(self, templatefile, template_col="retro_template"):
        """ Load the template file"""
        self.templates_df = pd.read_hdf(templatefile, "table")
        self.templates = self.templates_df[template_col]

    def smiles_to_fp(self, smiles):
        """Convert a SMILES into a fingerprint usable for prediction"""
        return self.mol_to_fp( Chem.MolFromSmiles(smiles) )
        
    def mol_to_fp(self, mol):
        """Convert a RDKit Mol object into a fingerprint usable for prediction"""
        AllChem.SanitizeMol(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.input_dims)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr) #Sub ms conversion
        return  arr.reshape([1,self.input_dims])
 
    def predict_policy(self, mol):
        """Get predictions of the policy given a SMILES string of the product"""
        fp = self.mol_to_fp(mol)
        return np.array(self.policy.predict(fp)).flatten()
    
    def cutoff_prediction(self, predictions):
        """Get the idx of the top predictions
        
        cutoff using cumulated prediction or a max number
        
        self.cutoff_cumulative
        self.cutoff_number
        """
        #Get the top transforms, truncated at cummulated 99 or at most 10
        sortidx = np.argsort(predictions)[::-1]
        cumsum = np.cumsum(predictions[sortidx])
        maxidx = min(np.argmin(cumsum < self.cutoff_cumulative), self.cutoff_number)
        return sortidx[:maxidx + 1]

    def get_actions(self, state):
        """Get all the probable actions of a state, using the policy and given cutoff's"""
        possible_actions = []
        priors = []
        
        #Itereate through the molecules and find the possible moves
        for i, mol in enumerate(state.mols):
            #Only try to break down molecules that are not in stock
            if not state.stock.mol_in_stock(mol):
                all_transforms_prop = self.predict_policy(mol)
                probable_transforms_idx = self.cutoff_prediction(all_transforms_prop)
                possible_moves = self.templates[probable_transforms_idx]
                probs = all_transforms_prop[probable_transforms_idx]
                
                for move,prob in zip(possible_moves, probs):
                    possible_actions.append( (i, move, 0 ) ) #Using tuple makes it hashable and thus usable as key, 0 is default as assumes only one set of reactants
                    priors.append( prob )
        return possible_actions, priors
    
    def display_action(self, action_key):
        reaction = AllChem.ReactionFromSmarts(action_key[1])
        legend = "Selected molecule number %i"%(action_key[0]+1)
        print(legend)
        display(Draw.ReactionToImage(reaction, subImgSize=(100, 100), useSVG=False))
    
    def display_transformation(self, transformation):
        """Display transformation in jupyter notebook"""
        display(AllChem.ReactionFromSmarts(transformation))

        
class Tree(object):
    __shared_state = {} #Borg Pattern, class level properties shared amoung instances
    
    def __init__(self):
        self.__dict__ = self.__shared_state
        if not hasattr(self, "initialized"):
            self.initialize()
            
    def initialize(self):
        self.G = nx.DiGraph()
        #self.clear_route()
        self.initialized = True
        
    def add_root(self, node):
        self.root = node
        self.G.add_node(node)
        
    def add_child(self, parent, child, action_key):
        self.G.add_nodes_from([parent, child])
        self.G.add_edge(parent, child, action=action_key)
        
    def get_node_actions(self, node):
        return [child['action'] for child in self.G[node].values()]
    
    def is_action_instantiated(self, node, action_key):
        #Looks in the dictionaries of the children, if action is there
        return action_key in [child["action"] for child in self.G[node].values()]
    
    def get_parent(self, node):
        for parent, _ in self.G.in_edges(node):
            return parent #Early return only first
        
    def get_child(self, node, action_key):
        return  [child for child, edge in self.G[node].items() if edge["action"] == action_key][0]
    
    def get_action_from_nodes(self, parent, child):
        return self.G[parent][child]["action"]
    
    def is_root(self, node):
        return self.G.in_degree(node) == 0
    
    def find_route_to_node(self, node):
        """Find the route (list of states) going from the root to the query node"""
        route = nx.shortest_path(self.G, self.root, node)
        return route


class MolNode(object):
    """MolNode class for reaction networks in NetworkX
    
    As the __hash__ is default based on the InchiKey, there can be only one molecule node in the network
    can be switched off at instantiation time with the molhash variable or by deleting the .hash property
    
    """
    def __init__(self, rdmol, molhash=True):
        self.mol = rdmol
        if molhash:
            self.hash = hash(AllChem.MolToInchi(self.mol))
            
        self.type = "mol"
        
    def __hash__(self):
        """if custom mol hash, return that, otherwise use default python hash"""
        if hasattr(self, "hash"):
            return self.hash
        else:
            return super(MolNode,self).__hash__()
    
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    
    def __str__(self):
        return Chem.MolToSmiles(self.mol)
    
    def __repr__(self):
        return ("MolNode(Chem.MolFromSmiles(%s))"%Chem.MolToSmiles(self.mol))

        
class ReactionNode(object):
    """ReactionNode object for reaction networks in NetworkX
    
    The hash is the standard object hash, so there can be several of the same reaction in the tree
    
    """    
    
    def __init__(self, reaction):
        self.reaction = reaction
        self.type = "reaction"
        #Chem.ReactionFromSmarts(reactionSmarts)
        
    def __str__(self):
        reactants = self.reaction.GetReactants()
        products = self.reaction.GetProducts()
        forstring = []
        for i,mol in enumerate(reactants):
            if i > 0: forstring.append("+")
            forstring.append(Chem.MolToSmiles(mol))
            
        forstring.append("==>")
        for i, mol in enumerate(products):
            if i > 0: forstring.append("+")
            forstring.append(Chem.MolToSmiles(mol))
        
        return " ".join(forstring)
    
    def __repr__(self):
        return "ReactionNode(AllChem.ReactionFromSmarts('%s'))"%AllChem.ReactionToSmarts(self.reaction)
            
    