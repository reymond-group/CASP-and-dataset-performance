import functools
import copy
import math
import time
import logging
#from itertools import product
from io import StringIO
import networkx as nx
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

from .MCTSglobals import Stock, Policy, Log, Tree

#RdChiral
from . import rdc

class State(object):
    def __init__(self, mols, transforms, parentmol_ids):
        self.mols = mols
        self.hash = self.__calchash__()
        self.transforms = transforms 
        self.parentmol_ids = parentmol_ids
        self.stock = Stock() #Shared state
        self.policy = Policy()
        
    def __calchash__(self):
        """calculate the hash of based on the inchi's of the states molecules"""
        inchis = [AllChem.MolToInchiKey(mol) for mol in self.mols]
        inchis.sort()
        return hash(tuple(inchis))
   
    def __hash__(self):
        return self.hash
    
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    
    def __str__(self):
        """A string representation of the state (for print(state))"""
        string = "%s\n%s\n%s\n%s\nScore: %0.3F Solved: %s"%(str([Chem.MolToSmiles(mol) for mol in self.mols]),
                                                str(self.transforms),
                                                str(self.parentmol_ids),
                                                str(self.in_stock_list),
                                                self.score(),
                                                self.is_solved,
                                                         )
        return string
                                                     
       
    @property
    @functools.lru_cache(maxsize=1)
    def in_stock_list(self):
        """What compounds of the state are in stock
        
        returns list of booleans"""
        return [self.stock.mol_in_stock(mol) for mol in self.mols]
    
    @property    
    def is_solved(self):
        """If all compounds are in stock, the state is solved"""
        return np.all(self.in_stock_list)
    
    @property
    def is_terminal(self):
        """Node is terminal if it's solved or have a reached the maximum depth"""
        max_transforms = np.max(self.transforms)
        return (max_transforms > self.policy.max_transforms) or self.is_solved
    
    def squash_function(self, r, slope, yoffset, xoffset):
        """Squash function loosely adapted from a sigmoid function with parameters
        to modify and offset the shape
        
        :param r: the sign of the function, if the function goes from 1 to 0 or from 0 to 1
        :param slope: the slope of the midpoint
        :param xoffset: the offset of the midpoint along the x-axis
        :param yoffset: the offset of the curve along the y-axis"""
        return 1/(1+np.exp(slope*-(r-xoffset)))-yoffset
    
    @functools.lru_cache(maxsize=1)
    def score(self):
        """Returns the score of the current state"""
        #How many is in stock (number between 0 and 1)
        num_in_stock = np.sum(self.in_stock_list)
        #This fraction in stock, makes the algorithm artificially add stock compounds by cyclic addition/removal
        fraction_in_stock = num_in_stock/len(self.mols)
        
        
        #Max_transforms should be low
        max_transforms = np.max(self.transforms) #Max or mean? what is best
        #Squash function, 1 to 0, 0.5 around 4.
        max_transforms_score = self.squash_function(max_transforms, -1,0,4)
        
        #NB weights should sum to 1, to ensure that all 
        score4 = 0.95*fraction_in_stock + 0.05*max_transforms_score
        return score4

    def apply_reaction(self, mol, reactionsmarts):
        """Apply a reactions smarts to a molecule and return the products (reactanst for retro templates) + error code
        
        error code: 0, One reaction outcome 
        error code: -1, more that one reaction outcome
        error code: -2, no reaction outcome"""
        reaction = rdc.rdchiralReaction(reactionsmarts)
        rct = rdc.rdchiralReactants(Chem.MolToSmiles(mol))

        reactants = rdc.rdchiralRun(reaction, rct) #We split the "product" to the "reactants"

        #Turning rdchiral outcome into rdkit tuple of tuples to maintain compatibility
        outcomes = []
        for r in reactants:
            l = r.split('.')
            rct = tuple([Chem.MolFromSmiles(smi) for smi in l])
            outcomes.append(rct)
        reactants = tuple(outcomes)
        
        l = len(reactants)
        if l == 1:
            return reactants, 0
        if l > 1:
            return reactants, -1 
        else: #empty reaction
            return reactants, -2
        
    def take_action(self, action_key):
        """Apply the given action key to the internal state and return a new state
        
        The action_key is a tuple (Mols state_index + rule)  + number outcome to work on, buts that's ignored here
        """
        
        #Shallow copying should be enough
        mols = copy.copy(self.mols) 
        transforms = copy.copy(self.transforms)
        parentmol_ids = copy.copy(self.parentmol_ids)
        
        mol = mols.pop(action_key[0])
        num_transforms = transforms.pop(action_key[0])
        parentmol_id = parentmol_ids.pop(action_key[0])
        
        #Apply the reaction, get back possible outcomes
        reactants_list, score = self.apply_reaction(mol, action_key[1])

        #Reaction list can be empty
        if not reactants_list:
            logging.debug("Reactants_list empty %s, for mol %s and transformation %s"%( repr(reactants_list), Chem.MolToSmiles(mol), action_key[1]))
            #Return False (No State, if action not feasible)
            return False
        
        #Assert that a change happened, A reactants_list is a tuple of tuples with the RDKit mols (is a string of reactants joined by '.')
        if len(reactants_list) == 1:
            if len(reactants_list[0]) == 1:# && len(reactants_list[0][0])
                try:
                    if Chem.MolToInchiKey(mol) == Chem.MolToInchiKey(reactants_list[0][0]):
                        return False
                except:
                    logging.debug("Unsanitizable molecule in first tuple found using template %s on mol %s"%(action_key[1], Chem.MolToSmiles(mol)))
                    return False
                
        states = [] #List to hold states
        for i, reactants in enumerate(reactants_list):
            try:
                [AllChem.SanitizeMol(mol) for mol in reactants]
            except:
                logging.debug("Unsanitizable molecules found using template %s on mol %s"%(action_key[1], Chem.MolToSmiles(mol)))
                continue
            new_mols = mols + list(reactants)
            new_transforms = transforms + [ num_transforms + 1 ] * len(reactants)
            new_parentmol_ids = parentmol_ids + [action_key[0]] * len(reactants)
            states.append(State(new_mols, new_transforms, new_parentmol_ids))
        
        #Return a list with the new states with the new molecules
        return states


    def display(self):
        """Displays the states molecules for a jupyter notebook session"""
        #Sanitize molecules
        [Chem.SanitizeMol(mol) for mol in self.mols]
        #
        if hasattr(self.stock, 'stock_inchikeys'):
            #legends = [str(value) for value in self.in_stock_list]
            legends = ["" if v else "Not in Stock" for v in self.in_stock_list]
        else:
            legends = []
        display(Draw.MolsToGridImage(self.mols, molsPerRow=6, legends=legends))
    
    