import functools
import copy
import math
import logging
import time
import os
#Fix hash seed, so that hashing of nodes is deterministic
os.environ["PYTHONHASHSEED"] = "42"

#from itertools import product
from io import StringIO
import numpy as np
import pandas as pd
import networkx as nx

#Grow tf as needed
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

import keras
from keras.models import load_model
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

from .MCTSglobals import Policy, Log, Tree


class Node(object):
    def __init__(self, state=None, parent=None, prior=None):
        self.state = state #Game state should be deepcopied when creating child
        self.policy = Policy()
        self.tree = Tree()
        self.log = Log()
        
        if prior:
            self.total_value = prior
        else:
            self.total_value = self.policy.default_prior  # float, collected rewards from roll-outs
            
        self.number_visits = 1  # int, times visited. Creation counts as one visit
        self.C = self.policy.C #Exploitation/exploration balance
    
    @property
    def unexpandable(self):
        """Node may be unexpandable if children's values are all negative
        They are set to be negative if retro reaction do not results in any molecules"""
        if hasattr(self, "children_values"):
            try:
                return np.max(self.children_values) < 0
            except:
                #self.children values is empty, possibly arising from root state being in stock
                return False
        else:
            return False
    
    @property    
    def is_root(self):
        """Boolean showing if node is root"""
        return self.tree.is_root(self)
    
    def root_is_solved(self):
        """Boolean showing if node is root and the state is solved"""
        return self.tree.is_root(self) and self.state.is_solved
    
    @property
    def is_expanded(self):
        """If some children has been added to the node"""
        return bool(self.tree.G[self])
    
    
    @property
    def is_terminal(self):
        """Node is terminal if its unexpandable, or the internal state is terminal (solved)"""
        return self.unexpandable or self.state.is_terminal
    
    
    def Q(self):  # returns float
        """Returns the average score of the node"""
        return self.total_value / ( self.number_visits )

    
    def children_Q(self):
        """Returns the average scores of the children nodes"""
        return np.array(self.children_values) / np.array(self.children_times_visited)


    def U(self):  # returns float
        """Returns the uncertainty of the node. The uncertainty is based on the number of visits as well as the total
        number of visits to all siblings of the parent"""
        parent = self.tree.get_parent(self)
        if parent:
            u = self.C * math.sqrt(
                2*np.log(parent.number_visits) / (self.number_visits)
                )
            return  u
        else:
            print("No parent available")
            return 0.0
        
    def children_U(self):
        """Returns the uncertainty of the childrens nodes. The uncertainty is based on the number of visits as well as the total
        number of visits to all siblings"""
        u = self.policy.C * np.sqrt( 
            2*np.log(np.sum(self.children_times_visited)) /
                     np.array(self.children_times_visited)
        )
        return u
    
    def expand(self):
        """Expand the node. Expansion is the process of creating the children of the node"""
        #Do not expand if solved
        if not self.state.is_solved: 
            #Calculate the possible actions, fill the child_info lists
            self.children_actions, self.children_priors = self.policy.get_actions(self.state) #Actions by default only assumes 1 set of reactants
            self.children_times_visited = [1] * len(self.children_actions)
            if self.policy.use_prior:
                self.children_values = copy.copy(self.children_priors) #When using the NN policy as a prior, allow updating the values without affecting the prior
            else:
                self.children_values = [self.policy.default_prior] * len(self.children_actions)

    def select_child(self, idx):
        """Select the child with the given index. If the child is not instantiated as a node, 
        it will be instantiated. Instantiation from a yet untested action_key may result in several outcomes, and the children list
        
        :param idx: the index of the child to select"""
        #Get the action corresponding to the selected idx
        action_key = self.children_actions[idx]
        
        #Instantiate is action is not instantiated on self
        if self.tree.is_action_instantiated(self, action_key):
            #Action already existed, return child corresponding to that action_key
            return self.tree.get_child(self, action_key)
        else:
            #Run reaction
            states = self.state.take_action(action_key) #Default reaction outcome is ignored
            children = []
            if states:
                    for i, state in enumerate(states):
                            action_key = tuple(list(action_key[0:2]) + [i])
                            child = Node(state=state, prior=self.children_priors[idx])
                            self.tree.add_child(self, child, action_key)
                            children.append(child)
                            #If there's more than one outcome, the lists need be expanded
                            if i > 0:
                                #Add the new actions to the action list and update Value and 
                                self.children_actions.append(action_key)
                                self.children_priors.append(self.children_priors[idx])
                                self.children_values.append(self.children_values[idx])
                                self.children_times_visited.append(self.children_times_visited[idx])
            if len(children) == 0: 
                #Action is invalid
                self.children_values[idx] = -1e6 # Set a high negative score to never select it again
                return None 
            elif len(children) == 1:
                return children[0]
            else:
                return np.random.choice(children)
            
        
    def get_promising_child(self):
        """Return the child with the currently highest Q+U
        
        returns Node"""
        scores = self.children_Q() + self.children_U()
        indices = np.where(scores == scores.max())[0]
        index = np.random.choice(indices)
        
        child = self.select_child(index)
        
        if child:
            self.log.last_node = child
        
        return child
    
    def get_best_child(self):
        """Get the child with the currently highest Q
        
        returns Node"""
        idx = np.argmax(self.children_Q())
        child = self.select_child(idx)
 
        if child:
            self.log.last_node = child
        
        return child

    
    def select_leaf(self):
        """Recursively traverse the tree selecting the most promising child at each step until leaf node returned
        
        returns leaf Node"""
        
        current = self
        limit = 0
        while current.is_expanded and not current.state.is_solved:
            promising_child = current.get_promising_child()
            #If there's a faulty action, a None child can be found.
            if promising_child:
                current = promising_child
            #There's a risk of a run-away here, if there's only faulty actions, they will all get set to -1e6 and the loop will run forever
            limit = limit + 1
            if limit > 20:
                self.log.logger.debug("Runaway process identified in node with state %s"%current.state)
                self.log.runaway_nodes.append(current)
                break
        return current

    def backpropagate(self, value_estimate):
        """Backpropagate the value estimate and update all nodes in the current route
        
        :params value_estimate: The score value to backpropagate"""
        current = self
        while not current.is_root:
            parent = self.tree.get_parent(current)
            action_key = self.tree.get_action_from_nodes(parent,current)
            idx = np.argmax([action == action_key for action in parent.children_actions])
            #Update the value estimates and number visits on the parent level
            parent.children_times_visited[idx] = parent.children_times_visited[idx] + 1 
            parent.children_values[idx] = parent.children_values[idx] + value_estimate
            current = parent
    
    def best_action(self):
        """Selects and returns the action with the current highest Q"""
        index = np.argmax(self.children_Q())
        return self.children_actions[index] 
    
    def select_best_forward_route(self, display=True):
        """Select the best forward route, at each step select the child with highest Q
        
        :param displat: Boolean, if True the reaction will be displayed"""
        route = []
        current = self
        if display:
            current.state.display()
        #Traverse the tree selecting best child.
        while current.is_expanded:
            action = current.best_action()
            route.append(action)
            current = current.get_best_child()
            if display:
                current.policy.display_action(action)
                current.state.display()
        return route, current
    
    def return_route_to_node(self, display=False):
        """Return the route to the node. 
        
        :param display: Boolean, if true the reactions will be shown
        """
        route = []
        nodes = []
        current = self
        
        while current is not None:
            #Identify action that led to this node, assume only one match as matches object
            if display: current.state.display()
            parent = self.tree.get_parent(current)
            if parent is not None:
                action = self.tree.G[parent][current]["action"]
                route.append(action)
                if display:
                    current.policy.display_action(action)
            nodes.append(current)
            current = parent
        return route, nodes
            
     
