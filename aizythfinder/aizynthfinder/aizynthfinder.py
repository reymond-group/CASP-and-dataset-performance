#MCTS

import time

#Grow tf as needed
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

import numpy as np
from IPython.display import display, HTML

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
# Suppress RDKit errors due to incomplete template (e.g. aromatic non-ring atoms)
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

#Project imports
from .MCTSglobals import Stock, Policy, Log, Tree
from .node import Node
from .state import State

#RdChiral
import rdchiral as rdc

import hashlib

class AiZynthFinder(object):
    def __init__(self):
        #properties needed
        self.stock_files = {"stock": #Path to stock file as .hdf containing inchi keys}

        self.policy_files = ( #Path to policy model file as .hdf,
                             #Path to template library file as .hdf)
        
        self.log = Log()
        self.stock = Stock()
        self.tree = Tree()
        self.policy = Policy()

        #Default properties on policy 
        self.policy_defaults = {"C":1.4,
                        "max_transforms": 6,
                         "cutoff_cumulative": 0.995,
                         "cutoff_number":50,
                        }
        for key,value in self.policy_defaults.items():
            setattr(self.policy, key, value)

        
        self.time_limit = 120
        self.iteration_limit = 100
        
        self._target_mol = Chem.MolFromSmiles('')
        
    @property
    def target_smiles(self):
        return Chem.MolToSmiles(self._target_mol)
    
    @target_smiles.setter
    def target_smiles(self, smiles):
        self.target_mol = Chem.MolFromSmiles(smiles)

    @property
    def target_mol(self):
        return self._target_mol
        
    @target_mol.setter
    def target_mol(self, mol):
        self._target_mol = mol

    #TODO refactor name to reflect that its keys
    def load_stock(self, key="Addendum"):
        stockfile = self.stock_files[key]
        print("Loading Stockfile: %s"%stockfile)
        self.stock.load_stock(stockfile)
        
    def load_policy(self, policy=None):
        policy = (policy or self.policy_files) #Use default one if none provided
        print("Loading Policy: %s\nLoading Templates: %s"%policy)
        self.policy.load_policy(policy[0])
        self.policy.load_templates(policy[1])
    
    def prepare_tree(self):
        self.tree.initialize()
        print("Defining tree root: %s"%self.target_smiles)
        state = State(mols = [self._target_mol], transforms=[0], parentmol_ids=[0])
        self.root = Node(state=state)
        self.tree.add_root(self.root)

    def reinitialize_globals(self):
        """Function to reinitialize the globals"""
        print("TBD")
        
    def prepare_search(self):
        self.load_stock()
        self.load_policy()
        self.prepare_tree()
        print("All Set")
    
    def tree_search(self, stop_when_solved=False):
        time0 = time.time()
        i = 1
        print("Starting search")
        while ( time.time()-time0 < self.time_limit ) and ( i < self.iteration_limit + 1 ):
            #Select leaf
            print('.', end='')
            if i%80 == 0:
                print(' ')
            leaf = self.root.select_leaf()
            leaf.expand()
            #Make a roll-out
            while not leaf.is_terminal:
                child = leaf.get_promising_child() #
                if child: #child can be none, if the action is invalid
                    child.expand()
                    leaf = child
            leaf.backpropagate(leaf.state.score())
            #Stop early if solved
            if stop_when_solved:
                if leaf.state.is_solved:
                    return [time.time() - time0, 1] 
            i = i + 1
        print("Search completed")
        return [time.time() - time0, 0]
        
    def extract_route(self):
        """Function to find the best route(s)"""
        print("Analyzing_routes")
        nodes = list(self.tree.G.nodes)
        scores = [node.state.score() for node in nodes]
        idx = np.argmax(scores)
        best_node = nodes[idx]
        print("Best Score %0.2F"%best_node.state.score())
        return best_node.return_route_to_node(display=False)
    
    def route_to_text(self, route):
        """Convert a given route into text format
        
        Textformat is: 
        Reaction Template
        Reaction Smiles
        newline
        
        returns route in text format"""
        t_route = []
        for i in range(len(route[0])):
            t_route.append(route[0][i][1]) #The reaction key
            t_route.append(self.get_reactionsmiles(route[0][i], route[1][i+1])) #The reaction smiles
            t_route.append('')
        return '\n'.join(t_route)
    
    def get_reactionsmiles(self, reactionkey, node):
        """Get the reaction SMILES from a parent node and the reactionkey
        
        returns Reaction SMILES"""
        #Obtain the product of the reaction
        product = node.state.mols[reactionkey[0]]
        #Get the reactants by re-applying the reaction (and selection of the outcome)
        reaction = rdc.rdchiralReaction(reactionkey[1])
        rct = rdc.rdchiralReactants(Chem.MolToSmiles(product))
        reactants = rdc.rdchiralRun(reaction, rct)
        #Join together to create the reaction smiles and create the reaction from that
        react_smiles = "%s>>%s"%(
            reactants[0], #Join the reaction smiles around a "."
            Chem.MolToSmiles(product) #The product as SMILES
        )
        return react_smiles
    
    def display_route(self, route):
        """Displays a route in the jupyter output
        :param route: reverse route as found with node.return_route_to_node()
        """
        state =  route[1][0].state
        status = 'Solved' if state.is_solved else "Not Solved"
        
        display(HTML("<H2>%s"%status))
        display("Route Score: %0.3F"%state.score())
        display(HTML("<H2>Compounds to Procure"))
        state.display()
        display(HTML("<H2>Steps"))
        for i in range(len(route[0])):
            display(AllChem.ReactionFromSmarts(self.get_reactionsmiles(route[0][i], route[1][i+1]), useSmiles=True))
            
    def get_best_node(self):
        #Find best route
        nodes = list(self.tree.G.nodes)
        scores = [node.state.score() for node in nodes]
        idx = np.argmax(scores)
        best_node = nodes[idx]
        return best_node

    
if __name__ == "__main__":
    print("In Main")
    import argparse
    from defaults import defaults
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("target", help="The target molecule to search for (in SMILES notation)")
    parser.add_argument("-t","--time_limit", type=float, default=2, help="Maximum time to run the search algorithm (in minutes)") #In minutes
    parser.add_argument("-i","--iteration_limit", type=int, default=100, help="Maximum number of iterations for the search algorithm")

    #Policy arguments
    parser.add_argument("--policy_files", nargs=2, type=str, help="Policy files to load. Policy file, then template file")
    parser.add_argument("-C", type=float, default=1.4, help="C, the balancing between the exploitation and exploration (lower C is more exploitation)")
    parser.add_argument("--cutoff_number", type=int, default=50, help="The maximum number of templates to evaulate pr. compound pr. node.")
    parser.add_argument("--cutoff_cumulative", type=float, default=0.995, help='The maximum cummulative "probability" of actions consideret pr. compound pr. node')
    parser.add_argument("--max_transforms", type=int, default=6, help="The maximum number of conversions a given fragment can go through (Reaction tree depth)")
    parser.add_argument("--policy_priors", action="store_true", dest="use_priors", default=False, help="Optionally use the NN policy predicted values as priors")
    parser.add_argument("--prior_value",  default=0.5, help="Prior to set of untested actions (overruled by --policy_priors)")

    #Stock
    parser.add_argument("--stocks", nargs='*', type=str, default=["stock"], help="Stock selected from abbreviations %s"%defaults["stock_files"].keys())
    parser.add_argument("--stock_files", nargs='*', type=str, help="Stock files in HDF format")
    
    args = parser.parse_args()
    
    from defaults import defaults
    
    log = Log()
    s= Stock()
    
    finder = AiZynthFinder()
    
    #Search Algorithm
    
    #Stocks 
    if args.stocks:
        for stockabbr in args.stocks:
            finder.load_stock(stockabbr) #Need the defaults dictionary
    if args.stock_files:
        for stockfile in args.stock_files:
            finder.load_stock(stockfile)
        
    #Policy
    if args.policy_files:
        finder.load_policy(args.policy_files)
    else:
        finder.load_policy()
        
    #Set C from interface
    finder.policy.C = args.C
    
    finder.policy.max_transforms = args.max_transforms
    finder.policy.cutoff_number = args.cutoff_number
    finder.policy.cutoff_cumulative = args.cutoff_cumulative
    finder.policy.use_priors = args.use_priors
    finder.policy.prior_value = args.prior_value

    
    finder._target_mol = Chem.MolFromSmiles(args.target)
    finder.prepare_tree()
    print("All set, good to go")
    
    finder.time_limit = args.time_limit*60
    finder.iteration_limit = args.iteration_limit
    finder.tree_search()
    
    #Report Route
    route = finder.extract_route()
   
   
    state =  route[1][0].state
    status = 'Solved' if state.is_solved else "Not Solved"
        
    print("%s"%status)
    print("Route Score: %0.3F"%state.score())

    print(finder.route_to_text(route)) 


    
    


