import numpy as np
from numpy.random import default_rng
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

import os



class Optimiser:
    # Base class for all optimisation procedures, independent of the algorithm
    def __init__(self, params, obj, n_it, n_batch, n_sol = 1, data = None, folder = None):
        
        # params --------> a dict where keys are the parameters names (used for the database) and
        #                  values are iterables with lower and upper bounds
        # obj -----------> a dict where keys are the names of the objectives (used for the database) and
        #                  values are the type of problem for each: 'max' for maximisation, 'min' for minimisation, or a target value

        self.params_names = tuple(params.keys())  # name of the variables
        self.obj_names    = tuple(obj.keys())     # name of the objectives

        self.n_dims = len(self.params_names)      # number of dimensions
        self.n_obj  = len(self.obj_names)         # number of objectives

        self.problem      = [obj[i] for i in obj] # type of problem of objective
        self.lims     = np.array([params[i] for i in params]).astype(float) # space boundaries

        self.folder   = folder
        self.n_it     = int(n_it)                    # number of iterations
        self.n_sol    = int(n_sol)                   # number of solutions to return

        self.data = self.init_data(data)

    def init_data(self, data):
        if data is not None:
            try:
                self.data = pd.read_csv(data)
            except:
                print('ERROR: Invalid database file. Generate new solution database? y/n')
                flag = True
                while flag:
                    s = input()
                    if s == 'n':
                        print('Closing optimiser.')
                        quit()
                    elif s == 'y':
                        print(f'Generating new solution database.')
                        data = pd.DataFrame(columns = self.params_names + self.obj_names)
                        flag = False
                    else:
                        print('Wrong choice.')
        else:
            data = pd.DataFrame(columns = self.params_names + self.obj_names)
        
        return data

    def update_interpolator(self):

        x = np.array(self.data[self.params_names])
        y = np.array(self.data[self.obj_names   ])
        
        return LinearNDInterpolator(x, y)

    def generate_csv():
        pass
    
    def run_batch():
        pass

    def plot_scatter(self, param_x, param_y, subplot_kw, scatterplot_kw, param_z = None, figsize = (5, 5)):
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, **subplot_kw)
        sns.scatterplot(data = self.data, x = param_x, y = param_y, ax = ax, **scatterplot_kw)

        if param_z is None:
            filename = self.folder + f'/{param_x}_x_{param_y}.png'
        else:
            filename = self.folder + f'/{param_x}_x_{param_y}_x_{param_z}.png'

        plt.savefig(filename)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

class GeneticAlg(Optimiser):
    def __init__(self, n_cand, n_bits, refine = False,  seed = None):
        
        
        # n_cand --------> number of candidates in the population
        # n_it ----------> number of iterations of each batch
        # n_batch -------> number of batches to run
        # n_bits --------> an array of ints where each number is the number of bits of the parameter, in order of declaration 
        
        # data ----------> database of already known solutions
        # folder --------> path of the folder to where save the results
        # seed ----------> the seed for the random generator

        self.random_gen = default_rng(seed)

        self.n_cand   = int(n_cand)                  # number of candidates
        
        self.n_bits   = np.array(n_bits).astype(int) # number of bits to convert each dimension values to binary
        self.bit_lims = np.concatenate(([0], self.n_bits.cumsum())) # limits of the bits
        self.bin_base = np.concatenate([2**np.flip(np.arange(self.n_bits[i])) for i in range(self.n_dims)])
        self.n_steps  = 2**self.n_bits-1             # number of steps in each dimension
        
        self.n_dims   = self.lims.shape[1]           # dimensions of the search space

    def bin_to_float(self, x):
        x = x.reshape(-1, self.n_bits.sum())

        y = np.zeros((x.shape[0], self.n_dims))

        x = 2**x

        for d in range(self.n_dims):
            y[:, d] = x[:, self.bit_lims[d]:self.bit_lims[d+1]].sum(axis = 1)
        
        y /= self.n_steps            # normalise
        y *= self.lims.ptp(axis = 0) # multiply by range
        y += self.lims[0, :]         # translate to origin

        return y

    def generate_cand(self, n_cand):
        x = np.around(self.random_gen.rand(n_cand, self.n_bits)).astype(int) # bin candidates

        y = self.bin_to_float(x)
        f = self.evaluate_obj(y)
        return x, y, f

    def mutate(self, x, p, n_mut):
        
        r = self.random_gen.rand(x.shape[0]) # roll dice

        mutate = (r < p).nonzero()[0] # compare with mutation probability

        rm = self.random_gen.rand(x[mutate, :].shape) # roll another dice
        i = np.argsort(rm, axis = 1) < n_mut          # the n_mut smallest numbers will be mutated

        for i in mutate:                                                   # for each mutated candidate
            j = self.random_gen.choice(x.shape[0], n_mut, replace = False) # pick indices
            x[i, j] = np.where(x[i, j] == 0, 1, 0)                         # flip bits
        
        return x # return mutated candidates
    
    def cross(self, x, f, n_children, n_cuts = 1):
        
        n_children -= n_children % 2 # needs to be pair
        n_sec       = n_cuts + 1     # number of sections is one more than the number of cuts
        n_pairs     = n_children/2   # number of couples is the half of number of children

        p = f/f.sum() # calculate probability of crossing

        pairs = self.random_gen.choice(x.shape[0], (n_pairs, 2), p = p, replace = True) # trow random dice to choose crosses

        y = np.zeros((n_children, x.shape[1]))

        for ip, pair in enumerate(pairs):
            children = np.copy(x[pair, :])

            cuts = self.random_gen.choice(np.arange(1, x.shape[1], dtype = int), n_cuts, replace = False) # pick where to cut
            cuts = np.sort(cuts)                                                                          # sort them
            cuts = np.insert(cuts, [0, n_cuts], [0, x.shape[1]])                                          # and add first and last indices
            for sec in range(1, n_sec, 2):                                                                # for each second section
                children[cuts[sec]:cuts[sec+1], 0], children[cuts[sec]:cuts[sec+1], 1] = children[cuts[sec]:cuts[sec+1], 1], children[cuts[sec]:cuts[sec+1], 0].copy() # flip 

            y[[ip*2, ip*2+1], :] = np.copy(children) # save in the nest
        
        return y # return nest

    def kill(self, x, f, n_kill):

        i = np.argsort(f) # sorted by crescent order

        if self.problem == 'min':
            return x[i >= x.shape[0] - n_kill, :]
        elif self.problem == 'max':
            return x[i < n_kill, :]

    def remove_repetitions(self, x, f):
        n0 = x.shape[0] # original number of candidates
        x, i = np.unique(x, axis = 0, return_index = True)  # getting only unique solutions
        f = f[i]

        xn, fn = self.generate_cand(n0 - x.shape[0]) # generate random new to fill the gap

        x = np.vstack((x, xn))      # add to population
        f = np.concatenate((f, fn))

        return x, f
    
    def evaluate_obj(self, y, w = None):

        f = self.interpolator(y)

        mins = self.data[list(self.obj_names)].min(axis = 0)
        maxs = self.data[list(self.obj_names)].max(axis = 0)

        imin = self.problem == 'min'
        imax = self.problem == 'max'
        ival = [i for i in range(self.n_obj) if self.problem[i] not in ['min', 'max']]

        f[:, imin] = (f[:, imin] - mins[imin])/(maxs[imin] - mins[imin]) # normalising minimisation
        f[:, imax] = (maxs[imax] - f[:, imax])/(maxs[imax] - mins[imax]) # normalising maximisation

        f[:, ival] = np.absolute(f[:, ival] - self.problem[ival])
        f[:, ival] /= f[:, ival].max(axis = 0)

        if w is None:
            w = np.ones(self.n_obj)
        else:
            w = np.array(w)

        cost = ((f**2)*w).sum(axis = 1)

        return cost

    def run_timestep(self, x):

        y = self.bin_to_float(x)       # convert to float
        f = self.evaluate_obj(y) # evaluate f
        
        x = self.cross(x, f)           # cross over
        x = self.mutate(x)             # mutate

    def optimise(self):

        x, f = self.init_pop()

        pass
