import numpy as np
import matplotlib.pyplot as plt

import os

# Class to optimise geometry.

class Optimiser:
    def __init__(vars):
        pass

    def generate_csv():
        pass
    
    def optimise():
        pass

    def run_batch():
        pass


class GeneticAlg:
    def __init__(self, n_cand, n_it, lims, n_sol, n_bits, fit_space = False, refine = False, known_sol = None, interp_kw = None):
        
        # interp_space --> whether to use the known solutions to fit an ML model and optimse the models
        # refine --------> whether to reduce the search space around the best solution for the next batch

        self.n_cand   = int(n_cand)                  # number of candidates
        self.n_it     = int(n_it)                    # number of iterations
        self.lims     = np.array(lims).astype(float) # space boundaries
        self.n_bits   = np.array(n_bits).astype(int) # number of bits to convert each dimension values to binary
        self.bit_lims = np.concatenate(([0], self.n_bits.cumsum())) # limits of the bits
        self.bin_base = np.concatenate([2**np.flip(np.arange(self.n_bits[i])) for i in range(self.n_dims)])
        self.n_steps  = 2**self.n_bits-1             # number of steps in each dimension
        self.n_sol    = int(n_sol)                   # number of solutions to return
        self.n_dims   = self.lims.shape[1]           # dimensions of the search space

        if self.known_sol is not None:
            self.known_sol = known_sol
        else:
            self.knwon_sol = np.zeros((0, self.n_dims+1))
        
        self.init_pop(n_cand)

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

    # def float_to_bin(self, y):
        
    #     y -= self.lims[0, :]
    #     y /= self.lims.ptp(axis = 0)
    #     y *= self.n_steps
    #     y = np.around(y, decimals = 0).astype(int)

    #     return x

    def init_pop(self):
        self.pop = np.random.rand(self.n_cand, self.n_dims)*self.lims.ptp(axis = 0)+self.lims[0, :]
    

    def mutate(self, x):
        pass
    
    def cross(self, x, f):
        p = f/f.cumsum().max()

        r = np.random.choice(x.shape[0]*2, p = p)



        pass

    def evaluate_obj(self, y):
        return (y**2).sum()**0.5

    def plot_pareto(self):
        pass

    def plot_space(self, x, y = None):
        pass

    def run_timestep(self):

        y = self.bin_to_float(x)
        f = self.evaluate_obj(f_obj, y)

        

        x = self.cross(x, f)
        x = self.mutate(x)

    
    def optimise(self):
        



    




    

