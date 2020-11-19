import numpy as np
import h5py

class Grid:
    def __init__(self):
        pass

    def __str__(self):
        try:
            self.L_x
        except NameError:
            print('Dimensions - L_x: Undefined, L_y: Undefined, L_z: Undefined')
        else:
            print('Dimensions - L_x: {:<6.2f}, L_y: {:<6.2f}, L_z: {:<6.2f}'.format(self.L_x, self.L_y, self.L_z))
        
        try:
            self.N_x
        except NameError:
            print('     Cells - N_x: Undefined, N_y: Undefined, N_z: Undefined')
        else:
            print('     Cells - N_x: {:<6d}, N_y: {:<6d}, N_z: {:<6d}'.format(self.N_x, self.N_y, self.N_z))

    def set_dimensions(self, L_x, L_y, L_z):
        self.L_x = L_x
        self.L_y = L_y
        self.L_z = L_z

    def set_cells(self, N_x, N_y, N_z):

        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z
        self.N_c = N_x*N_y*N_z

    def set_boundaries(self):

        self.dL_x = self.L_x/self.N_x
        self.dL_y = self.L_y/self.N_y
        self.dL_z = self.L_z/self.N_z

        self.cell_bounds_x = np.arange(self.N_x+1)*self.dL_x
        self.cell_bounds_y = np.arange(self.N_y+1)*self.dL_y
        self.cell_bounds_z = np.arange(self.N_z+1)*self.dL_z

    def set_temperature(self, T_c, T_f):
        self.T_c = T_c
        self.T_f = T_f
    
        self.T_mean = (self.T_f + self.T_c)/2

        self.T_diff = self.T_c - self.T_f

        self.temperature = np.ones( (self.N_z, self.N_x, self.N_y) )*self.T_f
        self.temperature[0,:,:] = self.T_c


class Phonon:
    def __init__(self):
        pass
    
    def load_data(self, filename):
        self.data = h5py.File(filename,'r')
    
    def get_frequency(self):
        self.frequency = np.array(self.data['frequency']) # THz
    
    def initialise_positions(self, N, grid):
        self.positions = np.random.rand(grid.N_z, N, 3, grid.N_x, grid.N_y)