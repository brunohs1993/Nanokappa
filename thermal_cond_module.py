import numpy as np
import h5py
import scipy.constants as ct
import phonopy
from scipy.interpolate import LinearNDInterpolator, griddata

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
    
    def load_properties(self, filename):
        self.load_data(filename)
        self.load_frequency()
        self.convert_to_omega()
        self.load_points()
        self.load_weights()
        self.load_group_vel()
        self.load_temperature()
        self.load_heat_cap()

    def load_data(self, filename):
        self.data = h5py.File(filename,'r')

    def load_frequency(self):
        '''frequency shape = q-points X p-branches '''
        self.frequency = np.array(self.data['frequency']) # THz
    
    def convert_to_omega(self):
        '''omega shape = q-points X p-branches '''
        self.omega = self.frequency*2*ct.pi # Trad/s
    
    def load_points(self):
        '''q-points shape = q-points X reciprocal reduced coordinates '''
        self.q_points = np.array(self.data['qpoint'])
    
    def load_weights(self):
        '''weights shape = q_points '''
        self.weights = np.array(self.data['weight'])

    def load_temperature(self):
        '''temperature_array shape = temperatures '''
        self.temperature_array = np.array(self.data['temperature'])
    
    def load_group_vel(self):
        '''groupvel shape = q_points X p-branches X cartesian coordinates '''
        self.group_vel = np.array(self.data['group_velocity'])

    def load_heat_cap(self):
        '''heat_cap shape = temperatures X q-points X p-branches '''
        self.heat_cap = np.array(self.data['heat_capacity'])
        
    def initialise_positions(self, N, grid):
        self.positions = np.random.rand(grid.N_z, N, 3, grid.N_x, grid.N_y)

    def calculate_occupation(self, T, omega):
        return 1/(np.exp(ct.hbar*omega/ct.k * T)-1)

    def calculate_energy(self, n, omega):
        return ct.hbar * omega * (n + 0.5)
    
    def get_frequency(self, k, p):
        ''' k should be a 1d array with 3 coordinates of the q-points between 0 and 0.5; branch should be an integer'''
        k = self.adjust_k(k)
        return griddata(self.q_points, self.omega[:,p], k, method='nearest') # fix this 'nearest' later
    
    def adjust_k(self, k):
        ''' for structures with symmetry [100] = [010]  = [-100] = [0-10] and [001] = [00-1]'''
        k = np.abs(k)   # transfering to positive quadrant
        k = k % 1     # taking away extra periods

        if k[0]<k[1]:
            k = np.array([k[1], k[0], k[2]])

        return k

    def get_heat_cap(self, T, k, p):
        pass





        

