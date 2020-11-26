import numpy as np
import h5py
import scipy.constants as ct
import phonopy
from scipy.interpolate import LinearNDInterpolator, griddata

class Geometry:
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

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

class Phonon:
    def __init__(self):
        pass
    
    def load_properties(self, hdf5_file, poscar_file):
        self.load_data(hdf5_file)
        self.load_frequency()
        self.convert_to_omega()
        self.load_points()
        self.load_weights()
        self.load_group_vel()
        self.load_temperature()
        self.load_heat_cap()

    def load_data(self, hdf5_file,):
        self.data = h5py.File(hdf5_file,'r')

    def load_frequency(self):
        '''frequency shape = q-points X p-branches '''
        self.frequency = np.array(self.data['frequency']) # THz
    
    def convert_to_omega(self):
        '''omega shape = q-points X p-branches '''
        self.omega = self.frequency*2*ct.pi*10**12 # rad/s
    
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
    
    def load_lattice_vectors(self, poscar_file):
        f = open(poscar_file, 'r')
        lines = [line.strip() for line in f.readlines()]
        
        self.a = [s.strip() for s in lines[2].split(' ')]
        self.a = np.array([i for i in self.a if i!=''], dtype=float)

        self.b = [s.strip() for s in lines[3].split(' ')]
        self.b = np.array([i for i in self.b if i!=''], dtype=float)

        self.c = [s.strip() for s in lines[4].split(' ')]
        self.c = np.array([i for i in self.c if i!=''], dtype=float)

        f.close()
    
    def load_atoms(self, poscar_file):
        self.atoms = []
        f = open(poscar_file, 'r')
        lines = [line.strip() for line in f.readlines()]

        at = [s.strip() for s in lines[5].split(' ')]
        at = np.array([i for i in at if i!=''])

        qt = [s.strip() for s in lines[6].split(' ')]
        qt = np.array([i for i in qt if i!=''])

        atoms = []
        for i in range(len(atoms)):
            atoms += [{'name': at[i], 'quantity':qt[i]}]

        f.close()
    
    def initialise_positions(self, N, grid):
        self.positions = np.random.rand(grid.N_z, N, 3, grid.N_x, grid.N_y)

    def calculate_occupation(self, T, omega):
        return 1/(np.exp(ct.hbar*omega/ct.k * T)-1)

    def calculate_energy(self, T, omega):
        n = self.calculate_occupation(T, omega)
        return ct.hbar * omega * (n + 0.5) / ct.eV # eV

    







        

