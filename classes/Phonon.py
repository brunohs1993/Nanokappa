# calculations
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.spatial.transform import Rotation as rot

# hdf5 files
import h5py

# crystal structure
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure

# other
import sys
import os
import pickle

from classes.Constants import Constants

import matplotlib.pyplot as plt

np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=np.nan)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class defining phonon propeties from given arguments.

#   TO DO
#
#   - Get the wave vector in cartesian coordinates.
#   - Think in how to get the direct or cartesian coordinates of atoms positions from POSCAR file.
#   - 

class Phonon(Constants):    
    ''' Class to get phonon properties and manipulate them. '''
    def __init__(self, arguments, mat_index):
        super(Phonon, self).__init__()
        self.args = arguments
        self.mat_index = mat_index
        self.name = self.args.mat_names[mat_index]
        
        self.mat_folder = self.args.mat_folder[mat_index]
        if not self.mat_folder.endswith('\\'):
            self.mat_folder += '\\'
                    
    def load_properties(self):
        '''Initialise all phonon properties from input files.'''
        
        self.args.pickled_mat = [int(i) for i in self.args.pickled_mat]
        self.T_reference = self.args.reference_temp[0]

        #### we have to discuss for the following ####
        poscar_file = self.mat_folder + self.args.poscar_file[self.mat_index]
        unitcell, _ = read_crystal_structure(poscar_file, interface_mode='vasp')
        lattice = unitcell.get_cell()   # vectors as lines
        reciprocal_lattice = np.linalg.inv(lattice)*2*np.pi # vectors as columns

        phonon = Phonopy(unitcell,
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    primitive_matrix=[[1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 1]])

        symmetry_obj = phonon.primitive_symmetry
        rotations = symmetry_obj.get_reciprocal_operations()
        #############################################

        self.volume_unitcell = unitcell.get_volume()   # angstrom³
        
        print('Reading hdf file...')
        hdf_file = self.mat_folder + self.args.hdf_file[self.mat_index]

        self.load_hdf_data(hdf_file)

        self.load_q_points()
        self.load_weights()

        print('Expanding frequency to FBZ...')
        self.load_frequency()
        qpoints_FBZ,frequency=expand_FBZ(0,self.weights,self.q_points,self.frequency,0,rotations,reciprocal_lattice)
        self.frequency= frequency
        self.convert_to_omega()
        
        print('Expanding group velocity to FBZ...')
        self.load_group_vel()
        qpoints_FBZ,group_vel=expand_FBZ(0,self.weights,self.q_points,self.group_vel,1,rotations,reciprocal_lattice)
        self.group_vel=np.around(group_vel, decimals = 8)

        self.load_temperature()
        # print('Expanding heat capacity to FBZ...')  # Do we need heat capacity? For now it is not used anywhere...  
        # self.load_heat_cap()
        # qpoints_FBZ,heat_cap=expand_FBZ(1,self.weights,self.q_points,self.heat_cap,0,rotations,reciprocal_lattice)
        # self.heat_cap=heat_cap

        print('Expanding gamma to FBZ...')
        self.load_gamma()
        qpoints_FBZ,gamma=expand_FBZ(1,self.weights,self.q_points,self.gamma,0,rotations,reciprocal_lattice)
        self.gamma=gamma

        self.q_points=qpoints_FBZ
        # self.weights = np.ones((len(self.q_points[:,0]),))
        self.weights = np.ones(self.q_points.shape[0])
        
        self.number_of_qpoints = self.q_points.shape[0]
        self.number_of_branches = self.frequency.shape[1]

        self.number_of_modes = self.number_of_qpoints*self.number_of_branches
        
        self.inactive_modes_mask  = np.all(self.group_vel == 0, axis = 2)
        self.number_of_inactive_modes = self.inactive_modes_mask.sum()

        self.number_of_active_modes = self.number_of_modes - self.number_of_inactive_modes

        # [[a*1, b*1, c*1], [a*2, b*2, c*2], [a*3, b*3, c*3]]
        self.reciprocal_lattice = np.around(reciprocal_lattice, decimals = 6)

        self.unique_modes = np.stack(np.meshgrid( np.arange(self.number_of_qpoints), np.arange(self.number_of_branches) ), axis = -1 ).reshape(-1, 2).astype(int)

        # self.get_wavevectors_all_directions()
        self.get_wavevectors()

        if len(self.args.mat_rotation) > 0:
            self.rotate_crystal()

        print('To describe phonons:')
        print(' nq=',self.number_of_qpoints)
        print(' nb=',self.number_of_branches)
        print(' number of modes=', self.number_of_modes)

        print('Interpolating lifetime...')
        self.calculate_lifetime()

        print('Generating T = f(E)...')
        self.zero_point = self.calculate_zeropoint()
        self.calculate_reference(self.T_reference)
        self.initialise_temperature_function()

        print('Generating p_ref = f(T)...')
        self.initialise_ref_momentum_function()

        self.initialise_density_of_states()

        print('Pickling...')
        self.pickle_material()

        print('Material initialisation done!')

    def load_hdf_data(self, hdf_file):
        ''' Get all data from hdf file.
            Module documentation: https://docs.h5py.org/en/stable/'''

        self.data_hdf = h5py.File(hdf_file,'r')
        self.data_mesh = np.array(self.data_hdf['mesh']) # mesh discretisation points in each dimension
    
    def load_frequency(self):
        '''frequency shape = q-points X p-branches '''
        self.frequency = np.array(self.data_hdf['frequency']) # THz
        self.frequency = np.where(self.frequency<0, 0, self.frequency)

    def convert_to_omega(self):
        '''omega shape = q-points X p-branches '''
        self.omega = self.frequency*2*self.pi # THz*rad

    def load_q_points(self):
        '''q-points shape = q-points X reciprocal reduced coordinates '''
        self.q_points = np.array(self.data_hdf['qpoint'])   # reduced reciprocal coordinates
    
    def load_weights(self):
        '''weights shape = q_points '''
        self.weights = np.array(self.data_hdf['weight'])

    def load_temperature(self):
        '''temperature_array shape = temperatures '''
        self.temperature_array = np.array(self.data_hdf['temperature']) # K
    
    def load_group_vel(self):
        '''groupvel shape = q_points X p-branches X cartesian coordinates '''
        self.group_vel = np.array(self.data_hdf['group_velocity'])  # THz * angstrom

    # def load_heat_cap(self):
    #     '''heat_cap shape = temperatures X q-points X p-branches '''
    #     self.heat_cap = np.array(self.data_hdf['heat_capacity'])    # eV/K
    
    def get_wavevectors(self):

        self.wavevectors = self.q_to_k(self.q_points)
        
        self.norm_wavevectors = np.linalg.norm(self.wavevectors, axis = 1)

    def k_to_q(self, k):
        # convert wave vectors to q-points in the first brillouin zone
        a = np.linalg.inv(self.reciprocal_lattice)  # transformation vector
        
        q = np.dot(k, a.T)

        # bring all points to the first brillouin zone
        q = q % 1

        # adjust for nearest interpolation
        q = np.where(q >=  1-1/(2*self.data_mesh), q-1, q)
        q = np.where(q <    -1/(2*self.data_mesh), q+1, q)

        return q
    
    def q_to_k(self, q):
        # convert q-points to wave vectors in the first brillouin zone
        k = np.dot(q, self.reciprocal_lattice.T)

        return k        

    def rotate_crystal(self):
        '''Rotates the orientation of the crystal in relation to the geometry axis.'''
        print('Rotating crystal...')        
        
        # initialize angles and order
        self.rotation_angles = None
        self.rotation_order  = None

        n_mats = len(self.args.mat_names) # how many materials are listed

        for i in range(n_mats): # for each material
            if int(self.args.mat_rotation[i*5]) == self.mat_index: # if it is the material of this object

                self.rotation_angles = np.array(self.args.mat_rotation[i*5+1:i*5+4]).astype(float) # get rotation angles

                self.rotation_order  = self.args.mat_rotation[i*4+4]                 # get rotation order
        
        if self.rotation_angles is not None: # if the rotation was defined for this material
            
            # generate rotation object
            R = rot.from_euler(self.rotation_order, self.rotation_angles, degrees = True)

            self.wavevectors = R.apply(self.wavevectors) # rotate k

            for i in range(self.number_of_branches): # for each branch
                self.group_vel[:, i, :] = R.apply(self.group_vel[:, i, :]) # rotate v_g
        
        self.group_vel = np.around(self.group_vel, decimals = 6)

    def load_gamma(self):
        '''gamma = temperatures X q-pointsX p-branches'''
        self.gamma = np.array(self.data_hdf['gamma'])   # THz
        self.gamma = np.where(self.gamma > 0 , self.gamma, -1)

    def calculate_lifetime(self):
        '''lifetime = temperatures X q-pointsX p-branches'''

        self.lifetime = np.where( self.gamma>0, 1/( 2*2*np.pi*self.gamma), 0) # ps

        q_array = np.arange(self.number_of_qpoints )
        j_array = np.arange(self.number_of_branches)
        T = self.temperature_array
        tau = self.lifetime

        self.lifetime_function = RegularGridInterpolator((T, q_array, j_array), tau)

    def calculate_occupation(self, T, omega, reference = False):
        '''Calculate the Bose-Einstein occupation number of a given frequency at temperature T.'''

        flag = (T>0) & (omega>0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            if reference:
                occupation = np.where(~flag, 0, 1/( np.exp( omega*self.hbar/ (T*self.kb) ) - 1) - 1/( np.exp( omega*self.hbar/ (self.T_reference*self.kb) ) - 1) )
            elif not reference:
                occupation = np.where(~flag, 0, 1/( np.exp( omega*self.hbar/ (T*self.kb) ) - 1) )
        
        return occupation

    def calculate_energy(self, T, omega, reference = False):
        '''Energy of a mode given T and omega due to its occupation (ignoring zero-point energy).'''
        n = self.calculate_occupation(T, omega, reference = reference)
        return self.hbar*omega*n    # eV
    
    def calculate_crystal_energy(self, T):
        '''Calculates the energy density at a given temperature for the crystal.'''

        T = np.array(T)             # ensuring right type
        T = T.reshape( (-1, 1, 1) ) # ensuring right dimensionality

        crystal_energy = (self.calculate_energy(T, self.omega)*~self.inactive_modes_mask).sum( axis = (1, 2) )  # eV - energy sum of all modes
        crystal_energy = self.normalise_to_density(crystal_energy)                  # eV / a³ - normalising to density
        crystal_energy += self.zero_point                                           # eV / a³ - adding zero-point energy density

        return crystal_energy

    def calculate_zeropoint(self):
        '''Calculates the minimum possible energy density the system can have, called zero-point energy density'''
        
        zero = self.hbar*self.omega.sum()/2      # 1/2 sum of all modes
        zero = self.normalise_to_density(zero)   # normalising to density

        return zero
    
    def calculate_reference(self, T):
        '''Calculate occupation and energy at reference temperatures for each modes, as well as the 
           equilibrium momentum function at different temperatures.'''

        self.reference_occupation = self.calculate_occupation(T, self.omega)        # shape = q_points x branches
        self.reference_energy = self.calculate_crystal_energy(self.T_reference)     # float, eV/a³

    def calculate_crystal_momentum(self, T):
        
        T = np.array(T)             # ensuring right type
        T = T.reshape( (-1, 1, 1) ) # ensuring right dimensionality

        n = (self.calculate_occupation(T, self.omega)*~self.inactive_modes_mask).sum(axis = 2).T # (Q, T)

        momentum = (self.hbar*self.wavevectors.reshape(3, -1, 1)*n).sum(axis = 1).T # (T, D)

        return momentum
    
    def initialise_ref_momentum_function(self):

        # interval
        T_min = self.temperature_array.min()
        T_max = self.temperature_array.max()

        # Temperature array
        dT = 0.1
        T_array = np.arange(T_min, T_max+dT, dT)

        # momentum array
        momentum_array = np.array( list( map(self.calculate_crystal_momentum, T_array) ) ).squeeze()

        self.reference_momentum_function_x = interp1d( T_array.reshape(-1), momentum_array[:, 0], kind = 'linear', fill_value = '' )
        self.reference_momentum_function_y = interp1d( T_array.reshape(-1), momentum_array[:, 1], kind = 'linear', fill_value = '' )
        self.reference_momentum_function_z = interp1d( T_array.reshape(-1), momentum_array[:, 2], kind = 'linear', fill_value = '' )

    def initialise_temperature_function(self):
        '''Calculate an array of energy density levels and initialises the function to recalculate T = f(E)'''

        # interval
        T_min = self.temperature_array.min()
        T_max = self.temperature_array.max()

        # Temperature array
        dT = 0.1
        T_array = np.arange(T_min, T_max+dT, dT)

        # Energy array
        self.energy_array = np.array( list( map(self.calculate_crystal_energy, T_array) ) ).reshape(-1)

        # Interpolating
        self.temperature_function = interp1d( self.energy_array, T_array, kind = 'linear', fill_value = '' )

    def normalise_to_density(self, x):
        '''Defines conversion from energy to energy density here so it is easy to change.'''
        
        # Q = N_uc              - Number of available q-points is equal to the number of unitcells in the crystal
        # 
        # N_uc = V_s / V_uc     - The number of unit cells is how many of them fit in the volume of the solid
        #
        # V_s = Q x V_uc        - Hence the volume of the equivalent solid can be estimated by Q x V_uc

        return x/(self.number_of_qpoints*self.volume_unitcell)  # unit / angstrom³

    def initialise_density_of_states(self, bins = 100):
        self.g_counts, self.g_bins = np.histogram(self.omega, bins = bins)
    
    def g(self, omega):
        i = np.searchsorted(self.g_bins, omega, side = 'left')
        return self.g_counts[i-1]

    def pickle_material(self):
        
        del(self.data_hdf)
        
        files = os.listdir(self.mat_folder)
        
        short_fname = self.name + '_{:d}'.format(int(self.T_reference)) + '.pickle'

        pickled_file = self.mat_folder + short_fname

        if short_fname not in files: # if it is not already pickled
            pickle_out = open(pickled_file, 'wb')
            pickle.dump(self, pickle_out)
            pickle_out.close()
            print('Material ' + self.name + ' pickled!')
        else:
            print('Material already pickled! Check your files.')
    
    def open_pickled_material(self):
        
        self.T_reference = self.args.reference_temp[0]

        short_fname = self.name + '_{:d}'.format(int(self.T_reference)) + '.pickle'

        pickled_file = self.mat_folder + short_fname

        files = os.listdir(self.mat_folder)

        if short_fname in files:
            pickle_in = open(pickled_file, 'rb')
            return pickle.load(pickle_in)
        else:
            print('Material not yet pickled!')
            self.load_properties()
            self.open_pickled_material()

def expand_FBZ(axis,weight,qpoints,tensor,rank,rotations,reciprocal_lattice):
        # expand tensor from IBZ to BZ
        # q is in the direction "axis" in tensor, and #rank cartesian coordinates the last

       for i,q in enumerate(qpoints):
           q_in_BZ = np.mod(q,1.0)
           tq=np.take(tensor,i,axis=axis)

           l_q=[]
           l_t=[]

           for r in rotations:
               rot_q = np.dot(r, q_in_BZ)
               l_q.append(rot_q)

               r_cart = np.dot(reciprocal_lattice, np.dot(r, np.linalg.inv(reciprocal_lattice)))

               if rank == 0:
                  l_t.append(tq)
               elif rank == 1:
                  rot_t=np.dot(r_cart, tq.T).T
                  l_t.append(rot_t)
               else:
                  sys.exit("error in expand_FBZ: not coded")

           star_mul=np.array(l_q, dtype='double', order='C')
           star_mulv=np.array(l_t, dtype='double', order='C')

           star_mul=np.mod(star_mul,1.0)
           star_mul=np.around(star_mul, decimals=6)
           star_q,return_index,return_inverse,return_counts=np.unique(star_mul,
                                                               return_index=True,
                                                               return_inverse=True,
                                                               return_counts=True,
                                                               axis=0)
           if weight[i] != len(return_index) :
               sys.exit("error in expand_FBZ")

           star_t=star_mulv[return_index]
           #print(star_q.shape,star_t.shape)

           if (i == 0):
              qpoints_out=star_q
              tensor_out=star_t
           else:
              qpoints_out=np.concatenate((qpoints_out,star_q),axis=0)
              tensor_out=np.concatenate((tensor_out,star_t),axis=0)

       tensor_out=np.swapaxes(tensor_out,0,axis)
       return qpoints_out,tensor_out
    
    
