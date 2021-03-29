# calculations
import numpy as np
from scipy.interpolate import interp1d

# hdf5 files
import h5py

# crystal structure
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure

# other
import sys

from Constants import *

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
    def __init__(self, arguments):
        super(Phonon, self).__init__()
        self.args = arguments
        
    def load_properties(self):
        '''Initialise all phonon properties from input files.'''

        #### we have to discuss for the following ####
        unitcell, _ = read_crystal_structure(self.args.poscar_file, interface_mode='vasp')
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

        self.load_hdf_data(self.args.hdf_file)

        self.load_q_points()
        self.load_weights()

        self.load_frequency()
        qpoints_FBZ,frequency=expand_FBZ(0,self.weights,self.q_points,self.frequency,0,rotations,reciprocal_lattice)
        self.frequency= frequency
        self.convert_to_omega()
        
        self.load_group_vel()
        qpoints_FBZ,group_vel=expand_FBZ(0,self.weights,self.q_points,self.group_vel,1,rotations,reciprocal_lattice)
        self.group_vel=group_vel

        self.load_temperature()
    
        self.load_heat_cap()
        qpoints_FBZ,heat_cap=expand_FBZ(1,self.weights,self.q_points,self.heat_cap,0,rotations,reciprocal_lattice)
        self.heat_cap=heat_cap

        self.load_gamma()
        qpoints_FBZ,gamma=expand_FBZ(1,self.weights,self.q_points,self.gamma,0,rotations,reciprocal_lattice)
        self.gamma=gamma

        self.q_points=qpoints_FBZ
        self.weights = np.ones((len(self.q_points[:,0]),))
        self.number_of_qpoints = self.q_points.shape[0]
        self.number_of_branches = self.frequency.shape[1]
        self.number_of_modes = self.number_of_qpoints*self.number_of_branches

        print('To describe phonons:')
        print(' nq=',self.number_of_qpoints)
        print(' nb=',self.number_of_branches)
        print(' number of modes=', self.number_of_modes)

        ### please explain the following during the next meeting
        self.calculate_lifetime()
        self.initialise_temperature_function()
        
    def load_hdf_data(self, hdf_file):
        ''' Get all data from hdf file.
            Module documentation: https://docs.h5py.org/en/stable/'''

        self.data_hdf = h5py.File(hdf_file,'r')
    
    def load_frequency(self):
        '''frequency shape = q-points X p-branches '''
        self.frequency = np.array(self.data_hdf['frequency']) # THz
        self.frequency = np.where(self.frequency<0, 0, self.frequency)

    def convert_to_omega(self):
        '''omega shape = q-points X p-branches '''
        self.omega = self.frequency*2*self.pi*1e12 # rad/s

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
        self.group_vel = np.array(self.data_hdf['group_velocity'])*1e12  # Hz * angstrom

    def load_heat_cap(self):
        '''heat_cap shape = temperatures X q-points X p-branches '''
        self.heat_cap = np.array(self.data_hdf['heat_capacity'])    # eV/K
    
    def load_gamma(self):
        '''gamma = temperatures X q-pointsX p-branches'''
        self.gamma = np.array(self.data_hdf['gamma'])   # THz
        self.gamma = np.where(self.gamma > 0 , self.gamma, -1)

    def calculate_lifetime(self):
        '''lifetime = temperatures X q-pointsX p-branches'''

        self.lifetime = np.where( self.gamma>0, 1/( 2*2*np.pi*self.gamma), 0)*1e-12  # s

        self.lifetime_function = [[interp1d(self.temperature_array, self.lifetime[:, i, j], kind = 'cubic') for j in range(self.number_of_branches)] for i in range(self.number_of_qpoints)]
        
    def calculate_occupation(self, T, omega):
        flag = (T>0) & (omega>0)
        occupation = np.where(~flag, 0, 1/( np.exp( omega*self.hbar/ (T*self.kb) ) - 1) )
        return occupation

    def calculate_energy(self, T, omega):
        '''Energy of a mode given T'''
        n = self.calculate_occupation(T, omega)
        return self.hbar * omega * (n + 0.5)
    
    def calculate_crystal_energy(self, T):
        '''Calculates the average energy per mode at a given temperature for the crystal.'''

        T = np.array(T)             # ensuring right type
        T = T.reshape( (-1, 1, 1) ) # ensuring right dimensionality

        return self.calculate_energy(T, self.omega).sum( axis = (1, 2) )

    def initialise_temperature_function(self):
        '''Calculate an array of energy levels and initialises the function to recalculate T = f(E)'''

        margin = 10
        dT = 0.1
        
        T_boundary = np.array([min(self.args.temperatures)-margin, max(self.args.temperatures)+margin]) # setting temperature interval with 10K of upper and lower margin

        T_array = np.arange(T_boundary.min(), T_boundary.max()+dT, dT)    # defining discretisations

        self.energy_array = self.calculate_crystal_energy(T_array)
        self.temperature_function = interp1d( self.energy_array, T_array, kind = 'cubic' )

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
                  sys.exit("error in exapand_FBZ: not coded")

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
               sys.exit("error in exapand_FBZ")

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