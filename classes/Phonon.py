# calculations
# from msilib.schema import AdvtUISequence
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.spatial.transform import Rotation as rot

# hdf5 files
import h5py

# crystal structure
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure

# other
import sys, os, re

from classes.Constants import Constants

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
        self.mat_index = int(mat_index)
        self.get_mat_folder()

        self.load_base_properties()

        if len(self.args.mat_rotation) > 0:
            self.rotate_crystal()
        
        self.plot_FBZ()
        self.plot_relaxation_time()
        self.plot_density_of_states()

        print('Material initialisation done!')

    def get_mat_folder(self):
        
        if len(self.args.mat_folder) > 0:
            folder = os.path.relpath(self.args.mat_folder[self.mat_index])
        else:
            folder = ''

        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)
        
        self.mat_folder = folder

    def load_base_properties(self):
        '''Initialise all phonon properties from input files.'''
        
        poscar_file = os.path.join(self.mat_folder, self.args.poscar_file[self.mat_index])
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

        self.volume_unitcell = unitcell.get_volume()   # angstrom³
        
        print('Reading hdf file...')
        hdf_file = os.path.join(self.mat_folder, self.args.hdf_file[self.mat_index])

        self.load_hdf_data(hdf_file)

        self.load_q_points()
        self.load_weights()

        print('Expanding frequency to FBZ...')
        self.load_frequency()
        qpoints_FBZ,frequency=self.expand_FBZ(0,self.weights,self.q_points,self.frequency,0,rotations,reciprocal_lattice)
        self.frequency= frequency
        self.convert_to_omega()
        
        print('Expanding group velocity to FBZ...')
        self.load_group_vel()
        qpoints_FBZ,group_vel=self.expand_FBZ(0,self.weights,self.q_points,self.group_vel,1,rotations,reciprocal_lattice)
        self.group_vel = np.around(group_vel, decimals = 10)

        self.load_temperature()
        # print('Expanding heat capacity to FBZ...')  # We don't use heat capacity for anything. Keeping this for future use if needed.
        # self.load_heat_cap()
        # qpoints_FBZ,heat_cap=self.expand_FBZ(1,self.weights,self.q_points,self.heat_cap,0,rotations,reciprocal_lattice)
        # self.heat_cap=heat_cap

        print('Expanding gamma to FBZ...')
        self.load_gamma()
        qpoints_FBZ,gamma=self.expand_FBZ(1,self.weights,self.q_points,self.gamma,0,rotations,reciprocal_lattice)
        self.gamma=gamma

        self.q_points = qpoints_FBZ
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

        self.get_wavevectors()
        self.get_norms()

        print('Searching for degeneracies...')
        self.find_degeneracies()

        print('Material info: {:d} q-points; {:d} branches -> {:d} modes in total.'.format(self.number_of_qpoints, self.number_of_branches, self.number_of_modes))

        print('Interpolating lifetime...')
        self.calculate_lifetime()

        print('Generating T = f(E)...')
        self.zero_point = self.calculate_zeropoint()
        
        self.initialise_temperature_function()

        self.initialise_density_of_states()

        

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
        q = np.copy(self.q_points)
        k = self.q_to_k(q)

        self.wavevectors = self.find_min_k(k)

    def plot_FBZ(self):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, subplot_kw={'projection': '3d'}, figsize = (6, 5), dpi = 300)
        ax.scatter(self.wavevectors[:, 0], self.wavevectors[:, 1], self.wavevectors[:, 2], s = 1, c = np.sum(self.wavevectors**2, axis = 1))

        ax.set_xlabel(r'$\vec{k}_x$')
        ax.set_ylabel(r'$\vec{k}_y$')
        ax.set_zlabel(r'$\vec{k}_z$')

        fig.suptitle(r'Wavevectors in FBZ, coloured by $|\vec{\kappa}|$.')
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.mat_folder, 'FBZ.png'))
        plt.close(fig)

    def find_min_k(self, k, return_disp = False):
        '''Gives the equivalent wavevector inside the FBZ.
           k = wavevector to be analysed (N, 3)
           return_disp = if true, returns the displacement in reciprocal space to translate k to k_fbz.
           
           Returns:
           k_fbz  = the equivalent wavevector in FBZ.
           k_disp = the diplacement from k to k_fbz.'''

        a = np.array([-1, 0, 1])
        b = np.meshgrid(a, a, a)
        n = (np.vstack(tuple(map(np.ravel, b))).T)
        n = np.expand_dims(n, 1)

        i0 = np.nonzero(np.all(n == 0, axis = -1))[0][0] # index where n = [0, 0, 0]

        disp = np.zeros(k.shape)
        
        q = self.k_to_q(k)
        active = np.ones(q.shape[0], dtype = bool)

        while np.any(active):                               # For reference: Qa = number of qpoints still active
            q_new = q[active, :] + n                        # every q in the neighbourhood (27, Qa, 3)
            k_new = self.q_to_k(q_new)                      # convert to k (27, Qa, 3)
            k_norm = np.linalg.norm(k_new, axis = -1).T     # calculate the norm (Qa, 27)
            k_min = k_norm.min(axis = 1, keepdims = True)   # minimum norm for each mode (Qa, 1)
            i_min = np.argmax(k_norm == k_min, axis = 1)    # get the index where the minimum is located (Qa,)

            if return_disp:
                disp[active, :] += n[i_min, 0, :]              # displacement from previous k in reduced coordinates

            q[active, :] = np.copy(q_new[i_min, np.arange(active.sum()), :])

            active[active] = ~(i_min == i0)                  # If the minimum is in i0, the shortest wavevector was found for that mode

        if return_disp:
            return self.q_to_k(q), self.q_to_k(disp)
        else:
            return self.q_to_k(q)

    def is_k_min(self, k):
        a = np.array([-1, 0, 1])
        b = np.meshgrid(a, a, a)
        n = (np.vstack(map(np.ravel, b)).T)

        d = self.q_to_k(n)          # convert to k coordinates
        d = np.expand_dims(d, 1)    # (27, 1, 3)

        i0 = np.nonzero(np.all(n == 0, axis = 1))[0][0] # index where n = [0, 0, 0]

        k_new = k + d                                   # dislocate k (27, N, 3)
        k_norm = np.linalg.norm(k_new, axis = -1).T     # calculate the norm (N, 27)
        k_min = k_norm.min(axis = 1, keepdims = True)   # minimum norm for each k (N, 1)
        i_min = np.argmax(k_norm == k_min, axis = 1)    # get the index where the minimum is located (Qa,)

        return i_min == i0
    
    def get_norms(self):
        self.norm_group_vel = np.linalg.norm(self.group_vel, axis = 2)
        self.norm_wavevectors = np.linalg.norm(self.wavevectors, axis = 1)

    def k_to_q(self, k):
        # convert wave vectors to q-points in the first brillouin zone
        a = np.linalg.inv(self.reciprocal_lattice)  # transformation vector
        
        q = np.dot(k, a.T)

        return q
    
    def q_to_k(self, q):
        # convert q-points to wave vectors in the first brillouin zone
        k = np.dot(q, self.reciprocal_lattice.T)

        return k        

    def rotate_crystal(self):
        '''Rotates the orientation of the crystal in relation to the geometry axis.'''
        print('Rotating crystal...')        
        
        rot_groups = []
        g = []
        for i, s in enumerate(self.args.mat_rotation):
            if type(s) != str:
                s = str(s)
            if re.fullmatch('[0-9]+', s):
                g.append(i)
            elif re.fullmatch('[A-Z]+|[a-z]+', s):
                g.append(i)
                rot_groups.append(g)
                g = []
            else:
                Exception('Wrong mat_rotation parameter. Quitting simulaton.')
        
        if len(rot_groups) > 0:
            group = rot_groups[self.mat_index]
            
            rot_params = [self.args.mat_rotation[i] for i in group]
            self.rotation_angles = [float(i) for i in rot_params[:-1]]
            self.rotation_order  = rot_params[-1]
            
            R = rot.from_euler(self.rotation_order, self.rotation_angles, degrees = True)

            self.wavevectors = R.apply(self.wavevectors) # rotate k

            for i in range(self.number_of_branches): # for each branch
                self.group_vel[:, i, :] = R.apply(self.group_vel[:, i, :]) # rotate v_g

    def load_gamma(self):
        '''gamma = temperatures X q-pointsX p-branches'''
        self.gamma = np.array(self.data_hdf['gamma'])   # THz
        if self.mat_index in self.args.isotope_scat:
            try:
                self.gamma += np.array(self.data_hdf['gamma_isotope'])
            except:
                raise Exception('hdf file does not contain the field "gamma_isotope".')
        self.gamma = np.where(self.gamma > 0 , self.gamma, -1)

    def calculate_lifetime(self):
        '''lifetime = temperatures X q-pointsX p-branches'''

        self.lifetime = np.where( self.gamma>0, 1/( 2*2*np.pi*self.gamma), 0) # ps

        q_array = np.arange(self.number_of_qpoints )
        j_array = np.arange(self.number_of_branches)
        T = self.temperature_array
        tau = self.lifetime

        self.lifetime_function = RegularGridInterpolator((T, q_array, j_array), tau)

    def calculate_occupation(self, T, omega):
        '''Calculate the Bose-Einstein occupation number of a given frequency at temperature T.'''

        flag = (T>0) & (omega>0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            occupation = np.where(~flag, 0, 1/( np.exp( omega*self.hbar/ (T*self.kb) ) - 1))
        
        return occupation

    def calculate_energy(self, T, omega):
        '''Energy of a mode given T and omega due to its occupation (ignoring zero-point energy).'''
        n = self.calculate_occupation(T, omega)
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
        self.temperature_function = interp1d(self.energy_array, T_array, kind = 'linear', fill_value = (T_min, T_max), bounds_error = False)

        # saving crystal energy function as well
        self.crystal_energy_function = interp1d(T_array, self.energy_array, kind = 'linear', fill_value = (self.energy_array.min(), self.energy_array.max()), bounds_error = False)

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

    def find_degeneracies(self):
        self.degenerate_modes = []       # list of groups of degenerate modes
        for q in range(self.number_of_qpoints): # for each wavevector (q-point)
            u_omega, i_omega, c_omega = np.unique(self.omega[q, :], return_inverse = True, return_counts= True) # unique frequencies, inverse and their counts

            if np.any(c_omega > 1):                     # if there is any repeated omega
                ui_omega = np.nonzero(c_omega > 1)[0]   # unique indexes of the degenerate branches
                for ui in ui_omega:                     # for each degenerate branch
                    b = np.nonzero(i_omega == ui)[0]    # degenerate branch index
                    v = self.group_vel[q, b, :]         # velocities of degenerate branches

                    u_v, i_v, c_v = np.unique(v, axis = 0, return_inverse = True, return_counts = True) # unique velocities, inverse and counts

                    if np.any(c_v > 1):
                        ui_v = np.nonzero(c_v > 1)[0]
                        for uui in ui_v:
                            d_q = (np.ones(c_v[uui])*q).astype(int) # degenerate qpoints
                            d_b = np.nonzero(i_v == uui)[0]         # degenerate branches

                            d = np.vstack((d_q, d_b)).T # array with q

                            self.degenerate_modes.append(d)
    
    def plot_relaxation_time(self):
        '''Plots the scattering probability of the maximum temperature (in which scattering mostly occurs)
        and gives information about simulation instability due to de ratio dt/tau.'''

        T_all = self.temperature_array[self.temperature_array % 100 == 0] # multiples of 100 K available in the data

        cmap   = matplotlib.cm.get_cmap('jet')
        colors = cmap((T_all - T_all.min())/T_all.ptp())

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5), dpi = 300)
        
        x_data = self.omega[self.unique_modes[:, 0], self.unique_modes[:, 1]]

        for i, T in enumerate(T_all):
            # calculating y data
            Tqj = np.hstack( ( ( np.ones(self.unique_modes.shape[0])*T).reshape(-1, 1), self.unique_modes ) )
        
            lifetime = self.lifetime_function(Tqj)

            ax.scatter(x_data, lifetime, s = 1, color = colors[i, :])

        handles = [Ellipse(xy = np.random.rand(2)*np.array([np.array(ax.get_xlim()).ptp(), np.array(ax.get_ylim()).ptp()])+np.array([ax.get_xlim()[0], ax.get_ylim()[0]]),
                           height = 1, width = 1,
                           color = c, label = '{:.1f} K'.format(T_all[i])) for i, c in enumerate(colors)]

        ax.legend(handles = handles, fontsize = 'medium')
       
        ax.set_xlabel(r'Angular frequency $\omega$ [rad THz]',
                        fontsize = 'large')
        ax.set_ylabel(r'Phonon relaxation time $\tau$ [ps]',
                        fontsize = 'large')

        ax.set_yscale('log')
        
        plt.tight_layout()

        plt.savefig(os.path.join(self.mat_folder,'relaxation_times.png'))

        plt.close(fig)

    def plot_density_of_states(self):

        n_bins    = 200
        
        d_omega   = self.omega.max()/n_bins

        intervals = np.linspace(0, self.omega.max(), n_bins+1)
        
        centers = (intervals[1:] + intervals[:-1])/2

        dos = np.zeros((n_bins, self.number_of_branches))

        cmap   = matplotlib.cm.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, self.number_of_branches))

        for b in range(self.number_of_branches):
            omega = self.omega[:, b]

            below = (omega.reshape(-1, 1) < intervals)[:, 1:]
            above = (omega.reshape(-1, 1) >= intervals)[:, :-1]
            
            dos[:, b] = (below & above).sum(axis = 0)/d_omega

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5), dpi = 300)
        

        ax.stackplot(centers, dos.T, labels = ['Branch {:d}'.format(i) for i in range(self.number_of_branches)], step = 'pre', colors = colors)

        ax.set_ylim(0)

        ax.legend(fontsize = 'medium')

        ax.set_xlabel(r'Angular Frequency $\omega$ [THz]', fontsize = 'medium')
        ax.set_ylabel(r'Density of states $g(\omega)$ [THz$^{-1}$]', fontsize = 'medium')

        plt.title(r'Density of states. {:d} bins, $d\omega = $ {:.3f} THz'.format(n_bins, d_omega), fontsize = 'large')

        plt.tight_layout()

        plt.savefig(os.path.join(self.mat_folder,'density_of_states.png'))
        plt.close(fig)

    def expand_FBZ(self, axis,weight,qpoints,tensor,rank,rotations,reciprocal_lattice):
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

           if (i == 0):
              qpoints_out=star_q
              tensor_out=star_t
           else:
              qpoints_out=np.concatenate((qpoints_out,star_q),axis=0)
              tensor_out=np.concatenate((tensor_out,star_t),axis=0)

       tensor_out=np.swapaxes(tensor_out,0,axis)
       return qpoints_out,tensor_out
    
    
