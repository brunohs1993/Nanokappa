
# calculations
import numpy as np
from datetime import datetime

import scipy.constants as ct
from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import interp1d

# plotting
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# hdf5 files
import h5py

# crystal structure
# import phonopy
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure
import pyiron.vasp.structure

# geometry
import trimesh as tm
from trimesh.ray.ray_pyembree import RayMeshIntersector

# other
import sys
import os
import copy
from functools import partial

np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=np.nan)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

class Constants:
    def __init__(self):
        self.hbar = ct.physical_constants['reduced Planck constant in eV s'  ][0]   # hbar in eV s
        # self.hbar = ct.physical_constants['Planck constant over 2 pi in eV s'][0]   # hbar in eV s
        self.kb   = ct.physical_constants['Boltzmann constant in eV/K'       ][0]   # kb in eV/K
        self.ev_in_J = ct.physical_constants['electron volt'][0]                     # J/eV

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - Define boundary conditions

class Geometry:
    def __init__(self, arguments):
        
        self.args = arguments

        self.standard_shapes = ['cuboid', 'cillinder', 'sphere']
        self.scale           = self.args.scale
        self.dimensions      = self.args.dimensions             # angstrom
        self.rotation        = np.array(self.args.rotation)
        self.rot_order       = self.args.rot_order
        self.shape           = self.args.geometry

        # Processing mesh

        self.load_geo_file(self.shape)  # loading
        self.transform_mesh()           # transforming
        self.get_mesh_properties()      # defining useful properties


    def load_geo_file(self, shape):
        '''Load the file informed in --geometry. If an standard geometry defined in __init__, adjust
        the path to load the native file. Else, need to inform the whole path.'''
        
        if shape == 'cuboid':
            self.mesh = tm.creation.box( self.dimensions )
        elif shape == 'cylinder':
            self.mesh = tm.creation.cone( self.dimensions[0], height = self.dimensions[1] )
        elif shape == 'cone':
            self.mesh = tm.creation.cone( self.dimensions[0], self.dimensions[1] )
        elif shape == 'capsule':
            self.mesh = tm.creation.capsule( self.dimensions[0], self.dimensions[1] )
        else:
            self.mesh = tm.load(shape)

    def transform_mesh(self):
        '''Builds transformation matrix and aplies it to the mesh.'''

        self.mesh.rezero()  # brings mesh to origin such that all vertices are on the positive octant

        scale_matrix = np.identity(4)*np.append(self.scale, 1)

        self.mesh.apply_transform(scale_matrix) # scale mesh

        rotation_matrix       = np.zeros( (4, 4) ) # initialising transformation matrix
        rotation_matrix[3, 3] = 1

        rotation_matrix[0:3, 0:3] = rot.from_euler(self.rot_order, self.rotation).as_matrix() # building rotation terms

        self.mesh.apply_transform(rotation_matrix) # rotate mesh

    def get_mesh_properties(self):
        '''Get useful properties of the mesh.'''

        self.bounds = self.mesh.bounds

    # def set_boundary_cond(self):
                
    #     self.bound_cond = np.array(self.args.bound_cond)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

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
        self.rank_energies()
        self.calculate_lifetime()
        self.initialise_temperature_function()
        
        self.load_poscar_data(self.args.poscar_file)

        ### I am surpised than you need lattice vectors, and nor reciprocal lattice vectors. Check that it is the same as "lattice" computed before.
        self.load_lattice_vectors()

    def load_hdf_data(self, hdf_file,):
        ''' Get all data from hdf file.
            Module documentation: https://docs.h5py.org/en/stable/'''

        self.data_hdf = h5py.File(hdf_file,'r')
    
    def load_poscar_data(self, poscar_file):
        ''' Get structure from poscar file.
            Module documentation: https://pyiron.readthedocs.io/en/latest/index.html'''

        self.data_poscar = pyiron.vasp.structure.read_atoms(poscar_file)

    def load_frequency(self):
        '''frequency shape = q-points X p-branches '''
        self.frequency = np.array(self.data_hdf['frequency']) # THz
        self.frequency = np.where(self.frequency<0, 0, self.frequency)

    def convert_to_omega(self):
        '''omega shape = q-points X p-branches '''
        self.omega = self.frequency*2*ct.pi*1e12 # rad/s
    
    def rank_energies(self):
        matrix = self.calculate_energy(10, self.omega)
        matrix = np.where(matrix == 0, np.inf, matrix)
        self.rank = matrix.argsort(axis = None).argsort().reshape(matrix.shape)

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
        
    def load_lattice_vectors(self):
        '''Unit cell coordinates from poscar file.'''
        self.lattice_vectors = self.data_poscar.cell    # lattice vectors in cartesian coordinates, angstrom
    
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


    
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#   - Generalise functions for other geometries --> calculate slice volume based on mesh
#   - Add option to apply heat flux as boundary condition
#   
#   QUESTIONS
#   
#   - Do the q-poins even matter? Because they are not used in any calculation, and they
#     can be recovered by using self.modes i.e. q-point and branch references.

class Population(Constants):
    '''Class comprising the particles to be simulated.'''

    def __init__(self, arguments, geometry, phonon):
        super(Population, self).__init__()

        self.args = arguments
        
        if self.args.particles_ex == None:
            self.N_p   = int(float(self.args.particles[0]))
            self.exact = False
        else:
            self.N_p   = int(float(self.args.particles_ex[0]))
            self.exact = True
         
        self.dt            = float(self.args.timestep[0])
        self.n_of_slices   = self.args.slices[0]
        self.slice_axis    = self.args.slices[1]
        self.slice_length  = geometry.bounds[:, self.slice_axis].ptp()/self.n_of_slices

        self.bound_cond     = self.args.bound_cond    
        self.T_boundary     = np.array(self.args.temperatures)
        self.T_distribution = self.args.temp_dist[0]

        self.colormap = self.args.colormap
        self.fig_plot = self.args.fig_plot
        self.rt_plot  = self.args.rt_plot

        self.t = 0

        self.initialise_all_particles(phonon, geometry) # initialising particles
        self.initialise_reservoirs(geometry, phonon)            # initialise reservoirs
        self.n_timesteps = self.timesteps_to_boundary(self.positions, self.group_vel, geometry) # calculating timesteps to boundary

        self.conv_filename = self.args.conv_file
        self.open_convergence(self.conv_filename)

        if len(self.rt_plot) > 0:
            
            self.rt_graph, self.rt_fig = self.init_plot_real_time(geometry)
        

    def initialise_modes(self, number_of_particles, phonon):
        '''Uniformily organise modes for a given number of particlesprioritising low energy ones.'''

        part_p_mode = np.floor(number_of_particles/phonon.number_of_modes)    # particles per mode

        ppm_array = (np.ones(phonon.number_of_modes)*part_p_mode).astype(int)

        if self.exact: # if the exact number of particles is enforced

            lack = int(number_of_particles - part_p_mode*phonon.number_of_modes)    # lack of particles due to rounding

            lack_modes = np.where( np.delete(phonon.rank, [0, 1, 2]).reshape(-1) < lack )[0]    # modes to correct based on rank

            ppm_array[lack_modes] += 1   # corrected number of particles per mode
        
        else: # if not
            ppm_array += 1  # add to all modes

            self.N_p = ppm_array.sum()
            number_of_particles = self.N_p

        modes = np.empty( (number_of_particles, 2) ) # initialising mode array

        counter = 0

        for i in range(phonon.number_of_qpoints):
            for j in range(phonon.number_of_branches):

                index = int(i*phonon.number_of_branches+j)

                modes[ counter:counter+ppm_array[index], 0] = i                         
                modes[ counter:counter+ppm_array[index], 1] = j

                counter += ppm_array[index]
    
        return modes.astype(int), ppm_array

    def generate_positions(self, number_of_particles, geometry):
        '''Initialise positions of a given number of particles'''
        
        # Get geometry bounding box
        bounds = geometry.bounds
        bounds_range = bounds[1,:]- bounds[0,:]

        positions = np.random.rand(number_of_particles, 3)*bounds_range + bounds[0,:] # Generate random points inside bounding box, positions in angstrom
        positions, points_out = self.remove_out_points(positions, geometry)       # Remove points outside mesh

        while positions.shape[0] != number_of_particles:
            # Keep generating points until all of them are inside the mesh
            new_positions = np.random.rand(points_out, 3)*bounds_range + bounds[0,:] # generate new
            new_positions, points_out = self.remove_out_points(new_positions, geometry)         # check which are inside
            positions = np.append(positions, new_positions, 0)         # add the good ones
        
        return positions

    def remove_out_points(self, points, geometry):
        '''Remove points outside the geometry mesh.'''
        in_points = geometry.mesh.contains(points)     # boolean array
        out_points = ~in_points
        return np.delete(points, out_points, axis=0), out_points.sum()   # new points, number of particles

    def initialise_all_particles(self, phonon, geometry):
        '''Uses Population atributes to generate all particles.'''
        
        self.modes, ppm_array = self.initialise_modes(self.N_p, phonon) # getting modes
        
        # initialising positions one mode at a time (slower but uses less memory)

        self.positions = np.empty( (self.N_p, 3) )
        
        counter = 0
        for i in range(phonon.number_of_qpoints):
            for j in range(phonon.number_of_branches):

                index = int(i*phonon.number_of_branches+j)
                
                self.positions[counter:counter+ppm_array[index], :] = self.generate_positions(ppm_array[index], geometry)

                counter += ppm_array[index]
        
        self.temperatures = self.assign_temperatures(self.positions, geometry)
        
        # assigning properties from Phonon
        self.omega, self.q_points, self.group_vel = self.assign_properties(self.modes, self.temperatures, phonon)
        
        self.occupation   = phonon.calculate_occupation(self.temperatures, self.omega)
        self.energies     = phonon.calculate_energy(self.temperatures, self.omega)

    def initialise_reservoirs(self, geometry, phonon):
        # coded at first for simple boxes at the extremities of reservoirs.

        self.res1_active_modes = np.stack(np.meshgrid( np.arange(phonon.number_of_qpoints), np.arange(phonon.number_of_branches) ), axis = -1 ).reshape(-1, 2)
        self.res2_active_modes = np.stack(np.meshgrid( np.arange(phonon.number_of_qpoints), np.arange(phonon.number_of_branches) ), axis = -1 ).reshape(-1, 2)


        reservoir_length = self.slice_length

        self.N_p_res = int(self.N_p/self.n_of_slices) # equivalent number of particles in that distance

        self.bound_res1 = copy.deepcopy(geometry.bounds)
        self.bound_res1[:, self.slice_axis] = np.array([-reservoir_length, 0])
        
        self.bound_res2 = copy.deepcopy(geometry.bounds)
        self.bound_res2[:, self.slice_axis] = np.array([self.bound_res2[1, self.slice_axis], self.bound_res2[1, self.slice_axis]+reservoir_length])

        # Generate first particles
        self.fill_reservoirs(phonon)

    def generate_reservoir_particles(self, box, modes):

        '''Generates radomly particles and modes for reservoirs given their bounding box and phonon properties'''

        # positions

        positions_res = np.random.rand(self.N_p_res, 3)*np.ptp(box, axis = 0)+box[0,:]

        # modes

        modes_res = np.floor(np.random.rand(self.N_p_res)*modes.shape[0]).astype(int)
        modes_res = modes[modes_res, :]

        return positions_res, modes_res
    
    def fill_reservoirs(self, phonon):

        # - positions and modes
        self.positions_res1, self.modes_res1 = self.generate_reservoir_particles(self.bound_res1, self.res1_active_modes) #phonon)
        self.positions_res2, self.modes_res2 = self.generate_reservoir_particles(self.bound_res2, self.res2_active_modes) #phonon)

        # temperatures
        self.temperatures_res1 = np.ones(self.N_p_res)*self.T_boundary[0]
        self.temperatures_res2 = np.ones(self.N_p_res)*self.T_boundary[1]

        # - assign angular frequencies, q points, group velocities and lifetimes
        self.omega_res1, self.q_points_res1, self.group_vel_res1 = self.assign_properties(self.modes_res1, self.temperatures_res1, phonon)
        self.omega_res2, self.q_points_res2, self.group_vel_res2 = self.assign_properties(self.modes_res2, self.temperatures_res2, phonon)

        self.energies_res1 = phonon.calculate_energy(self.temperatures_res1, self.omega_res1)
        self.energies_res2 = phonon.calculate_energy(self.temperatures_res2, self.omega_res2)

        self.occupation_res1 = phonon.calculate_occupation(self.temperatures_res1, self.omega_res1)
        self.occupation_res2 = phonon.calculate_occupation(self.temperatures_res2, self.omega_res2)
        
    def add_reservoir_particles(self, geometry, phonon):
        '''Add the particles that came from the reservoir into the geometry. Similar to the boundary scattering method.'''
        
        positions    = np.vstack((self.positions_res1, self.positions_res2))
        modes        = np.vstack((self.modes_res1    , self.modes_res2    ))
        group_vel    = np.vstack((self.group_vel_res1, self.group_vel_res2))
        q_points     = np.vstack((self.q_points_res1 , self.q_points_res2 ))

        temperatures       = np.concatenate((self.temperatures_res1      , self.temperatures_res2      ))
        omega              = np.concatenate((self.omega_res1             , self.omega_res2             ))
        occupation         = np.concatenate((self.occupation_res1        , self.occupation_res2        ))
        energies           = np.concatenate((self.energies_res1          , self.energies_res2          ))
        

        if (self.bound_cond == 'periodic') and (geometry.shape == 'cuboid'):    # applicable just for cuboids

            check = positions >= geometry.bounds.reshape(2, 1, 3)    # True if less or equal than each bound value

            check = check.sum(axis = 0) # this will result in 0 for numbers out of the lower limit, 1 for points inside the limits, 2 for outside de upper limit

            # deleting particles that stil are in the reservoirs
            indexes_del = (check[:, self.slice_axis] != 1)

            positions    = positions[~indexes_del, :]
            modes        = modes[~indexes_del, :]
            check        = check[~indexes_del, :]

            # applying periodicity
            lower_points = np.where(check == 0, geometry.bounds[1,:] - positions % np.ptp(geometry.bounds, axis = 0), 0)
            in_points    = np.where(check == 1, positions                                                           , 0)
            upper_points = np.where(check == 2, geometry.bounds[0,:] + positions % np.ptp(geometry.bounds, axis = 0), 0)

            new_positions = lower_points + in_points + upper_points

            temperatures       = temperatures[~indexes_del]
            omega              = omega[~indexes_del]
            group_vel          = group_vel[~indexes_del, :]
            occupation         = occupation[~indexes_del]
            energies           = energies[~indexes_del]
            q_points           = q_points[~indexes_del, :]

            n_timesteps = self.timesteps_to_boundary(new_positions, group_vel, geometry)  # calculates ts until boundary scattering
            
        # add new particles to the population

        self.positions          = np.vstack((self.positions  , new_positions))
        self.modes              = np.vstack((self.modes      , modes      ))
        self.q_points           = np.vstack((self.q_points   , q_points   ))
        self.group_vel          = np.vstack((self.group_vel  , group_vel  ))
        self.n_timesteps        = np.concatenate((self.n_timesteps, n_timesteps))
        self.temperatures       = np.concatenate((self.temperatures      , temperatures      ))
        self.omega              = np.concatenate((self.omega             , omega             ))
        self.energies           = np.concatenate((self.energies          , energies          ))
        self.occupation         = np.concatenate((self.occupation        , occupation        ))

    def assign_properties(self, modes, temperatures, phonon):
        '''Get properties from the indexes.'''

        omega              = phonon.omega[ modes[:,0], modes[:,1] ]        # rad/s
        q_points           = phonon.q_points[ modes[:,0], : ]                 # reduced reciprocal coordinates
        group_vel          = phonon.group_vel[ modes[:,0], modes[:,1], : ] # Hz * angstrom

        return omega, q_points, group_vel

    def assign_temperatures(self, positions, geometry):
        '''Atribute initial temperatures imposing fixed temperatures on first and last slice. Randomly within delta T unless specified otherwise.'''

        number_of_particles = positions.shape[0]
        key = self.T_distribution

        temperatures = np.empty(number_of_particles)    # initialise temperature array

        if key == 'linear':
            # calculates T for each slice
            step = (self.T_boundary[1]-self.T_boundary[0])/(self.n_of_slices-1)
            T_array = np.arange(self.T_boundary[0], self.T_boundary[1] + step, step)

            for i in range(self.n_of_slices):   # assign temperatures for particles in each slice
                indexes = (positions[:, self.slice_axis] >= i*self.slice_length) & (positions[:, self.slice_axis] < (i+1)*self.slice_length)
                temperatures[indexes] = T_array[i]
                    
        else:
            if key == 'random':
                temperatures = np.random.rand(number_of_particles)*(self.T_boundary.ptp() ) + self.T_boundary.min()
                
            elif key == 'constant_hot':
                temperatures = np.ones(number_of_particles)*self.T_boundary.max()
            elif key == 'constant_cold':
                temperatures = np.ones(number_of_particles)*self.T_boundary.min()
            elif key == 'constant_mean':
                temperatures = np.ones(number_of_particles)*self.T_boundary.mean()
            
            # imposing boundary conditions

            indexes = positions[:, self.slice_axis] < self.slice_length
            temperatures[indexes] = self.T_boundary[0]

            indexes = positions[:, self.slice_axis] >= geometry.bounds[1, self.slice_axis]-self.slice_length
            temperatures[indexes] = self.T_boundary[1]

        return temperatures
    
    def refresh_temperatures(self, geometry, phonon):
        '''Refresh energies and temperatures while enforcing boundary conditions as given by geometry.'''
        slice_energy = np.empty(self.n_of_slices)
        self.slice_temperature = np.empty(self.n_of_slices)
        self.slice_heat_flux = np.empty(self.n_of_slices)

        self.slice_id = np.empty(self.positions.shape[0])

        for i in range(self.n_of_slices):   # assign temperatures for particles in each slice
            indexes = (self.positions[:, self.slice_axis] >= i*self.slice_length) & (self.positions[:, self.slice_axis] < (i+1)*self.slice_length)

            self.slice_id[indexes] = i

            slice_energy[i]      = self.energies[indexes].mean()*phonon.number_of_modes
            self.slice_heat_flux[i] = self.calculate_heat_flux(indexes, geometry)
            self.slice_temperature[i] = phonon.temperature_function( slice_energy[i] )
            # self.slice_temperature[i] = self.calculate_slice_temperature(i, phonon) # THIS IS FOR DEBUGGING

            self.temperatures[indexes] = copy.deepcopy(self.slice_temperature[i])

    #=#=#=#=#=#=#=#=#=#=#= DEBUGGING / SPECULATION #=#=#=#=#=#=#=#=#=#=
    # def calculate_slice_temperature(self, slice_index, phonon):
        
    #     indexes = (self.slice_id == slice_index)

    #     modes = self.modes[indexes]

    #     unique_modes, unique_times = np.unique(modes, axis = 0, return_counts=True)

    #     weights = np.zeros( (phonon.number_of_qpoints, phonon.number_of_branches) )

    #     weights[unique_modes[:, 0], unique_modes[:,1]] = unique_times

    #     if slice_index == 0:
    #         T_min = self.T_boundary[0]
    #     else:
    #         T_min = self.slice_temperature[slice_index-1]
        
    #     if slice_index+1 == self.n_of_slices:
    #         T_max = self.T_boundary[1]
    #     else:
    #         T_max = self.slice_temperature[slice_index+1]
        
    #     dT = (T_max-T_min)/100

    #     T_array = np.arange(T_min-dT, T_max+2*dT, dT)

    #     e_array = (phonon.calculate_energy(T_array.reshape(-1, 1, 1), phonon.omega)*weights).sum(axis = (1, 2) )

    #     T_function = interp1d(e_array, T_array, kind = 'cubic')

    #     energy = self.energies[indexes].sum()

    #     return T_function(energy)


    #=#=#=#=#=#=#=#=#=#=#= DEBUGGING / SPECULATION #=#=#=#=#=#=#=#=#=#=

    def calculate_heat_flux(self, indexes, geometry):

        group_vel = self.group_vel[indexes, self.slice_axis]   # angstrom * Hz

        energies   = self.energies[indexes]     # eV
        omega      = self.omega[indexes]        # rad/s
        occupation = self.occupation[indexes]   # -

        energies  = self.hbar*omega*occupation
        
        if geometry.shape == 'cuboid':
            volume = np.delete(geometry.bounds, self.slice_axis, axis = 1)
            volume = np.ptp(volume, axis = 0).prod()
            volume *= self.slice_length                                      # angstrom^3

        # For other geometries, check later:
        # trimesh.mass_properties(volume)
        
        heat_flux = ((group_vel*energies).mean()/volume)*self.ev_in_J*1e20  # W/m²

        return heat_flux

    def drift(self, geometry):
        '''Drift operation.'''

        self.positions += self.group_vel*self.dt

        self.positions_res1 += self.group_vel_res1*self.dt
        self.positions_res2 += self.group_vel_res2*self.dt

    def find_boundary(self, positions, velocities, geometry):
        '''Finds which mesh triangle will be hit by the particle, given an initial position and velocity
        direction. It works with individual particles as well with a group of particles.
        Returns: array of faces indexes for each particle'''

        boundary_pos, index_ray, _ = geometry.mesh.ray.intersects_location(ray_origins      = positions ,
                                                                            ray_directions   = velocities,
                                                                            multiple_hits    = False     )

        stationary = ~np.in1d(np.arange(positions.shape[0]), index_ray)   # find which particles have zero velocity, so no ray

        all_boundary_pos                = np.zeros( (positions.shape[0], 3) )
        all_boundary_pos[stationary, :] = np.inf
        all_boundary_pos[~stationary, :]= boundary_pos

        return all_boundary_pos

    def timesteps_to_boundary(self, positions, velocities, geometry):
        '''Calculate how many timesteps to boundary scattering.'''
        
        boundary_pos = self.find_boundary(positions, velocities, geometry)

        # calculate distances for given particles
        boundary_dist = np.linalg.norm( positions - boundary_pos, axis = 1 )

        ts_to_boundary = boundary_dist/( np.linalg.norm(velocities, axis = 1) * self.dt )

        ts_to_boundary = np.ceil(ts_to_boundary)    # such that particle hits the boundary when no_timesteps == 0 (crosses boundary)

        return ts_to_boundary.astype(int)
    
    def delete_particles(self, indexes):
        '''Delete all information about particles according to the given indexes'''

        self.positions          = np.delete(self.positions         , indexes, axis = 0)
        self.q_points           = np.delete(self.q_points          , indexes, axis = 0)
        self.group_vel          = np.delete(self.group_vel         , indexes, axis = 0)
        self.omega              = np.delete(self.omega             , indexes, axis = 0)
        self.energies           = np.delete(self.energies          , indexes, axis = 0)
        self.occupation         = np.delete(self.occupation        , indexes, axis = 0)
        self.temperatures       = np.delete(self.temperatures      , indexes, axis = 0)
        self.n_timesteps        = np.delete(self.n_timesteps       , indexes, axis = 0)
        self.modes              = np.delete(self.modes             , indexes, axis = 0)
    
    def boundary_scattering(self, geometry):
        '''Applies boundary scattering or other conditions to the particles where it happened, given their indexes.'''

        indexes = self.n_timesteps <= 0   # find all boundary scattering particles

        positions = self.positions[indexes, :]
        modes     = self.modes[indexes, :]
        reference = np.where(indexes)[0]
        group_vel = self.group_vel[indexes, :]

        if (self.bound_cond == 'periodic') and (geometry.shape == 'cuboid'):    # applicable just for cuboids

            check = positions >= geometry.bounds.reshape(2, 1, 3)    # True if plus or equal than each bound value

            check = check.sum(axis = 0).round().astype(int) # this will result in 0 for numbers out of the lower limit, 1 for points inside the limits, 2 for outside de upper limit

            # deleting particles that are in the reservoirs
            indexes_del = (check[:, self.slice_axis] != 1)

            positions = positions[~indexes_del, :]
            modes     = modes[~indexes_del, :]
            check     = check[~indexes_del, :]
            group_vel = group_vel[~indexes_del, :]

            # applying periodicity

            lower_points = np.where(check == 0, geometry.bounds[1,:] - positions % np.ptp(geometry.bounds, axis = 0), 0)
            in_points    = np.where(check == 1, positions                                                           , 0)
            upper_points = np.where(check == 2, geometry.bounds[0,:] + positions % np.ptp(geometry.bounds, axis = 0), 0)

            new_positions = lower_points + in_points + upper_points

            new_ts_to_boundary = self.timesteps_to_boundary(new_positions, group_vel, geometry)  # calculates new ts until boundary scattering
        
            indexes = np.delete(indexes, reference[indexes_del])
            self.delete_particles(reference[indexes_del])
            
            self.positions[indexes, :] = new_positions
            self.n_timesteps[indexes] = new_ts_to_boundary

    def lifetime_scattering(self, geometry, phonon):
        '''Performs lifetime scattering.'''

        # N_as = N_ad + dt/tau (N_BE(T*) - N_ad)

        # identifying unique combinations of modes and slices

        ids = np.hstack((self.modes, self.slice_id.reshape(-1, 1))).astype(int)

        unique_ids = np.unique(ids, axis = 0)
        
        func = partial(self.update_occupation, ids = ids, phonon = phonon)

        map(func, unique_ids)

        self.energies = self.omega*self.hbar*(0.5 + self.occupation)

    def update_occupation(self, id_array, ids, phonon):
            
            q      = id_array[0]
            branch = id_array[1]
            slc    = id_array[2]

            indexes = np.equal(ids, id_array)
            indexes = (indexes.sum(axis = 1).round().astype(int) == 3)

            occupation_ad = copy.deepcopy(self.occupation[indexes])

            omega = phonon.omega[q, branch]
            T     = self.slice_temperature[slc]
            occupation_BE = phonon.calculate_occupation(T, omega)

            tau = phonon.lifetime_function[q][branch](T)

            self.occupation[indexes] = occupation_ad + (self.dt/tau) *(occupation_BE - occupation_ad)

    def run_timestep(self, geometry, phonon):

        self.drift(geometry)                            # drift particles

        self.add_reservoir_particles(geometry, phonon)  # add reservoir particles that come in the domain

        self.boundary_scattering(geometry)              # perform boundary scattering/periodicity and particle deletion

        self.refresh_temperatures(geometry, phonon)     # refresh cell temperatures

        self.lifetime_scattering(geometry, phonon)      # perform lifetime scattering

        self.fill_reservoirs(phonon)                    # refill reservoirs

        self.n_timesteps -= 1                           # -1 timestep in the counter to boundary scattering

        if  ( int(self.t/self.dt) % 100) == 0:
            print('Timestep {:>5d}'.format(int(self.t/self.dt)))
            self.write_modes_data()
        
        self.write_convergence()      # write data on file

        self.t += self.dt                               # 1 dt passed

        if len(self.rt_plot) > 0:
            self.plot_real_time(geometry)
                
    def init_plot_real_time(self, geometry):
        '''Initialises the real time plot to be updated for each timestep.'''

        if self.rt_plot[0] in ['T', 'temperature']:
            colors = self.temperatures
        elif self.rt_plot[0] in ['e', 'energy']:
            colors = self.energies
        elif self.rt_plot[0] in ['omega', 'angular_frequency']:
            colors = self.omega        

        plt.ion()

        fig = plt.figure(figsize = (8,8), dpi = 150)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )

        graph = ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], cmap = self.colormap, s = 1)

        plt.show()
        
        return graph, fig


    def plot_real_time(self, geometry, property_plot = None, colormap = 'viridis'):
        '''Updates data on the real time plot at the end of the timestep.'''
        if property_plot == None:
            property_plot = self.rt_plot[0]
        
        if colormap == None:
            colormap = self.colormap
        
        if property_plot in ['T', 'temperature']:
            colors = self.temperatures
        elif property_plot in ['e', 'energy']:
            colors = self.energies
        elif property_plot in ['omega', 'angular_frequency']:
            colors = self.omega      
        
        self.rt_graph._offsets3d = [self.positions[:,0], self.positions[:,1], self.positions[:,2]]

        # rgba_colors = matplotlib.cm.ScalarMappable(cmap = colormap).to_rgba(colors)

        # self.rt_graph._color = rgba_colors

        self.rt_fig.canvas.draw()
        self.rt_fig.canvas.flush_events()

        return

    def plot_figures(self, geometry, property_plot=['temperature'], colormap = 'viridis'):

        fig = plt.figure( figsize = (8, 8), dpi = 150 )
        n = len(property_plot)

        for i in range(n):
            if property_plot[i] in ['T', 'temperature']:
                data = self.temperatures
                figname = 'temperature'
                title = 'Temperature [K]'
            elif property_plot[i] in ['omega', 'angular_frequency']:
                data = self.omega
                figname = 'angular_frequency'
                title = 'Angular Frequency [rad/s]'
            elif property_plot[i] in ['n', 'occupation']:
                data = np.log(self.occupation)
                figname = 'occupation'
                title = 'Occupation Number'
            elif property_plot[i] in ['e', 'energy']:
                data = self.energies
                figname = 'energy'
                title = 'Energy [eV]'

            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
            graph = ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], cmap = colormap, c = data, s = 1)
            fig.colorbar(graph, orientation='horizontal')
            plt.title(title, {'size': 15} )

            plt.tight_layout()
            
            plt.savefig('figures/'+figname+'.png')

    def open_convergence(self,
                         filename,
                         timestep   = True,
                         sim_time   = True,
                         avg_energy = True,
                         N_p        = True,
                         slice_T    = True,
                         slice_phi  = True,
                         real_time  = True):

        filename = filename + '.txt'

        self.f = open(filename, 'a+')

        line = ''

        if real_time:
            line += 'Real Time                  '
        if timestep:
            line += 'Timest. '
        if sim_time:
            line += 'Simula. Time '
        if avg_energy:
            line += 'Average Ene. '
        if slice_T:
            for i in range(self.n_of_slices):
                line += 'T Sl {:>2d} '.format(i)
        if slice_phi:
            for i in range(self.n_of_slices):
                line += 'Hf Sl {:>2d} '.format(i)
        if N_p:
            line += 'No. Part. '
        
        line += '\n'

        self.f.write(line)

    def write_convergence(self,
                          timestep   = True,
                          sim_time   = True,
                          avg_energy = True,
                          N_p        = True,
                          slice_T    = True,
                          slice_phi  = True,
                          real_time  = True):
        line = ''

        if real_time:
            line += datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f ')  # real time in ISO format
        if timestep:
            line += '{:>8d} '.format( int(np.round(self.t/self.dt)) )   # timestep
        if sim_time:
            line += '{:>11.5e} '.format( self.t )   # time
        if avg_energy:
            line += '{:>11.5e} '.format( self.energies.mean() )  # average energy
        if slice_T:
            for i in range(self.n_of_slices):
                line += '{:>7.2f} '.format( self.slice_temperature[i] ) # temperature per slice
        if slice_phi:
            for i in range(self.n_of_slices):
                line += '{:>9.5f} '.format( self.slice_heat_flux[i] ) # heat flux per slice
        if N_p:
            line += '{:>10d}'.format( self.energies.shape[0] )  # number of particles
        
        line += '\n'

        self.f.writelines(line)
    
    def plot_modes_histograms(self, phonon):

        columns = 5
        rows = int(np.ceil(self.n_of_slices/columns))

        self.hist_fig = plt.figure(figsize = (columns*3, rows*3) )
        
        ax = []

        for i in range(self.n_of_slices):
            ax += [self.hist_fig.add_subplot(rows, columns, i+1)]

            data = []

            for j in range(phonon.number_of_branches):

                indexes = (self.modes[:,1] == j) & (self.slice_id == i)

                data += [self.modes[indexes, 0]]

            ax[i].hist(data, stacked = True, histtype = 'bar', density = True, bins = int(phonon.number_of_qpoints/20) )
    
        plt.tight_layout()
        plt.savefig('histograms/time_{:>.3e}'.format(self.t)+'.png')
        # plt.show()

        plt.clf()

    def write_modes_data(self):

        folder = 'modes_data'

        if folder not in os.listdir():
            os.mkdir(folder)

        filename = 'modes_data/time_{:>.3e}'.format(self.t)+'.txt'

        data = np.hstack((self.modes, self.slice_id.reshape(-1, 1)))

        np.savetxt(filename, data, '%4d %2d %2d')
                
