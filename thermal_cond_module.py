
# calculations
import numpy as np
# import numpy.linalg

import scipy.constants as ct
from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import interp1d

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# hdf5 files
import h5py

# crystal structure
import phonopy
import pyiron.vasp.structure

# geometry
import trimesh as tm
import trimesh.ray.ray_pyembree

# other
import sys


np.set_printoptions(precision=3, threshold=sys.maxsize)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

class Constants:
    def __init__(self):
        self.hbar = ct.physical_constants['reduced Planck constant in eV s'  ][0]   # hbar in eV s
        # self.hbar = ct.physical_constants['Planck constant over 2 pi in eV s'][0]   # hbar in eV s
        self.kb   = ct.physical_constants['Boltzmann constant in eV/K'       ][0]   # kb in eV/K

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - Define boundary conditions

class Geometry:
    def __init__(self, arguments):
        
        self.args = arguments

        self.standard_shapes = ['cuboid', 'cillinder', 'sphere']
        self.scale           = self.args.scale
        self.dimensions      = self.args.dimensions
        self.rotation        = np.array(self.args.rotation)
        self.rot_order       = self.args.rot_order
        self.shape           = self.args.geometry

        # Processing mesh

        self.load_geo_file(self.shape)          # loading
        self.transform_mesh()         # transforming
        self.get_mesh_properties()    # defining useful properties


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

        self.load_hdf_data(self.args.hdf_file)
        self.load_frequency()
        self.convert_to_omega()
        
        self.load_q_points()
        self.load_weights()
        self.load_group_vel()
        self.load_temperature()
        self.load_heat_cap()
        self.load_gamma()
        
        self.rank_energies()
        self.calculate_lifetime()
        self.initialise_temperature_function()
        
        self.load_poscar_data(self.args.poscar_file)
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
        self.number_of_qpoints = self.frequency.shape[0]
        self.number_of_branches = self.frequency.shape[1]
        self.number_of_modes = self.number_of_qpoints*self.number_of_branches

    def convert_to_omega(self):
        '''omega shape = q-points X p-branches '''
        self.omega = self.frequency*2*ct.pi*1e12 # rad/s
    
    def rank_energies(self):
        matrix = self.calculate_energy(100, self.omega)
        self.rank = matrix.argsort(axis = None).argsort().reshape(matrix.shape)

    def load_q_points(self):
        '''q-points shape = q-points X reciprocal reduced coordinates '''
        self.q_points = np.array(self.data_hdf['qpoint'])   # reduced reciprocal coordinates
        self.number_of_qpoints = self.q_points.shape[0]
    
    def load_weights(self):
        '''weights shape = q_points '''
        self.weights = np.array(self.data_hdf['weight'])

    def load_temperature(self):
        '''temperature_array shape = temperatures '''
        self.temperature_array = np.array(self.data_hdf['temperature']) # K
    
    def load_group_vel(self):
        '''groupvel shape = q_points X p-branches X cartesian coordinates '''
        self.group_vel = np.array(self.data_hdf['group_velocity'])  # THz * angstrom

    def load_heat_cap(self):
        '''heat_cap shape = temperatures X q-points X p-branches '''
        self.heat_cap = np.array(self.data_hdf['heat_capacity'])    # eV/K
    
    def load_gamma(self):
        '''gamma = temperatures X q-pointsX p-branches'''
        self.gamma = np.array(self.data_hdf['gamma'])   # THz

    def calculate_lifetime(self):
        '''lifetime = temperatures X q-pointsX p-branches'''
        self.lifetime = np.where( self.gamma>0, 1/( 2*2*np.pi*self.gamma), 0)*1e12  # s
        self.lifetime_function = [[interp1d(self.temperature_array, self.lifetime[:, i, j]) for j in range(self.number_of_branches)] for i in range(self.number_of_qpoints)]
        
    def load_lattice_vectors(self):
        '''Unit cell coordinates from poscar file.'''
        self.lattice_vectors = self.data_poscar.cell    # lattice vectors in cartesian coordinates, angstrom
    
    def calculate_occupation(self, T, omega):
        occupation = np.where(T==0, 0, 1/( np.exp( omega*self.hbar/ (T*self.kb) ) - 1) )
        return occupation

    def calculate_energy(self, T, omega):
        '''Energy of a mode given T'''
        n = self.calculate_occupation(T, omega)
        return self.hbar * omega * (n + 0.5)
    
    def calculate_crystal_energy(self, T):
        '''Calculates the average energy per phonon at a given temperature for the crystal.'''

        T = np.array(T)             # ensuring right type
        T = T.reshape( (-1, 1, 1) ) # ensuring right dimensionality

        return self.calculate_energy(T, self.omega).mean( axis = (1, 2) )

    def initialise_temperature_function(self):
        '''Calculate an array of energy levels and initialises the function to recalculate T = f(E)'''

        self.energy_array = self.calculate_crystal_energy( self.temperature_array )
        self.temperature_function = interp1d( self.energy_array, self.temperature_array )

    # FOR LAURENT TO BUILD FULL BRELLOUIN ZONE
    # My idea is to have a method to return all properties (omega, q_points, etc) in the fbz

    def build_fbz(self):        
        return 
    
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - Conservation of momentum (wavevector), umklapp scattering
#   - Time
#   - Refresh temperature

class Population(Constants):
    '''Class comprising the particles to be simulated.'''

    def __init__(self, arguments, geometry, phonon):
        super(Population, self).__init__()

        self.args          = arguments
        self.N_p           = int(self.args.particles[0])
        self.dt            = self.args.timestep
        self.n_of_slices   = self.args.slices[0]
        self.slice_axis    = self.args.slices[1]
        self.slice_length  = geometry.bounds[:, self.slice_axis].ptp()/self.n_of_slices
            
        self.T_boundary     = np.array(self.args.temperatures)
        self.T_distribution = self.args.temp_dist

        self.t             = 0

        self.initialise_all_particles(phonon, geometry) # initialising particles
        self.calculate_ts_lifetime(geometry)            # calculating timesteps [collision, scatter]

    def initialise_modes(self, number_of_particles, phonon):
        '''Uniformily organise modes for a given number of particlesprioritising low energy ones.'''

        part_p_mode = np.ceil(number_of_particles/phonon.number_of_modes)    # particles per mode

        surplus = int(part_p_mode*phonon.number_of_modes-number_of_particles)    # excess of particles due to rounding

        surplus_modes = np.where(phonon.rank.reshape(-1)>(phonon.rank.max()-surplus))[0]    # modes to correct based on rank

        ppm_array = (np.ones(phonon.number_of_modes)*part_p_mode).astype(int)

        ppm_array[surplus_modes] -= 1   # corrected number of particles per mode

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

        positions = np.random.rand(number_of_particles, 3)*bounds_range + bounds[0,:] # Generate random points inside bounding box
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
        self.omega, self.q_points, self.group_vel, self.lifetime, self.lifetime_functions = self.assign_properties(self.modes, self.temperatures, phonon)
                
        self.energies     = phonon.calculate_energy(self.temperatures, self.omega)

        
    def assign_properties(self, modes, temperatures, phonon):
        '''Get properties from the indexes.'''

        omega              = phonon.omega[ modes[:,0], modes[:,1] ]        # rad/s
        q_points           = phonon.q_points[ modes[:,0] ]                 # reduced reciprocal coordinates
        group_vel          = phonon.group_vel[ modes[:,0], modes[:,1], : ] # THz * angstrom

        lifetime_functions = [phonon.lifetime_function[modes[i,0]][modes[i,1]] for i in range(modes.shape[0])]  # functions of lifetime for each particle
        lifetime = np.array([lifetime_functions[i](temperatures[i]) for i in range(temperatures.shape[0])])

        return omega, q_points, group_vel, lifetime, lifetime_functions
   
    def assign_temperatures(self, positions, geometry):
        '''Atribute initial temperatures imposing fixed temperatures on first and last slice. Randomly within delta T unless specified otherwise.'''

        number_of_particles = positions.shape[0]
        key = self.T_distribution

        temperatures = np.empty(number_of_particles)    # initialise temperature array

        if key == 'linear':
            # calculates T for each slice
            T_array = np.arange(self.T_boundary[0], self.T_boundary[1]+self.T_boundary.ptp()/self.n_of_slices, self.T_boundary.ptp()/self.n_of_slices)

            for i in range(self.n_of_slices):   # assign temperatures for particles in each slice
                indexes = (positions[:, self.slice_axis] >= i*self.slice_length) & (positions[:, self.slice_axis] < (i+1)*self.slice_length)
                temperatures[indexes] = T_array[i]
        
        else:
            if   key == 'random':
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

        print(temperatures)
        return temperatures
        
    def refresh_temperatures(self, geometry):
        '''Refresh temperatures while enforcing boundary conditions as given by geometry.'''
        # DO
        pass

    def drift(self, geometry):
        '''Drift operation.'''

        self.positions += self.group_vel*self.dt*1e-12     # adjustment to angstrom/s
        self.n_timesteps -= 1

    def find_collision(self, positions, velocities, geometry):
        '''Finds which mesh triangle will be hit by the particle, given an initial position and velocity
        direction. It works with individual particles as well with a group of particles.
        Returns: array of faces indexes for each particle'''

        coll_pos, index_ray, _ = geometry.ray.intersects_location(positions, velocities, multiple_hits=False)

        stationary = ~np.in1d(np.arange(positions.shape[0]), index_ray)   # find which particles have zero velocity, so no ray

        all_coll_pos                = np.zeros( (positions.shape[0], 3) )
        all_coll_pos[stationary, :] = np.inf
        all_coll_pos[~stationary, :]= coll_pos

        return all_coll_pos

    def timesteps_to_collision(self, positions, velocities, geometry):
        '''Calculate how many timesteps to collision.'''
        
        coll_pos = self.find_collision(positions, velocities, geometry)

        # calculate distances for given particles
        coll_dist = np.linalg.norm( positions - coll_pos, axis = 1 )        

        ts_to_collision = coll_dist/( np.linalg.norm(velocities, axis = 1) * self.dt )

        ts_to_collision = np.ceil(ts_to_collision)    # such that particle collides when no_timesteps == 0 (crosses boundary)

        return ts_to_collision.astype(int)
    
    def timesteps_to_scatter(self, lifetimes):
        '''Calculate the number of timestepps until a particle scatters according to its lifetime.'''
        return np.ceil(self.dt/lifetimes).astype(int)   # such that particle scatters when no_timesteps == 0 (decays)
    
    def calculate_ts_lifetime(self, geometry, indexes='all'):
        '''Calculate number of timesteps until annihilation for particles given their indexes in the population.'''

        if indexes == 'all':
            indexes = np.arange(self.N_p)
            self.n_timesteps = np.empty( (self.N_p, 2) )
        
        new_ts_to_collision = self.timesteps_to_collision(self.positions[indexes, :], self.group_vel[indexes, :], geometry.mesh)  # calculates ts until collision

        new_ts_to_scatter   = self.timesteps_to_scatter(self.lifetime[indexes]) # calculates ts until scatter according to lifetime

        self.n_timesteps[indexes, 0] = new_ts_to_collision  # saves collision timestep counter
        self.n_timesteps[indexes, 1] = new_ts_to_scatter    # saves scatter timestep counter

    def delete_particles(self, indexes):
        '''Delete all information about particles according to the given indexes'''

        self.positions    = self.positions[~indexes]
        self.q_points     = self.q_points[~indexes]
        self.group_vel    = self.group_vel[~indexes]
        self.omega        = self.omega[~indexes]
        self.lifetime     = self.lifetime[~indexes]
        self.energies     = self.energies[~indexes]
        self.occupation   = self.occupation[~indexes]
        self.temperatures = self.temperatures[~indexes]
    
    def generate_new_particles(self, number_of_particles, geometry, slice_index = 'all'):
        
        if slice_index == 'all':
            positions = np.random.rand(number_of_particles)

    def run_timestep(self, geometry):

        self.drift(geometry)        # drift particles
        # self.scatter()              #

    

        self.generate_new_particles()

        



        self.n_timesteps -= 1      # -1 timestep to collision

        self.t += self.dt           # 1 dt passed

        return
    
    def plot_population(self, property_plot=['temperature'], colormap = 'viridis'):

        n = len(property_plot)

        fig = plt.figure( figsize=(10,n*5) )
        

        for i in range(n):
            if property_plot[i] == 'temperature':
                ax = fig.add_subplot(1, n+1, i+1, projection='3d')
                graph = ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], cmap = colormap, c = self.temperatures, s = 1)
                fig.colorbar(graph)
        
        plt.tight_layout()
        
        plt.savefig('figure.png')
















    # POSSIBLY DELETED CODE

    # def calculate_distances(self, main_position, mesh):
    #     '''Calculates de distance from a given phonon or group of phonons (main position) in relation to all others'''
    #     distances = main_position-self.positions.reshape(-1, 1, 3)  # difference vector
        
    #     # _, indexes, _ = mesh.intersects_id(main_position,
    #     #                                    distances,
    #     #                                    return_locations=False,
    #     #                                    multiple_hits=False)    # check whether there is a wall in between
        
    #     distances = np.norm(distances, axis = 2)    # compute euclidean norms



    #     return distances

    # def locate_all_neighbours(self, position, radius):
    #     '''Locates neighbours within radius for every particle.'''
    #     # import shape to use mesh.contain to locate neighbours

    #     distances = self.calculate_distances(position)
    #     return (distances<=radius).astype(int)   # mask matrix classifying as neighbours (1) or not (0) all other particles (columns) for each particle (lines)


    # def distribute_energies(self, indexes, radius):
    #     '''Distribute their energies of scattered particles.'''

    #     scattered_positions = self.positions[indexes, :]                        # get their positions
    #     scattered_energies  = self.energies[indexes]                            # get their energies

    #     neighbours = self.locate_all_neighbours(scattered_positions, radius)    # find their neighbours
    #     neighbours[:, indexes] = 0                                              # removing scattering neighbours
    #     neighbours = neighbours.astype(bool)                                    # converting to boolean

    #     n_neighbours = neighbours.sum(axis=1)                                   # number of neighbours for each scattered particle

    #     add_energy = scattered_energies/n_neighbours                            # calculate energy addition from each scattering particle
    #     add_energy = add_energy.reshape(-1, 1)                                  # reshape to column array

    #     add_energy = neighbours*add_energy                                      # set the contribution to each neighbour
    #     add_energy = add_energy.sum(axis = 0)                                   # add all contributions for each non-scattered particle
    #     add_energy = add_energy.reshape(-1, 1)                                  # reshape to column array

    #     return self.energies + add_energy                                       # change energy level of particles

    # def scatter(self, scatter_radius = 1):
    #     '''Checks which particles have to be collided or scattered and distribute its energies around.'''

    #     # THINK ABOUT HOW TO CONSERVE MOMENTUM, UMKLAPP SCATTERING AND BOUNDARY EFFECTS

    #     indexes = np.where(self.n_timesteps == 0, True, False)[0] # define which particles to be scattered (boolean array)

    #     self.energies     = self.distribute_energies(indexes, radius = scatter_radius)  # update energies

    #     self.delete_particles(indexes)

    #     # Do i need this? Because I have to update local temperature anyway...
    #     # self.occupation   = self.calculate_n_from_E(self.omega, self.energies)          # update occupation from energies
    #     # self.temperatures = self.calculate_T_from_n(self.omega, self.occupation)        # update temperature from occupation

    # def calculate_n_from_T(self, omega, temperature):
    #     '''Occupation number of a mode given T'''
    #     return 1/(np.exp( self.hbar*omega / (self.kb * temperature) )-1)
    
    # def calculate_n_from_E(self, omega, energy):
    #     '''Occupation number of a mode given E'''
    #     return energy/(self.kb*self.hbar) - 0.5

    # def calculate_T_from_n(self, omega, occupation):
    #     '''Occupation number of a mode given E'''
    #     return (self.hbar*omega/self.kb) * 1/np.log(1/occupation + 1)

    # def check_boundaries(self, geometry):
    #     '''Check boundaries and apply reflection or periodic boundary conditions.'''
    #     pass












        

