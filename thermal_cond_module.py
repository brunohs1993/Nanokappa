import numpy as np
import h5py
import dpdata
import scipy.constants as ct
import phonopy
import trimesh as tm
import pymesh as pm

# from scipy.interpolate import LinearNDInterpolator, griddata

class Constants:
    def __init__(self):
        self.hbar = ct.physical_constants['Planck constant over 2 pi in eV s'][0]   # hbar in eV s
        self.kb   = ct.physical_constants['Boltzmann constant in eV/K'       ][0]   # kb in eV/K

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - 

class Geometry:
    def __init__(self, arguments):
        
        self.args = arguments

        self.shape = self.args.geometry
        self.dimensions = self.args.dimensions

    def set_dimensions(self):

        self.dimensions = np.array(self.args.dimensions)

        if self.args.geometry == 'cuboid':
            self.L_x = self.dimensions[0]
            self.L_y = self.dimensions[1]
            self.L_z = self.dimensions[2]

    def set_boundary_cond(self):
        # THINKING ABOUT IMPOSING BOUNDARY CONDITIONS IN COLLISION DETECTION: WHEN A PARTICLE
        # COLLIDES WITH A BOUNDARY, THE TEMPERATURE OF THE BOUNDARY IS IMPOSED TO THAT PARTICLE
        # (WHEN APPLICABLE). BUT FIRST I NEED TO THINK ON HOW TO DEFINE FACES.
        
        self.bound_cond = np.array(self.args.bound_cond)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - Get the wave vector in cartesian coordinates.
#   - Think in how to get the direct or cartesian coordinates of atoms positions from POSCAR file.
#   - 
#   - 
#   - 
#   - 
#   - 

class Phonon(Constants):    
    ''' Class to get phonon properties and manipulate them. '''
    def __init__(self, arguments):
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
        # self.load_energy_levels()
        
        self.load_poscar_data(self.args.poscar_file)
        self.load_atom_coord()
        self.load_atoms_types()
        self.load_lattice_vectors()

    def load_hdf_data(self, hdf_file,):
        ''' Get all data from hdf file.
            Module documentation: https://docs.h5py.org/en/stable/'''

        self.data_hdf = h5py.File(hdf_file,'r')
    
    def load_poscar_data(self, poscar_file):
        ''' Get all data from poscar file.
            Module documentation: https://pypi.org/project/dpdata/'''

        self.data_poscar = dpdata.System(poscar_file, fmt = 'vasp/poscar')

    def load_frequency(self):
        '''frequency shape = q-points X p-branches '''
        self.frequency = np.array(self.data_hdf['frequency']) # THz
    
    def convert_to_omega(self):
        '''omega shape = q-points X p-branches '''
        self.omega = self.frequency*2*ct.pi*1e12 # rad/s
    
    def load_q_points(self):
        '''q-points shape = q-points X reciprocal reduced coordinates '''
        self.q_points = np.array(self.data_hdf['qpoint'])   # reduced reciprocal coordinates
    
    def load_weights(self):
        '''weights shape = q_points '''
        self.weights = np.array(self.data_hdf['weight'])    # Question: what is this weight??

    def load_temperature(self):
        '''temperature_array shape = temperatures '''
        self.temperature_array = np.array(self.data_hdf['temperature']) # K
    
    def load_group_vel(self):
        '''groupvel shape = q_points X p-branches X cartesian coordinates '''
        self.group_vel = np.array(self.data_hdf['group_velocity'])  # THz * angstrom

    def load_heat_cap(self):
        '''heat_cap shape = temperatures X q-points X p-branches '''
        self.heat_cap = np.array(self.data_hdf['heat_capacity'])    # eV/K
    
    def load_lattice_vectors(self):
        '''Unit cell coordinates from poscar file.'''
        self.lattice_vectors = self.data_poscar['cells']    # lattice vectors in cartesian coordinates, angstrom
    
    def load_atoms_types(self):
        '''Dictionary defining elements and its quantities.'''
        self.atoms =  { self.data_poscar['atom_names'][i] : self.data_poscar['atom_numbs'][i] for i in range( len( self.data_poscar['atom_names'] ) ) }
    
    def load_atom_coord(self):
        '''Loads atom coordinates.'''
        self.atom_coord = self.data_poscar['coords'] # need to see how to read if direct or cartesian. dpdata doesn't seem to do it.

    def calculate_energy(self, T, omega):
        '''Energy of a mode given T'''
        n = self.calculate_occupation(T, omega)
        return self.hbar * omega * (n + 0.5)
    
    def crystal_energy(self, T):
        '''Calculate the total energy of the crystal considering the input modes, at a given T.'''
        return self.calculate_energy(T, self.omega).sum()
        
    def load_energy_levels(self):
        '''Calculate an array of energy levels to recalculate T'''
        self.energy_levels = self.crystal_energy(self.temperature_array)
    
    # FOR LAURENT DO EDIT:

    def build_full_bz(self, symetry):
        return



#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - Set units to the positions
#   - Drift, scatter functions
#   - Collision with trimesh ray function
#   - Transport equations
#   - 
#   - 


class Population(Constants):
    '''Class comprising the particles to be simulated.'''

    def __init__(self, arguments, geometry):
        self.args = arguments
        self.N_p  = self.args.particles
        self.dt   = self.args.timestep

        self.initialise_positions(geometry)


    def initialise_positions(self, geometry):
        '''Initialise positions of phonons: normalised coordinates'''
        if geometry.shape == 'cuboid':
            self.positions = np.random.rand(self.N_p, 3)*np.array([geometry.L_x, geometry.L_y, geometry.L_z])
    
    def atribute_modes(self, phonon):
        '''Randomly generate indexes according to an uniform distribution, linking particle positions and phonon properties.'''

        branches = phonon.omega.shape[1]    # number of branches
        q_points = phonon.omega.shape[0]    # number of q points

        self.indexes = np.zeros( (self.N_p, 2) )
        self.indexes[:,0] = np.floor( np.random.rand(self.N_p)*q_points )
        self.indexes[:,1] = np.floor( np.random.rand(self.N_p)*branches )
        self.indexes = self.indexes.astype(int)

    def atribute_properties(self, phonon):
        '''Get properties from the indexes.'''

        self.omega          = phonon.omega[ self.indexes[:,0], self.indexes[:,1] ]          # rad/s
        self.q_points       = phonon.q_points[ self.indexes[:,0] ]                          # reduced reciprocal coordinates
        self.group_vel      = phonon.group_vel[ self.indexes[:,0], self.indexes[:,1], : ]   # THz * angstrom
        self.group_vel_norm = np.norm(self.group_vel, axis = 1)                             # Absolute value of group velocity in the same unit

    def calculate_distances(self, main_position):
        '''Calculates de distance from a given phonon or group of phonons (main position) in relation to all others'''
        distances = main_position-self.positions.reshape(-1, 1, 3)
        distances = np.norm(distances, axis = 2)
        # distances = distances**2
        # distances = distances.sum(axis = 2)
        # distances = distances**(1/2)
        return distances

    def locate_all_neighbours(self, position, radius):
        '''Locates neighbours within radius for every particle.'''
        distances = self.calculate_distances(position)
        self.neighbours = (distances<=radius).astype(int)   # mask matrix classifying as neighbours (1) or not (0) all other particles (columns) for each particle (lines)
    
    def initialise_temperatures(self, geometry, key = 'random'):
        '''Atribute initial temperatures according to the geometry. Randomly within delta T unless specified otherwise.'''

        # DO THESE FUNCTIONS MATTER? THE IMPORTANT SHOULD BE THE END RESULT ANYWAY, INDEPENDENTLY OF THE STARTING DISTRIBUTION, THOUGH SURELY THEY MAY HELP CONVERGENCE SPEED.

        if   key == 'random':
            self.temperatures = np.random.rand(self.N_p)*(geometry.T_diff) + geometry.T_f
        elif key == 'linear':
            pass # need to get positions. See how to do it with different geometries.
        elif key == 'constant_hot':
            self.temperatures = np.ones(self.N_p)*geometry.T_h
        elif key == 'constant_cold':
            self.temperatures = np.ones(self.N_p)*geometry.T_c
        elif key == 'constant_mean':
            self.temperatures = np.ones(self.N_p)*geometry.T_mean
        
    def calculate_occupation(self):
        '''Occupation number of a mode given T'''
        self.occupation = 1/(np.exp(self.hbar*self.omega/self.kb * self.temperatures)-1)

    def calculate_local_energies(self):
        '''Calculates local energies for every particle'''

        self.energies = self.hbar*self.omega*(0.5+self.occupation)

    def refresh_temperatures(self, geometry):
        '''Refresh temperatures while enforcing boundary conditions as given by geometry.'''
        # DO
        pass


    def drift(self, geometry):
        '''Drift operation.'''

        self.positions += self.group_vel*self.dt*1e-12     # adjustment to angstrom/s

    def check_boundaries(self, geometry):
        '''Check boundaries and apply reflection or periodic boundary conditions.'''
        pass

    def find_collision_face(self, mesh):
        '''Finds which mesh triangle the particle will hit, given an initial position and velocity
        direction. It works with individual particles as well with a group of particles.
        Returns: array of faces indexes for each particle'''

        return mesh.ray.intersects_first(self.positions, self.group_vel)

    def calculate_time_to_collision(self, mesh, indexes = 'all'):
        '''Calculate how many timesteps to collision.
           indexes = array of indexes of the particles'''

        if indexes == 'all':
            indexes = np.arange(self.N_p)    # all particles
        
        collision_locations = map(mesh.ray.ray_triangle_id, )












        

