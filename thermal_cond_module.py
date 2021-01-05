import numpy as np
import h5py
import scipy.constants as ct
from scipy.spatial.transform import Rotation as rot
import phonopy
import trimesh as tm
import pyiron

# from scipy.interpolate import LinearNDInterpolator, griddata

class Constants:
    def __init__(self):
        self.hbar = ct.physical_constants['Planck constant over 2 pi in eV s'][0]   # hbar in eV s
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
        self.rotation        = np.array(self.args.rotation)
        self.rot_order       = self.args.rot_order

        # Processing mesh

        self.load_geo_file          # loading
        self.transform_mesh         # transforming
        self.get_mesh_properties    # defining useful properties


    def load_geo_file(self):
        '''Load the file informed in --geometry. If an standard geometry defined in __init__, adjust
        the path to load the native file. Else, need to inform the whole path.'''
        
        self.shape = self.args.geometry

        if self.shape in self.standard_shapes:
            self.shape = 'std_geo/'+self.shape+'.stl'
        
        self.mesh = tm.load(self.shape)

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

    def set_boundary_cond(self):
                
        self.bound_cond = np.array(self.args.bound_cond)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - Get the wave vector in cartesian coordinates.
#   - Think in how to get the direct or cartesian coordinates of atoms positions from POSCAR file.
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
        self.load_gamma()
        self.calculate_lifetime()
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
        ''' Get structure from poscar file.
            Module documentation: https://pyiron.readthedocs.io/en/latest/index.html'''

        self.data_poscar = pyiron.vasp.structure.read_atoms(poscar_file)

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
        
    def load_energy_levels(self):
        '''Calculate an array of energy levels to recalculate T'''
        self.energy_levels = self.crystal_energy(self.temperature_array)

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
        self.args = arguments
        self.N_p  = self.args.particles
        self.dt   = self.args.timestep
        self.t    = 0

        self.initialise_positions(geometry) # initialising particles
        self.atribute_modes(phonon)         # setting their properties
        self.calculate_ts_lifetime(geometry)  # calculating collisions


    def initialise_positions(self, geometry):
        '''Initialise positions of phonons: normalised coordinates'''
        
        # Get geometry bounding box
        bounds = geometry.bounds
        bounds_range = bounds[1,:]- bounds[0,:]

        self.positions = np.random.rand(self.N_p, 3)*bounds_range + bounds[0,:] # Generate random points inside bounding box
        self.positions, points_out = self.remove_out_points(geometry)       # Remove points outside mesh

        while self.positions.shape[0] != self.N_p:
            # Keep generating points until all of them are inside the mesh
            new_positions = np.random.rand(points_out, 3)*bounds_range + bounds[0,:] # generate new
            new_positions, points_out = self.remove_out_points(geometry)         # check which are inside
            self.positions = np.append(self.positions, new_positions, 0)         # add the good ones

    def remove_out_points(self, geometry):
        '''Remove points outside the geometry mesh.'''
        out_points = geometry.mesh.contains(self.positions)
        indexes = np.where(out_points == False)[0]
        return np.delete(self.positions, indexes, 0), np.indexes.shape[0]
    
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
        self.lifetime       = phonon.lifetime[ :, self.indexes[:,0], self.indexes[:,1] ]    # PENSAR NA TEMPERATURA

    def calculate_distances(self, main_position, mesh):
        '''Calculates de distance from a given phonon or group of phonons (main position) in relation to all others'''
        distances = main_position-self.positions.reshape(-1, 1, 3)  # difference vector
        
        _, indexes, _ = mesh.intersects_id(main_position,
                                           distances,
                                           return_locations=False,
                                           multiple_hits=False)    # check whether there is a wall in between
        
        distances = np.norm(distances, axis = 2)    # compute euclidean norms



        return distances

    def locate_all_neighbours(self, position, radius):
        '''Locates neighbours within radius for every particle.'''
        distances = self.calculate_distances(position)
        return (distances<=radius).astype(int)   # mask matrix classifying as neighbours (1) or not (0) all other particles (columns) for each particle (lines)
    
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
        
    def calculate_n_from_T(self, omega, temperature):
        '''Occupation number of a mode given T'''
        return 1/(np.exp( self.hbar*omega / (self.kb * temperature) )-1)
    
    def calculate_n_from_E(self, omega, energy):
        '''Occupation number of a mode given E'''
        return energy/(self.kb*self.hbar) - 0.5

    def calculate_T_from_n(self, omega, occupation):
        '''Occupation number of a mode given E'''
        return (self.hbar*omega/self.kb) * 1/np.log(1/occupation + 1)

    def refresh_temperatures(self, geometry):
        '''Refresh temperatures while enforcing boundary conditions as given by geometry.'''
        # DO
        pass


    def drift(self, geometry):
        '''Drift operation.'''

        self.positions += self.group_vel*self.dt*1e-12     # adjustment to angstrom/s
        self.n_timesteps -= 1

    def check_boundaries(self, geometry):
        '''Check boundaries and apply reflection or periodic boundary conditions.'''
        pass

    def find_collision(self, positions, velocities, mesh):
        '''Finds which mesh triangle will be hit by the particle, given an initial position and velocity
        direction. It works with individual particles as well with a group of particles.
        Returns: array of faces indexes for each particle'''

        _, _, coll_pos = mesh.intersects_id(positions, velocities, return_locations=True, multiple_hits=False)

        return coll_pos

    def timesteps_to_collision(self, positions, coll_pos, velocities, mesh):
        '''Calculate how many timesteps to collision.
           indexes = array of indexes of the particles'''
        
        # calculate distances for given particles
        coll_dist = np.norm( positions - coll_pos, axis = 1 )        

        ts_to_collision = coll_dist/( np.norm(velocities, axis = 1) * self.dt )

        ts_to_collision = np.ceil(ts_to_collision)    # such that particle collides when no_timesteps == 0 (crosses boundary)

        return ts_to_collision.astype(int)
    
    def timesteps_to_scatter(self, lifetimes):
        return np.ceil(self.dt/lifetimes).astype(int)
    
    def calculate_ts_lifetime(self, geometry, indexes='all'):
        '''Calculate number of timesteps until annihilation for new particles.'''

        if indexes == 'all':
            indexes = np.arange(self.N_p)
            self.n_timesteps = np.zeros(self.N_p)

        new_coll_pos = self.find_collision(self.positions[indexes, :], self.group_vel[indexes, :], geometry.mesh)   # finds collision points

        new_ts_to_collision = self.timesteps_to_collision(self.positions[indexes, :], new_coll_pos, self.group_vel[indexes, :], geometry.mesh)  # calculates ts until collision

        new_ts_to_scatter   = self.timesteps_to_scatter(self.lifetime[indexes]) # calculates ts until scatter according to lifetime

        new_ts = np.column_stack( (new_ts_to_collision, new_ts_to_scatter) )    # groups both arrays

        self.n_timesteps[indexes] = new_ts.min(axis=1)  # takes which happens first, collision or scattering

    def distribute_energies(self, indexes, radius):
        '''Distribute their energies of scattered particles.'''

        scattered_positions = self.positions[indexes, :]                        # get their positions
        scattered_energies  = self.energies[indexes]                            # get their energies

        neighbours = self.locate_all_neighbours(scattered_positions, radius)    # find their neighbours
        neighbours[:, indexes] = 0                                              # removing scattering neighbours
        neighbours = neighbours.astype(bool)                                    # converting to boolean

        n_neighbours = neighbours.sum(axis=1)                                   # number of neighbours for each scattered particle

        add_energy = scattered_energies/n_neighbours                            # calculate energy addition from each scattering particle
        add_energy = add_energy.reshape(-1, 1)                                  # reshape to column array

        add_energy = neighbours*add_energy                                      # set the contribution to each neighbour
        add_energy = add_energy.sum(axis = 0)                                   # add all contributions for each non-scattered particle
        add_energy = add_energy.reshape(-1, 1)                                  # reshape to column array

        return self.energies + add_energy                                       # change energy level of particles

    def scatter(self, scatter_radius = 1):
        '''Checks which particles have to be collided or scattered and distribute its energies around.'''

        # THINK ABOUT HOW TO CONSERVE MOMENTUM, UMKLAPP SCATTERING AND BOUNDARY EFFECTS

        indexes = np.where(self.n_timesteps == 0, True, False)[0] # define which particles to be scattered (boolean array)

        self.energies     = self.distribute_energies(indexes, radius = scatter_radius)  # update energies

        self.delete_particles(indexes)

        # Do i need this? Because I have to update local temperature anyway...
        # self.occupation   = self.calculate_n_from_E(self.omega, self.energies)          # update occupation from energies
        # self.temperatures = self.calculate_T_from_n(self.omega, self.occupation)        # update temperature from occupation

    def delete_particles(self, indexes):
        self.positions    = self.positions[~indexes]
        self.q_points     = self.q_points[~indexes]
        self.group_vel    = self.group_vel[~indexes]
        self.omega        = self.omega[~indexes]
        self.lifetime     = self.lifetime[~indexes]
        self.energies     = self.energies[~indexes]
        self.occupation   = self.occupation[~indexes]
        self.temperatures = self.temperatures[~indexes]
        


    def generate_new_particles(self):
        '''Different from initialise_particles. This one generates particles at the heated boundary.'''
        # THINK ABOUT HOW TO DO THIS FOR SEVERAL BOUNDARY TEMPERATURES/FLUXES
        # MAYBE EXAMPLE:
        # - TEMPERATURES = 300, 250, 350
        # - REDUCES MINIMUM = 50, 0, 100
        # - GENERATION WILL BE = 1/3 , 0, 2/3

        number = int(self.N_p-self.positions.shape[0])

        # GENERATE PARTICLES AT THE BOUNDARY

        # ADD PARTICLES TO SELF.POSITIONS
        # RETRIEVE PROPERTIES

        pass

    def run_timestep(self, geometry):

        self.drift(geometry)        # drift particles
        self.scatter()              #

    

        self.generate_new_particles()

        



        self.n_timesteps -= 1      # -1 timestep to collision

        self.t += self.dt           # 1 dt passed

        return


















        

