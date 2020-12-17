import numpy as np
import h5py
import dpdata
import scipy.constants as ct
import phonopy
import pymesh as pm # https://pymesh.readthedocs.io/en/latest/installation.html

# from scipy.interpolate import LinearNDInterpolator, griddata

class Constants:
    def __init__(self):
        self.hbar = ct.physical_constants['Planck constant over 2 pi in eV s'][0]   # hbar in eV s
        self.kb   = ct.physical_constants['Boltzmann constant in eV/K'       ][0]   # kb in eV/K

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - Using pymesh, maybe this becomes unnecessary?

class Geometry:
    def __init__(self, arguments):
        
        self.args = arguments

        self.standard_shapes = ['cuboid', 'cillinder', 'sphere']
        self.dimensions = self.args.dimensions

    def load_geo_file(self):
        '''Load the file informed in --geometry.'''
        
        self.shape = self.args.geometry

        if self.shape in self.standard_shapes:
            self.shape = 'std_geo/'+self.shape+'.stl'
        
        self.mesh = pm.load_mesh(self.shape)

    def get_mesh_properties(self):
        self.bbox = self.mesh.bbox
        self.

    def set_dimensions(self):

        self.dimensions = np.array(self.args.dimensions)

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

    # FOR LAURENT TO BUILD FULL BRELLOUIN ZONE
    # My idea is to have a method to return all properties (omega, q_points, etc) in the fbz

    def build_fbz(self):
        
        self.omega = 
        
        return 
    
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   TO DO
#
#   - Drift, scatter functions
#   - Collision detection with boundaries to impose boundary conditions. How to detect collision efficiently?
#   - Transport equations
#   - 
#   - 


class Population(Constants):
    '''Class comprising the particles to be simulated.'''

    def __init__(self, arguments, geometry):
        self.args = arguments
        self.N_p = self.args.particles

        self.initialise_positions(geometry)

    def initialise_positions(self, geometry):
        '''Initialise positions of phonons: normalised coordinates'''
        
        # Get geometry bounding box
        bbox = geometry.bbox
        bbox_range = bbox[1,:]- bbox[0,:]

        self.positions = np.random.rand(self.N_p, 3)*bbox_range + bbox[0,:] # Generate random points inside bounding box
        self.positions, points_out = self.remove_out_points(geometry)       # Remove points outside mesh

        while self.positions.shape[0] != self.N_p:
            # Keep generating points until all of them are inside the mesh
            new_positions = np.random.rand(points_out, 3)*bbox_range + bbox[0,:] # generate new
            new_positions, points_out = self.remove_out_points(geometry)         # check which are inside
            self.positions = np.append(self.positions, new_positions, 0)         # add the good ones

    def remove_out_points(self, geometry):
        '''Remove points outside the geometry mesh.'''
        out_points = pm.compute_winding_number(geometry, self.positions)
        indexes = np.where(out_points == 0)[0]
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

        self.omega       = phonon.omega[ self.indexes[:,0], self.indexes[:,1] ]         # rad/s
        self.wavevectors = phonon.q_points[ self.indexes[:,0] ]                         # reduced reciprocal coordinates
        self.velocities  = phonon.group_vel[ self.indexes[:,0], self.indexes[:,1], : ]  # THz * angstrom

    def calculate_distances(self, main_position):
        '''Calculates de distance from a given phonon or group of phonons (main position) in relation to all others'''
        distances = main_position-self.positions.reshape(-1, 1, 3)
        distances = distances**2
        distances = distances.sum(axis = 2)
        distances = distances**(1/2)
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
        self.occupation = 1/(np.exp( self.hbar*self.omega / (self.kb * self.temperatures) )-1)

    def calculate_local_energies(self):
        '''Calculates local energies for every particle'''

        self.energies = self.hbar*self.omega*(0.5+self.occupation) 

    def refresh_temperatures(self, geometry):
        '''Refresh temperatures while enforcing boundary conditions as given by geometry.'''
        # DO
        pass


    def drift(self, geometry):
        '''Drift operation.'''

        self.positions += self.velocities*geometry.dt*1e-12     # adjustment to angstrom/s

    def check_boundaries(self, geometry):
        '''Check boundaries and apply reflection or periodic boundary conditions.'''
        pass






        

