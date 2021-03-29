# calculations
import numpy as np

from scipy.spatial.transform import Rotation as rot

# geometry
import trimesh as tm

# other
import sys
import copy

np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=np.nan)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class that prepares and defines the geometry to be simulated.

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
        self.n_of_slices     = int(self.args.slices[0])
        self.slice_axis      = int(self.args.slices[1])

        # Processing mesh

        self.load_geo_file(self.shape)  # loading
        self.transform_mesh()           # transforming
        self.get_mesh_properties()      # defining useful properties
        self.calculate_slice_volume()


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
    
    def calculate_slice_volume(self):
        self.slice_length = self.bounds[:, self.slice_axis].ptp()/self.n_of_slices

        # For other geometries, check later:
        # trimesh.mass_properties(volume)

        if self.shape == 'cuboid':
            volume = np.ptp(self.bounds, axis = 0).prod()/self.n_of_slices
            
            self.slice_volume = np.ones(self.n_of_slices)*volume # angstrom^3

    # def set_boundary_cond(self):
                
    #     self.bound_cond = np.array(self.args.bound_cond)