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
        self.rot_order       = self.args.rot_order[0]
        self.shape           = self.args.geometry[0]
        self.n_of_slices     = int(self.args.slices[0])
        self.slice_axis      = int(self.args.slices[1])
        

        # Processing mesh

        self.load_geo_file(self.shape)  # loading
        self.transform_mesh()           # transforming
        self.get_mesh_properties()      # defining useful properties
        self.slice_domain()

        print('Geometry processing done!')


    def load_geo_file(self, shape):
        '''Load the file informed in --geometry. If an standard geometry defined in __init__, adjust
        the path to load the native file. Else, need to inform the whole path.'''
        
        print('Loading geometry...')

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

        print('Transforming geometry...')

        self.mesh.rezero()  # brings mesh to origin such that all vertices are on the positive octant

        scale_matrix = np.identity(4)*np.append(self.scale, 1)

        self.mesh.apply_transform(scale_matrix) # scale mesh

        rotation_matrix       = np.zeros( (4, 4) ) # initialising transformation matrix
        rotation_matrix[3, 3] = 1

        rotation_matrix[0:3, 0:3] = rot.from_euler(self.rot_order, self.rotation).as_matrix() # building rotation terms

        self.mesh.apply_transform(rotation_matrix) # rotate mesh

    def get_mesh_properties(self):
        '''Get useful properties of the mesh.'''

        self.bounds          = self.mesh.bounds
        self.domain_centroid = self.mesh.center_mass
    
    def slice_domain(self):
        print('Slicing domain...')
        self.slice_length = self.bounds.ptp(axis = 0)[self.slice_axis]/self.n_of_slices

        self.slice_normal = np.zeros(3)
        self.slice_normal[self.slice_axis] = 1  # define the normal vector of the planes slicing the domain

        plus_normal  =  self.slice_normal
        minus_normal = -self.slice_normal

        normals = np.vstack( (plus_normal, minus_normal) )

        self.slice_meshes = []
        self.slice_bounds = np.empty( (self.n_of_slices, 2, 3 ) )
        self.slice_volume = np.empty(self.n_of_slices)
        self.slice_center = np.empty((self.n_of_slices, 3))

        for i in range(self.n_of_slices):

            plus_origin                   = np.zeros(3)
            plus_origin[self.slice_axis]  =    i *self.bounds.ptp(axis = 0)[self.slice_axis]/self.n_of_slices
            
            minus_origin                  = np.zeros(3)
            minus_origin[self.slice_axis] = (i+1)*self.bounds.ptp(axis = 0)[self.slice_axis]/self.n_of_slices

            origins = np.vstack( (plus_origin, minus_origin) )
            
            slice_mesh =  self.mesh.slice_plane( origins, normals, cap = True)    # slicing to the positive

            self.slice_meshes.append(slice_mesh)
            self.slice_volume[i] = slice_mesh.volume
            self.slice_center[i, :] = slice_mesh.center_mass
            self.slice_bounds[i, :, :] = slice_mesh.bounds

    # def set_boundary_cond(self):
                
    #     self.bound_cond = np.array(self.args.bound_cond)