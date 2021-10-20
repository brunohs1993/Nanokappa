# calculations
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as rot

# geometry
import trimesh as tm
from trimesh.graph import is_watertight
import routines.subvolumes as subvolumes

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
    def __init__(self, args):
        
        self.standard_shapes = ['cuboid', 'cillinder', 'sphere']
        self.scale           = args.scale
        self.dimensions      = args.dimensions             # angstrom
        self.rotation        = np.array(args.geo_rotation[:-1]).astype(float)
        self.rot_order       = args.geo_rotation[-1]
        self.shape           = args.geometry[0]
        self.subvol_type     = args.subvolumes[0]
        self.n_of_subvols    = int(args.subvolumes[1])
        self.folder          = args.results_folder
        
        if self.subvol_type == 'slice':
            self.slice_axis      = int(args.subvolumes[2])

        # Processing mesh

        self.load_geo_file(self.shape) # loading
        self.transform_mesh()          # transforming
        self.get_outer_hull(args)      # getting the outer hull, making it watertight and correcting bound_facets
        self.get_mesh_properties()     # defining useful properties
        self.get_bound_facets(args)    # get boundary conditions facets
        self.check_connections(args)   # check if all connections are valid
        self.set_subvolumes()          # define subvolumes and save their meshes and properties

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

        rotation_matrix[0:3, 0:3] = rot.from_euler(self.rot_order, self.rotation, degrees = True).as_matrix() # building rotation terms

        self.mesh.apply_transform(rotation_matrix) # rotate mesh

    def get_outer_hull(self, args, view = False):
        '''This ensures that the mesh does not have internal faces and gets only the external ones.
        It works by generating rays in a box around the mesh, and registering which faces these rays hit on.
        When the mesh is filled and can be considered watertight, it stops and substitutes the old mesh by the new one.'''

        print('Making mesh watertight...')
        # bounding box mesh
        box = tm.creation.box(extents = self.mesh.extents+0.2)
        box.rezero()
        box.vertices -= 0.1

        # if mesh is composed of more than one mesh, add them up
        if isinstance(self.mesh, tm.Scene):
            print(self.mesh.dump(concatenate=True))
        else:
            submesh = self.mesh

        # get external faces of the mesh to make it watertight
        exponent = 4
        known_faces = np.empty(0)
        i = 0

        print('Number of bodies: ', self.mesh.body_count)

        while (not submesh.is_watertight) and i<1000:
            
            i = i+1
            print('Try', i)

            n_points = 10**exponent

            # sampling points on bounding box
            start_points, _ = tm.sample.sample_surface(box, n_points)
            end_points  , _ = tm.sample.sample_surface(box, n_points)
            directions = end_points - start_points

            # detecting hit faces
            tri, _ = self.mesh.ray.intersects_id(ray_origins = start_points, ray_directions = directions, multiple_hits = False)

            # get all unique faces that were hit
            valid_faces = np.unique(tri[tri >= 0])

            # add detected faces to the external ones
            known_faces = np.unique(np.concatenate((known_faces, valid_faces))).astype(int)

            # generate submesh with known faces
            submesh = tm.util.submesh(mesh = self.mesh, faces_sequence = [known_faces], append = True)
            
            if view:
                submesh.show()

            # Try to make submesh watertight
            try:
                submesh.process(validate=True)
                submesh.fill_holes()
                submesh.fix_normals()
            except:
                pass
            # print(known_faces.shape[0], len(self.mesh.faces), submesh.is_watertight)

        # finalise submesh
        submesh.process(validate=True)
        submesh.fill_holes()
        submesh.fix_normals()

        # referencing vertices
        vertices_indexes = []

        for i in range(submesh.vertices.shape[0]):                                  # for each vertex in the new mesh
            vertex = submesh.vertices[i, :]                                         # get its coordinates
            index = np.where(np.all(self.mesh.vertices == vertex, axis = 1))[0]     # check where they are in the old mesh
            vertices_indexes.append(index)                                          # append the index to find them in the old mesh
        
        vertices_indexes = np.array(vertices_indexes)

        # referencing faces
        faces_indexes = []

        for i in range(submesh.faces.shape[0]):                                                     # check for each snew_mesh face
            new_face        = submesh.faces[i, :]                                                   # the vertices in the new mesh that compose the face - ex: [0, 1, 2]
            vertices_in_old = vertices_indexes[new_face]                                            # and how they are referenced in the old mesh
                                                                                                    # ex: [5, 2, 8] = [vertices_indexes[0], vertices_indexes[1], vertices_indexes[2]]
            for j in range(self.mesh.faces.shape[0]):                                               # Then for each face in the old mesh
                old_face = self.mesh.faces[j, :]                                                    # get its vertices
                is_in = np.array([vertex in old_face for vertex in vertices_in_old], dtype = bool)  # and check if all vertices match
                if np.all(is_in):                                                                   # If they do
                    faces_indexes.append(j)                                                         # register the index
                    break                                                                           # break the loop and check next face in new mesh.
        
        faces_indexes = np.array(faces_indexes)

        facets_indexes = []
        # referencing facets
        for i in range(len(submesh.facets)):                                                    # Check for each facet in the new mesh
            new_facet    = submesh.facets[i]                                                    # which are the faces that compose the facet
            faces_in_old = faces_indexes[new_facet]                                             # and how they are referenced in the old mesh.
            for j in range(len(self.mesh.facets)):                                              # Then for each facet in the old mesh
                old_facet = self.mesh.facets[j]                                                 # get its faces
                is_in = np.array([face in old_facet for face in faces_in_old], dtype = bool)    # and check if all faces match
                if np.all(is_in):                                                               # If they do
                    facets_indexes.append(j)                                                    # register the index
                    break                                                                       # break the loop and check next face in new mesh.

        args.bound_facets   = [facets_indexes.index(facet) for facet in args.bound_facets]    # correcting input arguments for boundary conditions
        args.connect_facets = [facets_indexes.index(facet) for facet in args.connect_facets]

        if view:
            submesh.show()

        # substitute geometry with submesh
        self.mesh = submesh

        print('Watertight?', self.mesh.is_watertight)

    def get_mesh_properties(self):
        '''Get useful properties of the mesh.'''

        self.bounds           = self.mesh.bounds
        self.domain_centroid  = self.mesh.center_mass
        self.volume           = self.mesh.volume
        self.number_of_facets = len(self.mesh.facets)
    
    def set_subvolumes(self):
        print('Generating subvolumes...')

        if self.subvol_type == 'slice':

            self.slice_length = self.bounds.ptp(axis = 0)[self.slice_axis]/self.n_of_subvols

            self.slice_normal = np.zeros(3)
            self.slice_normal[self.slice_axis] = 1  # define the normal vector of the planes slicing the domain

            plus_normal  =  self.slice_normal
            minus_normal = -self.slice_normal

            normals = np.vstack( (plus_normal, minus_normal) )

            self.subvol_meshes = []

            for i in range(self.n_of_subvols):

                plus_origin                   = np.zeros(3)
                plus_origin[self.slice_axis]  = i *self.bounds.ptp(axis = 0)[self.slice_axis]/self.n_of_subvols
                
                minus_origin                  = np.zeros(3)
                minus_origin[self.slice_axis] = (i+1)*self.bounds.ptp(axis = 0)[self.slice_axis]/self.n_of_subvols

                origins = np.vstack( (plus_origin, minus_origin) )
                
                if i == 0:
                    subvol_mesh =  self.mesh.slice_plane( origins[1, :], normals[1, :], cap = True)    # slicing only the positive face
                elif i == self.n_of_subvols-1:
                    subvol_mesh =  self.mesh.slice_plane( origins[0, :], normals[0, :], cap = True)    # slicing only the negative face
                else:
                    subvol_mesh =  self.mesh.slice_plane( origins, normals, cap = True)    # slicing both

                subvol_mesh.process(validate = True)

                self.subvol_meshes.append(subvol_mesh)

            check_wt = np.array([mesh.is_watertight for mesh in self.subvol_meshes])
            
            if ~np.all(check_wt):
                print('Subvolume is not watertight!!!!')
                print('Stopping simulation...')
                quit()
                
        elif self.subvol_type in ['sphere', 'box']:
            
            self.subvol_meshes = subvolumes.distribute(self.mesh, self.n_of_subvols, self.subvol_type, view = False)
                
        else:
            print('Invalid subvolume!')
            print('Stopping simulation.')
            quit()
        
        self.subvol_volume = np.array([subvol.volume      for subvol in self.subvol_meshes])
        self.subvol_center = np.array([subvol.center_mass for subvol in self.subvol_meshes])
        self.subvol_bounds = np.array([subvol.bounds      for subvol in self.subvol_meshes])

    def get_bound_facets(self, args):

        if len(args.bound_facets)+1 == len(args.bound_cond): # if there is one more boundary condition imposed than the number of specified facets,
                                                             # it means that there is one boundary condition imposed for all other faces

            bound_cond = [args.bound_cond[-1] for _ in range(self.number_of_facets)]    # set the last boundary condition to all facets
            
            # correct those that were specifically set
            count = 0
            for bound_facet in args.bound_facets:
                bound_cond[bound_facet] = args.bound_cond[count]
                count += 1
            
            if args.bound_cond[-1] == 'P':                                                      # periodic bound cond doesn't have values
                bound_values = np.array([np.nan for _ in range(self.number_of_facets)])
            else:                                                                                      # temperature, flux and roughness have
                bound_values = np.array([args.bound_values[-1] for _ in range(self.number_of_facets)]) # set the last value to all facets
            
            bound_values[args.bound_facets] = np.array(args.bound_values)[np.arange(len(args.bound_facets), dtype = int)]                      # correct those that were specifically set
            
            # update the arguments
            args.bound_values = bound_values
            args.bound_cond   = bound_cond
            args.bound_facets = np.arange(self.number_of_facets)  # considering all facets

        res_facets_indexes = [args.bound_facets[i] for i in range(len(args.bound_facets)) if args.bound_cond[i] in ['T', 'F']]
        res_facets_indexes = np.array(res_facets_indexes).astype(int)

        facets   = [self.mesh.facets[i]      for i in range(self.number_of_facets)] # indexes of the faces that define each facet
        faces    = [self.mesh.faces[i]       for i in facets                      ] # indexes of the vertices that ar contained by each facet
        vertices = [self.mesh.vertices[i, :] for i in faces                       ] # vertices coordinates contained by the boundary facet

        # facets of reservoirs
        res_facets = [facets[i] for i in res_facets_indexes]

        # meshes of the boundary facets to be used for sampling
        self.res_meshes = self.mesh.submesh(faces_sequence = res_facets, append = False)

        self.facet_vertices = [np.unique(np.vstack((v[0, :, :], v[1, :, :])), axis = 0) for v in vertices]

        # calculation of the centroid of each facet as the mean of vertex coordinates, weighted by how many faces they are connected to.
        # this is equal to the mean of triangles centroids.
        self.facet_centroid = np.array([vertices.mean(axis = 0) for vertices in self.facet_vertices])

        # Surface area of the reservoirs' facets' meshes
        self.res_areas = np.array([mesh.area for mesh in self.res_meshes])

        # debug plot
        fig = plt.figure(figsize = (10, 10), dpi = 100)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.mesh.vertices[:, 0],
                   self.mesh.vertices[:, 1],
                   self.mesh.vertices[:, 2], c = 'b')
        
        # facets
        ax.scatter(self.facet_centroid[:, 0],
                   self.facet_centroid[:, 1],
                   self.facet_centroid[:, 2], c = 'r')
        
        for i in range(self.facet_centroid.shape[0]):
            ax.plot([self.facet_centroid[i, 0], self.facet_centroid[i, 0]+100*self.mesh.facets_normal[i, 0]],
                    [self.facet_centroid[i, 1], self.facet_centroid[i, 1]+100*self.mesh.facets_normal[i, 1]],
                    [self.facet_centroid[i, 2], self.facet_centroid[i, 2]+100*self.mesh.facets_normal[i, 2]],
                    c = 'k')
            ax.text(self.facet_centroid[i, 0],
                    self.facet_centroid[i, 1],
                    self.facet_centroid[i, 2],
                    i)
        
        # faces
        # for i in range(np.array(self.mesh.faces).shape[0]):
        #     v_index = np.array(self.mesh.faces[i])
        #     centroid = self.mesh.vertices[v_index, :].mean(axis = 0)
        #     ax.scatter(centroid[0],
        #                centroid[1],
        #                centroid[2],
        #                c = 'r')
        #     ax.plot([centroid[0], centroid[0]+100*self.mesh.face_normals[i, 0]],
        #             [centroid[1], centroid[1]+100*self.mesh.face_normals[i, 1]],
        #             [centroid[2], centroid[2]+100*self.mesh.face_normals[i, 2]],
        #             c = 'k')
            
        #     ax.text(centroid[0],
        #             centroid[1],
        #             centroid[2],
        #             i)

        plt.tight_layout()
        plt.savefig(self.folder + 'facets.png')
        plt.close(fig=fig)
        # plt.show()
        
    def check_connections(self, args):

        print('Checking connected faces...')

        connections = np.array(args.connect_facets).reshape(-1, 2)

        for i in range(connections.shape[0]):
            normal_1 = self.mesh.facets_normal[connections[i, 0], :]
            normal_2 = self.mesh.facets_normal[connections[i, 1], :]
            normal_check = np.all(np.around(normal_1, decimals = 3) == -np.around(normal_2, decimals = 3))

            vertices_1 = np.around(self.facet_vertices[connections[i, 0]] - self.facet_centroid[connections[i, 0], :], decimals = 2)
            index_1 = np.lexsort(vertices_1.T)
            vertices_1 = vertices_1[index_1, :]

            vertices_2 = np.around(self.facet_vertices[connections[i, 1]] - self.facet_centroid[connections[i, 1], :], decimals = 2)
            index_2 = np.lexsort(vertices_2.T)
            vertices_2 = vertices_2[index_2, :]

            vertex_check = np.all(vertices_1 == vertices_2)

            check = np.all([vertex_check, normal_check])

            if check:
                print('Connection {:d} OK!'.format(i))
            else:
                print('Connection {:d} is wrong! Check arguments!'.format(i))
                print('#####')
                print('Normals', normal_1, normal_2)
                print('Vertices')
                print(vertices_1)
                print(vertices_2)
                print('#####')

                print('Stopping simulation...')
                quit()

    def faces_to_facets(self, index_faces):
        ''' get which facet those faces are part of '''
        
        index_facets = np.array([np.where( face == np.array(self.mesh.facets))[0][0] for face in index_faces])

        return index_facets



        


        

    
