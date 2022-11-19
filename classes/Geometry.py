# calculations
from calendar import c
from errno import EBADMSG
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import NearestNDInterpolator
from scipy.stats.qmc import Sobol
from scipy.spatial import Delaunay

# geometry
import trimesh as tm
from trimesh.ray.ray_pyembree import RayMeshIntersector # FASTER
import routines.subvolumes as subvolumes
from shapely.geometry import Polygon, Point
from mapbox_earcut import triangulate_float32 as triangulate_earcut
try:
    from triangle import triangulate as triangulate_tri
    # soft dependency, not really needed. Maybe add it as an option, like trimesh?
    # I tested both earcut and triangle and I don't see much difference, but I'll keep it for now. 
except: pass
from trimesh.proximity import ProximityQuery


# other
import sys
import copy
import gc
import time

np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=np.nan)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class that prepares and defines the geometry to be simulated.

#   TO DO
#
#   - Remove correlation length on the roughness boundary condition
#   - Write docstring for all methods
#   - Clean the code in general

class Geometry:
    def __init__(self, args):
        
        self.args            = args
        self.standard_shapes = ['cuboid', 'cillinder', 'sphere']
        self.scale           = args.scale
        self.dimensions      = args.dimensions             # angstrom
        self.rotation        = np.array(args.geo_rotation[:-1]).astype(float)
        self.rot_order       = args.geo_rotation[-1]
        self.shape           = args.geometry[0]
        self.subvol_type     = args.subvolumes[0]
        
        self.folder          = args.results_folder
        self.offset          = float(args.offset[0])
        
        self.path_points     = np.array(self.args.path_points[1:]).astype(float).reshape(-1, 3)
        
        
        # Processing mesh

        self.tol_decimals = 1

        self.load_geo_file(self.shape) # loading
        self.transform_mesh()          # transforming
        self.get_outer_hull(args)      # getting the outer hull, making it watertight and correcting bound_facets
        self.get_mesh_properties()     # defining useful properties
        self.get_bound_facets(args)    # get boundary conditions facets
        self.check_connections(args)   # check if all connections are valid and adjust vertices
        self.get_plane_k()
        self.save_reservoir_meshes(engine = 'earcut')   # save meshes of each reservoir after adjust vertices
        self.plot_mesh_bc()
        self.set_subvolumes()          # define subvolumes and save their meshes and properties

        self.get_path()

        print('Geometry processing done!')

    def load_geo_file(self, shape):
        '''Load the file informed in --geometry. If an standard geometry defined in __init__, adjust
        the path to load the native file. Else, need to inform the whole path.'''
        
        print('Loading geometry...')

        if shape == 'cuboid':
            self.mesh = tm.creation.box( self.dimensions )
        elif shape == 'cylinder':
            self.mesh = tm.creation.cylinder( radius = self.dimensions[1], height = self.dimensions[0], sections = int(self.dimensions[2]))
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

        self.mesh.rezero() # brings mesh to origin back again to avoid negative coordinates

        # THIS IS TO TRY AVOID PROBLEMS WITH PERIODIC BOUNDARY CONDITION
        # DUE TO ROUNDING ERRORS
        self.mesh.vertices = np.around(self.mesh.vertices, decimals = self.tol_decimals)
        self.mesh.vertices = np.where(self.mesh.vertices == -0., 0, self.mesh.vertices)

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
        self.n_of_facets = len(self.mesh.facets)
    
    def set_subvolumes(self):
        print('Defining subvolumes centers...')

        if self.subvol_type == 'slice':
            self.n_of_subvols    = int(self.args.subvolumes[1])
            self.slice_axis      = int(self.args.subvolumes[2])
            
            self.subvol_center = np.zeros((self.n_of_subvols, 3))
            self.subvol_center += np.mean(self.bounds, axis = 0)

            array  = (np.arange(self.n_of_subvols)+0.5)/self.n_of_subvols
            array *= np.ptp(self.bounds[:, self.slice_axis])
            array += self.bounds[0, self.slice_axis]

            self.subvol_center[:, self.slice_axis] = array
            self.subvol_center = self.subvol_center[np.lexsort((self.subvol_center[:,2],
                                                                self.subvol_center[:,1],
                                                                self.subvol_center[:,0]))] # sort it
            
            self.slice_length = np.ptp(self.bounds[:, self.slice_axis])/self.n_of_subvols

            self.get_subvol_connections()

            self.subvol_classifier = SubvolClassifier(n  = self.n_of_subvols,
                                                      xc = self.scale_positions(self.subvol_center))
            
            try: # try slicing the mesh first
                self.subvol_volume = self.calculate_subvol_volume()
            except: # if it gives an error, try with quasi monte carlo / sobol sampling
                self.subvol_volume = self.calculate_subvol_volume(algorithm = 'qmc')
            
            

        elif self.subvol_type == 'voronoi':
            self.n_of_subvols    = int(self.args.subvolumes[1])
            self.subvol_center = subvolumes.distribute(self.mesh, self.n_of_subvols, self.folder, view = True)

            # inside = self.mesh.contains(self.subvol_center)
            inside = self.contains_naive(self.subvol_center)
            self.subvol_center = self.subvol_center[inside, :]

            self.subvol_center = self.subvol_center[np.lexsort((self.subvol_center[:,2],
                                                                self.subvol_center[:,1],
                                                                self.subvol_center[:,0]))] # sort it
            
            self.get_subvol_connections()

            self.subvol_classifier = SubvolClassifier(n  = self.n_of_subvols,
                                                  xc = self.scale_positions(self.subvol_center))

            try: # try slicing the mesh first
                self.subvol_volume = self.calculate_subvol_volume()
            except: # if it gives an error, try with quasi monte carlo / sobol sampling
                self.subvol_volume = self.calculate_subvol_volume(algorithm = 'qmc')
            
            
        
        elif self.subvol_type == 'grid':
            nx = int(self.args.subvolumes[1])
            ny = int(self.args.subvolumes[2])
            nz = int(self.args.subvolumes[3])
            
            dx = 1/nx
            dy = 1/ny
            dz = 1/nz

            xx = np.linspace(dx/2, 1-dx/2, nx)
            yy = np.linspace(dy/2, 1-dy/2, ny)
            zz = np.linspace(dz/2, 1-dz/2, nz)

            g = np.meshgrid(xx, yy, zz)

            self.subvol_center = (np.vstack(list(map(np.ravel, g))).T)*self.bounds.ptp(axis = 0)+self.bounds[0, :] # create centers

            # passed = tm.proximity.signed_distance(self.mesh, self.subvol_center) > 0 

            passed = self.contains_naive(self.subvol_center)

            self.subvol_center = self.subvol_center[passed, :]

            self.subvol_center = self.subvol_center[np.lexsort((self.subvol_center[:,2],
                                                                self.subvol_center[:,1],
                                                                self.subvol_center[:,0]))] # sort them

            self.n_of_subvols = self.subvol_center.shape[0]

            self.get_subvol_connections()

            self.subvol_classifier = SubvolClassifier(n  = self.n_of_subvols,
                                                      xc = self.scale_positions(self.subvol_center))

            try: # try slicing the mesh first
                self.subvol_volume = self.calculate_subvol_volume(verbose = False)
            except: # if it gives an error, try with quasi monte carlo / sobol sampling
                self.subvol_volume = self.calculate_subvol_volume(algorithm = 'qmc', tol = 1e-4, verbose = False)

        else:
            print('Invalid subvolume type!')
            print('Stopping simulation...')
            quit()
        
        # saving subvol data
        np.savetxt(fname = self.folder + 'subvolumes.txt',
               X = np.hstack((self.subvol_center, self.subvol_volume.reshape(-1, 1))),
               fmt = '%.3f', delimiter = ',',
               header = 'Distribution of subvolumes. \n Center x, Center y, Center z')

    def calculate_subvol_volume(self, algorithm = 'submesh', tol = 1e-5, verbose = False):
        if verbose:
            print('Calculating volumes... Algorithm:', algorithm)
        # calculating subvol cover and volume

        if algorithm in ['submesh', 'submesh_qmc']:
            ################ TRYING TO CALCULATE VOLUME BY SLICING MESH #####################
            origins = (self.subvol_center+np.expand_dims(self.subvol_center, axis = 1))/2
            normals = self.subvol_center-np.expand_dims(self.subvol_center, axis = 1)

            subvol_volume = np.zeros(self.n_of_subvols)

            for sv in range(self.n_of_subvols):
                sv_mesh = copy.copy(self.mesh)
                for sv_p in [i for i in range(self.n_of_subvols) if i != sv]:
                    lines = tm.intersections.mesh_plane(sv_mesh, normals[sv_p, sv, :], origins[sv, sv_p, :])
                    if lines.shape[0] > 0:
                        sv_mesh = tm.intersections.slice_mesh_plane(sv_mesh, normals[sv_p, sv, :], origins[sv, sv_p, :], cap = True)
                
                subvol_volume[sv] = sv_mesh.volume # trimesh estimation
            
            if algorithm == 'submesh_qmc':
                # submesh quasi monte carlo estimation
                ns = int(2**20)
                nt  = 0
                nin = 0

                vbox = np.prod(sv_mesh.bounds.ptp(axis = 0))
                v    = vbox
                err = 1
                
                while err > tol:
                    s = np.random.rand(ns, 3)*sv_mesh.bounds.ptp(axis = 0)+sv_mesh.bounds[0, :]
                    i = sv_mesh.contains(s)

                    nt  += ns
                    nin += i.sum()

                    v_new = (nin/nt)*(vbox)

                    err = np.absolute((v_new - v)/v)

                    v = v_new

                subvol_volume[sv] = v_new
                if verbose:
                    print('Comparing: tm: {}, mc: {}, ratio-1:{}'.format(sv_mesh.volume, v_new, (v_new/sv_mesh.volume)-1))

        elif algorithm == 'qmc':
            ################# RANDOM SAMPLES ########################
            cover = np.zeros(self.n_of_subvols)
            err   = np.ones(self.n_of_subvols)

            n_t  = 0
            n_sv = np.zeros(self.n_of_subvols)

            ns = int(2**7)
            gen = Sobol(3)

            cnt = 1
            while err.max() > tol:
                
                samples = gen.random(ns)*self.bounds.ptp(axis = 0)+self.bounds[0, :]

                samples_in = np.nonzero(self.mesh.contains(samples))[0]

                samples = samples[samples_in, :]

                n_t += samples_in.shape[0]

                samples = self.scale_positions(samples)
                r = np.argmax(self.subvol_classifier.predict(samples), axis = 1)
                
                for i in range(self.n_of_subvols):
                    n_sv[i] += (r == i).sum()
                
                new_cover = n_sv/n_t

                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    err = np.absolute((new_cover - cover)/cover)
                
                if verbose:
                    print('{:4d} - Samples: {:>8.2e} - Max error: {:>8.2e}'.format(cnt, n_t, err.max()))

                cnt += 1
                cover = copy.copy(new_cover)

            subvol_volume = cover*self.mesh.volume

        ##############################################################
        
        return subvol_volume

    def get_bound_facets(self, args):

        # initialize boundary condition array with the last one
        self.bound_cond = np.array([args.bound_cond[-1] for _ in range(self.n_of_facets)])

        if len(args.bound_facets) > 0:
            # correct for the specified facets
            self.bound_facets = args.bound_facets
            for i in range(len(args.bound_facets)):
                self.bound_cond[self.bound_facets[i]] = args.bound_cond[i]
        elif len(args.bound_pos) > 0:
            # correct for specified positions
            try:
                self.bound_pos = np.array(args.bound_pos[1:]).reshape(-1, 3).astype(float)
            except:
                Exception('Boundary positions ill defined. Check input parameters.')
            
            if   args.bound_pos[0] == 'relative':
                    self.bound_pos = self.scale_positions(self.bound_pos, True)
            elif args.bound_pos[0] == 'absolute':
                pass
            else:
                Exception('Please specify the type of position for BC with the keyword "absolute" or "relative".')

            _, _, close_tri = tm.proximity.closest_point(self.mesh, self.bound_pos)
            
            self.bound_facets = np.zeros(0)
            for i_t, t in enumerate(close_tri):
                for i_f, f in enumerate(self.mesh.facets):
                    if t in f:
                        self.bound_facets = np.append(self.bound_facets, i_f)
                        self.bound_cond[i_f] = args.bound_cond[i_t]
                        break

        self.res_facets     = np.arange(self.n_of_facets, dtype = int)[np.logical_or(self.bound_cond == 'T', self.bound_cond == 'F')]
        self.res_bound_cond = self.bound_cond[np.logical_or(self.bound_cond == 'T', self.bound_cond == 'F')]
        self.rough_facets   = np.arange(self.n_of_facets, dtype = int)[self.bound_cond == 'R']

        # getting how many facets of each
        self.n_of_reservoirs   = self.res_facets.shape[0]
        self.n_of_rough_facets = self.rough_facets.shape[0]

        # getting how many 
        self.res_values          = np.ones(self.n_of_reservoirs  )*np.nan
        self.rough_facets_values = np.ones(self.n_of_rough_facets)*np.nan
        
        # saving the generalised boundary condition, if there's any
        if args.bound_cond[-1] in ['T', 'F']:
            self.res_values[:] = args.bound_values[-1]
        elif args.bound_cond[-1] == 'R':
            self.rough_facets_values[:] = args.bound_values[-1]
        
        # saving values
        for i in range(len(self.bound_facets)):                               # for each specified facet
            
            bound_facet = self.bound_facets[i]                                # get the facet location
            
            if bound_facet in self.res_facets:                                # if it is a reservoir
                j = self.res_facets == bound_facet                            # get where it is
                self.res_values[j] = args.bound_values[i]               # save the value in res array
                
            elif bound_facet in self.rough_facets:                            # if it is a rough facet
                j = self.rough_facets == bound_facet                          # get the facet location
                
                self.rough_facets_values[j] = args.bound_values[i]   # save roughness (eta)
        
        faces    = [self.mesh.facets[i]      for i in range(self.n_of_facets)] # indexes of the faces that define each facet
        vertices = [self.mesh.faces[i]       for i in faces                       ] # indexes of the vertices that ar contained by each facet
        coords   = [self.mesh.vertices[i, :] for i in vertices                    ] # vertices coordinates contained by the boundary facet

        self.facet_vertices = [np.unique(np.vstack((v[0, :, :], v[1, :, :])), axis = 0) for v in coords]

        # calculation of the centroid of each facet as the mean of vertex coordinates, weighted by how many faces they are connected to.
        # this is equal to the mean of triangles centroids.
        self.facet_centroid = np.array([vertices.mean(axis = 0) for vertices in self.facet_vertices])
        
    def check_connections(self, args):

        print('Checking connected faces...')

        if len(args.connect_facets) == 0:
            points = np.array(args.connect_pos[1:], dtype = float).reshape(-1, 3)
            if args.connect_pos[0] == 'relative':
                points = self.scale_positions(points, True)
            elif args.connect_pos[0] == 'absolute':
                pass
            else:
                raise Exception("Wrong option in --connect_pos. Choose between 'relative' or 'absolute'.")
            
            self.get_plane_k()
            args.connect_facets = self.find_boundary_naive(points)[2]

        connections = np.array(args.connect_facets).reshape(-1, 2)

        for i in range(connections.shape[0]):
            normal_1 = self.mesh.facets_normal[connections[i, 0], :]
            normal_2 = self.mesh.facets_normal[connections[i, 1], :]
            normal_check = np.all(np.absolute(normal_1+normal_2) < 10**-self.tol_decimals)

            vertices_1 = self.facet_vertices[connections[i, 0]] - self.facet_centroid[connections[i, 0], :]
            index_1 = np.lexsort(vertices_1.T)
            vertices_1 = vertices_1[index_1, :]

            vertices_2 = self.facet_vertices[connections[i, 1]] - self.facet_centroid[connections[i, 1], :]
            index_2 = np.lexsort(vertices_2.T)
            vertices_2 = vertices_2[index_2, :]

            vertex_check = np.all(np.absolute(vertices_1-vertices_2) < 10**-self.tol_decimals)

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
        
        print('Adjusting mesh.')
        self.adjust_connections(connections)

    def adjust_connections(self, connections):
        
        n_connections = connections.shape[0]

        check_array = np.zeros((2, n_connections), dtype = bool)
        check = np.all(check_array)
        
        count = 1
        while not check:
            print('Adjusting periodic vertices, iteration = ', count)
            for i in range(n_connections):

                facet_1 = connections[i, 0] # facets
                facet_2 = connections[i, 1]

                abs_vertices_1 = self.facet_vertices[facet_1] # absolute coordinates of vertices
                abs_vertices_2 = self.facet_vertices[facet_2]

                # the indices of each vertex in the mesh data
                vertices_indices_1 = np.all(abs_vertices_1.reshape(-1, 1, 3) == self.mesh.vertices, axis = 2)
                vertices_indices_1 = np.argmax(vertices_indices_1, axis = 0)

                vertices_indices_2 = np.all(abs_vertices_2.reshape(-1, 1, 3) == self.mesh.vertices, axis = 2)
                vertices_indices_2 = np.argmax(vertices_indices_2, axis = 0)

                # centroids of each facet
                centroid_1 = self.facet_centroid[facet_1, :]
                centroid_2 = self.facet_centroid[facet_2, :]

                rel_vertices_1 = abs_vertices_1 - centroid_1
                index_1 = np.lexsort(rel_vertices_1.T)
                rel_vertices_1 = rel_vertices_1[index_1, :]
                vertices_indices_1 = vertices_indices_1[index_1]

                rel_vertices_2 = abs_vertices_2 - centroid_2
                index_2 = np.lexsort(rel_vertices_2.T)
                rel_vertices_2 = rel_vertices_2[index_2, :]
                vertices_indices_2 = vertices_indices_2[index_2]

                new_vertices = (rel_vertices_1 + rel_vertices_2)/2

                var_1 = new_vertices - rel_vertices_1
                var_2 = new_vertices - rel_vertices_2

                check_array[0, i] = np.all(var_1 < tm.tol.merge)
                check_array[1, i] = np.all(var_2 < tm.tol.merge)

                for j in range(new_vertices.shape[0]):
                    
                    step = 0.3
                    # adjust vertices
                    self.mesh.vertices[vertices_indices_1[j], :] += step*var_1[j, :]
                    self.mesh.vertices[vertices_indices_2[j], :] += step*var_2[j, :]
                
                self.mesh.vertices = np.around(self.mesh.vertices, decimals = 8)
                
                # update centroids and facet_vertices
                faces    = [self.mesh.facets[ii]      for ii in range(self.n_of_facets)] # indexes of the faces that define each facet
                vertices = [self.mesh.faces[ii]       for ii in faces                       ] # indexes of the vertices that ar contained by each facet
                coords   = [self.mesh.vertices[ii, :] for ii in vertices                    ] # vertices coordinates contained by the boundary facet

                self.facet_vertices = [np.unique(np.vstack((v[0, :, :], v[1, :, :])), axis = 0) for v in coords]
                
                # calculation of the centroid of each facet as the mean of vertex coordinates, weighted by how many faces they are connected to.
                # this is equal to the mean of triangles centroids.
                self.facet_centroid = np.array([vertices.mean(axis = 0) for vertices in self.facet_vertices])

            check = np.all(check_array)
            count += 1

            self.mesh.rezero() # brings mesh to origin back again to avoid negative coordinates
            self.mesh.fix_normals()
        
    def save_reservoir_meshes(self, engine = 'earcut'):

        faces    = [self.mesh.facets[i] for i in range(self.n_of_facets)] # indexes of the faces that define each facet

        # facets of reservoirs
        self.res_faces = [faces[i] for i in self.res_facets]

        # meshes of the boundary facets to be used for sampling
        self.res_meshes = self.mesh.submesh(faces_sequence = self.res_faces, append = False)

        # Surface area of the reservoirs' facets' meshes. Saving before adjusting so that it does not change the probability of entering with the offset.
        self.res_areas = np.array([mesh.area for mesh in self.res_meshes])

        h = self.offset # margin to adjust - a little more than offset just to be sure

        for r in range(self.n_of_reservoirs):
            mesh = self.res_meshes[r]
            normal = mesh.facets_normal[0, :]   # plane normal
            origin = mesh.vertices[0, :]        # plane origin
            
            plane_coord, b1, b2 = self.transform_3d_to_2d(mesh, normal, origin)

            ring_list = self.get_boundary_rings(mesh)

            poly = self.save_polygon(plane_coord, ring_list)

            poly = self.offset_polygon(poly, h)
            
            if engine == 'triangle':

                v, seg, c, p = self.get_v_and_seg_for_triangle(poly, recenter = True, rescale = True)

                holes = self.get_holes_for_triangle(poly, recenter = c, rescale = p)

                if holes.shape[0] > 0:
                    poly_dict = dict(vertices=v, segments=seg, holes=holes)
                else:
                    poly_dict = dict(vertices=v, segments=seg)

                tri_dict = triangulate_tri(poly_dict, 'p')
                
                v_2d = tri_dict['vertices']*p+c
                
                faces = tri_dict['triangles']

            elif engine == 'earcut':
                v_2d, ind = self.get_v_and_ind_for_earcut(poly)

                faces = triangulate_earcut(v_2d, ind).reshape(-1, 3)

            v_3d = self.transform_2d_to_3d(v_2d, b1, b2, origin)

            v_3d = self.adjust_reservoirs_to_offset(v_3d, h, r)

            self.res_meshes[r] = tm.base.Trimesh(vertices = v_3d, faces = faces).process() # save mesh

    def plot_mesh_bc(self):
        
        fig, ax = self.plot_facet_boundaries(self.mesh, l_color = 'lightgrey', number_facets = False)

        fcts = np.arange(self.n_of_facets)[self.bound_cond == 'R']
        if fcts.shape[0] > 0:
            fig, ax = self.plot_facet_boundaries(self.mesh, fig, ax, facets = fcts, l_color = 'k', linestyle='-', m_color = 'k', number_facets=True)

        fcts = np.arange(self.n_of_facets)[self.bound_cond == 'T']
        if fcts.shape[0] > 0:
            fig, ax = self.plot_facet_boundaries(self.mesh, fig, ax, facets = fcts, l_color = 'b', linestyle='-', m_color = 'b', number_facets=True)
        
        fcts = np.arange(self.n_of_facets)[self.bound_cond == 'P']
        if fcts.shape[0] > 0:
            fig, ax = self.plot_facet_boundaries(self.mesh, fig, ax, facets = fcts, l_color = 'r', linestyle=':', m_color = 'r', number_facets=True)

        plt.savefig(self.args.results_folder + 'BC_plot.png')
        plt.close(fig)

    def transform_3d_to_2d(self, mesh, normal, origin):
        '''Transform the 3d vertices of a mesh to the 2d projection on a plane
        defined by normal and origin. The other two base vectors are generated randomly and
        orthogonal to the normal and between themselves.
        
        Returns the coordinates of the mesh vertices projected onto the plane and the base vectors.
        '''
        b1 = np.random.rand(3)              # generate random vector
        b1 = b1 - normal*np.sum(normal*b1)  # make b1 orthogonal to the normal
        b1 = b1/np.sum(b1**2)**0.5          # normalise b1
        b2 = np.cross(normal, b1)           # generate b2 = n x b1
        A = np.vstack((b1, b2, normal)).T   # get x and y bases coefficients
        
        plane_coord = np.zeros((0, 2))
        for v in mesh.vertices:             # for each vertex

            B = np.expand_dims((v - origin), 1)     # get relative coord
            
            plane_coord = np.vstack((plane_coord, np.linalg.solve(A, B).T[0, :2])) # store the plane coordinates
        
        return plane_coord, b1, b2

    def transform_2d_to_3d(self, v, b1, b2, o):
        return np.expand_dims(v[:, 0], 1)*b1 + np.expand_dims(v[:, 1], 1)*b2 + o 

    def get_boundary_rings(self, mesh, fct = 0):
        '''Returns a list of lists, where each inner list is a ring.
        For each ring, the correspondent list contains the indexes of
        the vertices ordered by connection, beginning from the lowest
        indexed vertex in the ring.
        
        If fct is not informed, the first facet of the mesh is considered.'''
        
        # ordering the vertices for line string
        boundary_vertices = np.unique(mesh.facets_boundary[fct])
        bound_edges = mesh.facets_boundary[fct]

        active = np.ones(bound_edges.shape[0], dtype = bool)
        v = boundary_vertices[0]
        ring_list = []
        ring = [v]
        while np.any(active):
            loc = np.logical_and(np.any(bound_edges == v, axis = 1), active).argmax() # where v is found on bound_edges
            e = bound_edges[loc, :] # get the edge
            next_v = e[e != v][0]
            active[loc] = False
            if next_v in ring: # if the next vertex is already registered, close the ring
                ring.append(next_v)
                ring_list.append(ring)
                v = boundary_vertices[np.argmax(~np.in1d(boundary_vertices, ring))]
                ring = [v]
            else:
                ring.append(next_v)
                v = next_v
        
        return ring_list

    def get_external_ring_index(self, v, ring_list):
        ring_bounds = [np.vstack((v[ring, :].min(axis = 0),
                                  v[ring, :].max(axis = 0))) for ring in ring_list]
        
        ext_i = 0 # assume exterior ring is the first
        for i, rbound in enumerate(ring_bounds):
            if np.all(rbound[0, :] <= ring_bounds[ext_i][0, :]) and np.all(rbound[1, :] >= ring_bounds[ext_i][1, :]):
                ext_i = i
        
        return ext_i

    def save_polygon(self, vertices, ring_list, return_ext_i = False):
        '''Save shapely Polygon object from vertices and ring list.
        
        Automatically identifies exterior and interior rings.'''
        ext_i = self.get_external_ring_index(vertices, ring_list)
        
        interiors = []
        for i in range(len(ring_list)):
            if i != ext_i:
                interiors.append(vertices[ring_list[i], :])

        poly = Polygon(vertices[ring_list[ext_i], :], interiors)
        
        if return_ext_i:
            return poly, ext_i
        else:
            return poly

    def offset_polygon(self, poly, h, join_style = 2, tolerance = 0.1):
    
        '''Returns the inner offset of a 2D shapely polygon by h, given join style and tolerance.'''

        # shell offset
        right_offset = poly.exterior.parallel_offset(h, 'right', 2)
        left_offset  = poly.exterior.parallel_offset(h, 'left' , 2)

        if right_offset.length < left_offset.length:
            ext_offset = right_offset.simplify(tolerance)
        else: ext_offset = left_offset.simplify(tolerance)

        # holes offsets
        int_offset = []
        for int_line in poly.interiors:
            right_offset = int_line.buffer(tolerance, join_style = join_style).exterior.parallel_offset(h, 'right', join_style)
            left_offset  = int_line.buffer(tolerance, join_style = join_style).exterior.parallel_offset(h, 'left' , join_style)

            if right_offset.length > left_offset.length: # holes get larger
                int_offset.append(right_offset.simplify(0.1))
            else: int_offset.append(left_offset.simplify(0.1))
        
        offset_poly = Polygon(ext_offset.coords, [i.coords for i in int_offset]) # offset polygon
        
        return offset_poly

    def get_holes_for_triangle(self, poly, N = 50, recenter = None, rescale = None):
        '''Returns a (M,2) numpy array with coordinates of points to be used
        in triangle.triangulate to identify holes in the polygon. All points are inside
        the convex hull of the polygon.'''

        # looking for holes
        bound = np.array(poly.bounds).reshape(2, 2)

        ratio = bound.ptp(axis = 0)[0]/bound.ptp(axis = 0)[1]

        if ratio >=1: # if dx > dy
            nx = int(np.ceil(ratio*N))+1
            XX, YY = np.meshgrid(np.arange(nx)/(nx-1), np.arange(N+1)/(N))
        else:
            ny = int(np.ceil(N/ratio))+1
            XX, YY = np.meshgrid(np.arange(N+1)/(N), np.arange(ny)/(ny-1))

        samples = np.vstack([XX.ravel(), YY.ravel()]).T*bound.ptp(axis = 0)+bound[0, :]

        far_enough = np.array([Point(samples[i, :]).distance(poly) for i in range(samples.shape[0])]) > np.min(bound.ptp(axis = 0)/(2*N))

        in_hull = np.array([Point(samples[i, :]).within(poly.convex_hull) for i in range(samples.shape[0])])

        in_holes = ~np.array([Point(samples[i, :]).within(poly) for i in range(samples.shape[0])])

        in_holes = np.logical_and(np.logical_and(in_holes, in_hull), far_enough)

        holes = samples[in_holes, :]

        if np.any(in_holes):
            if recenter is not None:
                holes -= recenter
            if rescale is not None:
                holes /= rescale

        return holes

    def get_v_and_seg_for_triangle(self, poly, recenter = True, rescale = True):
        
        v = np.array(poly.exterior.coords)[:-1, :]
        N = v.shape[0]
        seg = np.vstack([np.arange(N), np.arange(N)+1]).T % N

        for i in poly.interiors:
            new_v = np.array(i.coords)[:-1, :]
            
            N0 = v.shape[0]
            Ni = new_v.shape[0]
            
            v = np.vstack((v, new_v))
            
            new_seg = np.vstack([np.arange(Ni), np.arange(Ni)+1]).T % Ni + N0
            seg = np.vstack((seg, new_seg))

        # dislocate the vertices so that origin is inside the vertices (avoid errors, for some reason)
        if recenter:
            c = v.mean(axis = 0)
            v -= c
        if rescale:
            p = v.ptp(axis = 0)
            v /= p

        if recenter and rescale:
            return v, seg, c, p
        elif recenter:
            return v, seg, c
        elif rescale:
            return v, seg, p
        else:
            return v, seg

    def get_v_and_ind_for_earcut(self, poly):

        v = np.array(poly.exterior.coords)[:-1, :]
        ind = [v.shape[0]]

        for i in poly.interiors:
            new_v = np.array(i.coords)[:-1, :]
            
            ind.append(ind[-1]+new_v.shape[0])
            
            v = np.vstack((v, new_v))
            
        return v, ind

    def adjust_reservoirs_to_offset(self, v, h, r):
        fct = self.res_facets[r]
        normal = self.mesh.facets_normal[fct, :]

        v -= normal*h # move the reservoir mesh to the offsett plane

        _, d, close_face = tm.proximity.closest_point_naive(self.mesh, v)

        d = np.round(d, decimals = 8)

        too_close = np.logical_and(d < h, ~np.in1d(close_face, self.mesh.facets[fct]))
        
        while np.any(too_close):

            n_f = self.mesh.face_normals[close_face[too_close], :] # face normals

            dot_fr = np.sum(normal*n_f, axis = 1, keepdims = True)

            n_l = n_f - dot_fr*normal
            n_l /= np.linalg.norm(n_l, axis = 1, keepdims = True)

            dot_lf = np.sum(n_l*n_f, axis = 1, keepdims = True)
            
            o = self.mesh.vertices[self.mesh.faces[close_face[too_close], 0], :] # planes origins

            t_c = np.sum((o - v[too_close, :])*n_f, axis = 1, keepdims = True)/dot_lf

            t_r = t_c+h/dot_lf

            v[too_close, :] -= (t_r*n_l)

            _, d, close_face = tm.proximity.closest_point_naive(self.mesh, v)

            d = np.round(d, decimals = 8)

            too_close = np.logical_and(d < h, ~np.in1d(close_face, self.mesh.facets[self.res_facets[r]]))

        return v

    def plot_triangles(self, mesh, fig = None, ax = None, l_color = 'k', linestyle = '-', numbers = False, m_color = 'r', markerstyle = 'o'):
        
        if ax is None or fig is None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 200, subplot_kw={'projection':'3d'})
            ax.set_box_aspect( np.ptp(mesh.bounds, axis = 0) )
        
        for e in mesh.edges:
            ax.plot(mesh.vertices[e, 0], mesh.vertices[e, 1], mesh.vertices[e, 2], linestyle = linestyle, color = l_color)
        if numbers:
            for i, f in enumerate(mesh.faces):
                c = mesh.vertices[f, :].mean(axis = 0)  # mean of the vertices
                ax.scatter(c[0], c[1], c[2], marker = markerstyle, c = m_color)
                ax.text(c[0], c[1], c[2], s = i)
       
        return fig, ax

    def plot_facet_boundaries(self, mesh, fig = None, ax = None, facets = None, l_color = 'k', linestyle = '-', number_facets = False, m_color = 'r', markerstyle = 'o'):

        if ax is None or fig is None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 200, subplot_kw={'projection':'3d'})
            ax.set_box_aspect( np.ptp(mesh.bounds, axis = 0) )
            ax.tick_params(labelsize = 5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        if facets is None:
            facets = np.arange(len(mesh.facets))
        
        for fct in facets:
            for e in mesh.facets_boundary[fct]:
                ax.plot(mesh.vertices[e, 0], mesh.vertices[e, 1], mesh.vertices[e, 2], linestyle = linestyle, color = l_color)
            if number_facets:
                c = self.get_facet_centroid(int(fct))
                ax.scatter(c[0], c[1], c[2], marker = markerstyle, c = m_color)
                ax.text(c[0], c[1], c[2], s = fct)
        
        plt.tight_layout()

        return fig, ax

    def faces_to_facets(self, index_faces):
        ''' get which facet those faces are part of '''
        
        if isinstance(index_faces, (int, float)): # if only one index is passed
            face = int(index_faces)
            for j in range(len(self.mesh.facets)):
                facet = self.mesh.facets[j]
                if face in facet:
                    return int(j)
        elif isinstance(index_faces, (tuple, list, np.ndarray)):
            index_faces = np.array(index_faces)
            index_facets = np.zeros(index_faces.shape)
            for i in range(index_faces.shape[0]):
                face = index_faces[i]
                for j in range(len(self.mesh.facets)):
                    facet = self.mesh.facets[j]
                    if face in facet:
                        index_facets[i] = j

            return index_facets.astype(int)

    def scale_positions(self, x, inv = False):
        '''Method to scale positions x to the bounding box coordinates.
            
            x = positions to be scaled;
            inv = True if relative -> absolute. False if absolute -> relative. Standard is False.
            
            Returns:
            x_s = scaled positions.'''
        if inv:
            x_s = x*np.ptp(self.bounds, axis = 0) + self.bounds[0, :]
        else:
            x_s = (x - self.bounds[0, :])/np.ptp(self.bounds, axis = 0)

        return x_s

    def get_plane_k(self):
        # finds and stores the constant in the plane equation:
        # n . x + k = 0

        n = -self.mesh.facets_normal # inward normals

        # get the first vertex of each face as reference to calculate plane constant
        ref_vert = np.zeros(n.shape)

        for i in range(self.n_of_facets):
            face = self.mesh.facets[i][0]
            ref_vert[i, :] = self.mesh.vertices[self.mesh.faces[face, 0], :]

        self.facet_k = -np.sum(n*ref_vert, axis = 1) # k for each facet

    def contains_naive_single(self, x):
        v = tm.creation.icosphere(subdivisions = 0).vertices # 12 rays
        
        _, t, _ = self.find_boundary_naive(np.ones(v.shape)*x, direction = v)

        is_in = ~np.any(t == np.inf) and np.all(t > 0)

        return is_in
    
    def contains_naive(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, 3)
        
        # contains = np.array(list(map(self.contains_naive_single, x)))
        contains = np.zeros(x.shape[0], dtype = bool)
        for i, p in enumerate(x):
            # start = time.time()
            contains[i] = self.contains_naive_single(p)
            # print(time.time() - start)
        
        return contains

    def find_boundary_naive(self, x, direction = None):
        '''Finds the boundary in relation to a point. If v is informed, the
           the boundary is searched along the path in v direction. If not,
           it looks for the closest point to the mesh. This may be a better
           alternative than the Trimesh function if the mesh does not have
           many faces.
           
           Arguments:
           x = origin points
           directions (optional) = directions to evaluate the mesh.
           
           Returns:
           final_xc = position of the collision
           final_tc = distance of the collision. If direction is given, d = t/||direction||. Else, d = t.
           final_fc = the facet where the particle hits.
           '''

        # if direction is None:
        #     xc, tc, fc = zip(*map(self.find_boundary_single, x))
        # else:
        #     xc, tc, fc = zip(*map(self.find_boundary_single, x, direction))
        
        # xc = np.array(xc)
        # tc = np.array(tc)
        # fc = np.array(fc).astype(int)

        # return xc, tc, fc

        stride = int(1e5)
        step   = 0

        original_x = np.copy(x)
        if direction is not None:
            original_v = np.copy(direction)
        
        final_xc = np.zeros(x.shape)
        final_tc = np.zeros(x.shape[0])
        final_fc = np.zeros(x.shape[0])

        while step*stride < original_x.shape[0]:
            i_start = step*stride
            i_end   = min((step+1)*stride, original_x.shape[0])

            x = original_x[i_start:i_end, :]
            if direction is not None:
                direction = original_v[i_start:i_end, :]

            n = -self.mesh.facets_normal # normals - (F, 3)
            k = self.facet_k             # (F,)

            if direction is None:
                t = np.sum(x.reshape(-1, 1, 3)*n, axis = 2)+k
            else:
                v = direction
                with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
                    t = -(np.sum(x.reshape(-1, 1, 3)*n, axis = 2)+k)/np.sum(v.reshape(-1, 1, 3)*n, axis = 2)
            
            t_valid = t >= 0

            t = np.where(t_valid, t, np.inf) # update t - (N, F)
            
            xc = np.ones(x.shape)*np.nan    # (N, 3)
            tc = np.ones(x.shape[0])*np.inf # (N,)
            fc = -np.ones(x.shape[0], dtype = int) # (N,)

            active = fc < 0 # generate a follow up boolean mask - (N,)
            
            out = np.all(~t_valid, axis = 1) # particles that are out would result in an infinite loop
            active[out] = False              # those that are out are not active

            while np.any(active):
                gc.collect()
                
                t_min = np.amin(t[active, :], axis = 1, keepdims = True) # get minimum t

                cand_f = np.argmax(t[active, :] == t_min, axis = 1)      # which facet

                if direction is None:
                    v = self.mesh.facets_normal[cand_f, :]
                    cand_xc = x[active, :]+t_min*v # candidate collision positions based on t_min
                else:
                    cand_xc = x[active, :]+t_min*v[active, :] # candidate collision positions based on t_min

                for f, faces in enumerate(self.mesh.facets): # for each facet
                    f_particles = cand_f == f       # particles that may be close to that facet
                    if np.any(f_particles):
                        # faces = self.mesh.facets[f] # indexes of faces part of that facet
                        n_fp = f_particles.sum()

                        in_facet = np.zeros(n_fp, dtype = bool)
                        
                        for face in faces: # for each face in the facet
                            
                            tri = self.mesh.vertices[self.mesh.faces[face], :] # vertces of the face
                            tri = np.ones((n_fp, 3, 3))*tri # reshaping for barycentric
                            
                            # trying to use 'cross' because 'cramer' was causing infinite loops due to a division by zero
                            bar = tm.triangles.points_to_barycentric(tri, cand_xc[f_particles, :], method = 'cramer')

                            # check whether the points are between the vertices of the face
                            tol = 0
                            valid = np.all(np.logical_and(bar >= -tol, bar <= 1+tol), axis = 1)
                            
                            in_facet[valid] = True # those valid are in facet

                        in_indices = np.where(f_particles)[0][in_facet] # indices of the particles analysed for that facet that are really hitting
                        
                        active_indices = np.arange(x.shape[0])[active]  # active indices in general population

                        if direction is None:
                            better_indices = t_min[in_indices, 0] < tc[active_indices[in_indices]]
                            confirm_indices = active_indices[in_indices[better_indices]]
                        else:
                            confirm_indices = active_indices[in_indices] # those to be confirmed
                        
                        if len(confirm_indices) > 0:
                            # xc[confirm_indices, :] = np.copy(cand_xc[in_indices, :])
                            tc[confirm_indices] = np.copy(t_min[in_indices, 0])
                            fc[confirm_indices] = f
                        
                t_valid[active, cand_f] = False

                active = fc < 0

                t_valid[~active, :] = False
                t = np.where(t_valid, t, np.inf) # update t - (N, F)

                out = np.all(~t_valid, axis = 1) # particles that are out would result in an infinite loop
                out = np.logical_or(out, np.all(np.absolute(t) == np.inf, axis = 1))
                active[out] = False              # those that are out are not active

            if direction is None:
                v = self.mesh.facets_normal[fc, :]
            
            with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
                xc = x+tc.reshape(-1, 1)*v

            if direction is None:
                # if the code is looking for the nearest point the edges need to be checked.

                v1 = self.mesh.vertices[self.mesh.edges[:, 0], :]       # reference vertex (E, 3)
                e  = self.mesh.vertices[self.mesh.edges[:, 1], :] - v1  # edge vector      (E, 3)

                x_e  = np.sum( x*np.expand_dims(e, 1), axis = -1).T #  x . e - (N, E)
                v1_e = np.sum(v1*e                   , axis = -1)   # v1 . e - (E, )
                e_e  = np.sum( e*e                   , axis = -1).T #  e . e - (E, )

                t_e = (x_e - v1_e)/(e_e) # (N, E)

                t_e = np.where(t_e >= 0, t_e, np.inf) # removing t < 0
                t_e = np.where(t_e <= 1, t_e, np.inf) # removing t > 1

                #(E, N, 3)            (E, N, 1)            (E, 1, 3)           (E, 1, 3)
                with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
                    # (E, N, 3)            (E, N, 1)            (E, 1, 3)           (E, 1, 3)
                    xc_cand = np.expand_dims(t_e.T, 2)*np.expand_dims(e, 1)+np.expand_dims(v1, 1) # calculate candidate positions

                dist = np.linalg.norm(x-xc_cand, axis = -1).T # get distance from collision (N, E)

                d_min = np.nanmin(dist, axis = 1, keepdims = True) # get the closest

                ec = np.argmax(dist == d_min, axis = 1) # which edge

                t_e = t_e[np.arange(ec.shape[0]), ec] # get only the t parameter for the closest points

                fc_cand = -np.ones(ec.shape[0])
                for i, ec_i in enumerate(ec):
                    verts = self.mesh.edges[ec_i]
                    for j, faces in enumerate(self.mesh.facets):
                        fct_v = np.unique(self.mesh.faces[faces, :]) # get unique vertices
                        if np.all(np.in1d(verts, fct_v)): # if both vertices of the edge are in the facet, they are part of the facet
                            fc_cand[i] = j
                            break
                
                edge_is_closer = d_min.squeeze() < tc # which particles are actually closer to an edge than to a face
                
                xc[edge_is_closer, :] = v1[ec[edge_is_closer], :]+t_e[edge_is_closer].reshape(-1, 1)*e[ec[edge_is_closer], :]
                tc[edge_is_closer] = d_min[edge_is_closer, 0] # save tc (distance from collision point)
                fc[edge_is_closer] = fc_cand[edge_is_closer]

            final_xc[i_start:i_end, :] = np.copy(xc)
            final_tc[i_start:i_end]    = np.copy(tc)
            final_fc[i_start:i_end]    = np.copy(fc)

            step += 1
        
        final_fc = final_fc.astype(int)

        return final_xc, final_tc, final_fc

    def find_boundary_single(self, x, direction = None):
        '''Finds the boundary in relation to a single point. If v is informed, the
           the boundary is searched along the path in v direction. If not,
           it looks for the closest point to the mesh. This may be a better
           alternative for small meshes with not many faces than the Trimesh.
           
           Obs: THIS WORKS ONLY FOR POINTS INSIDE THE MESH (for now).'''

        n = -self.mesh.facets_normal # normals - (F, 3)
        k = self.facet_k             # (F,)

        if direction is None:
            t = np.sum(x*n, axis = 1)+k
        else:
            v = direction
            with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
                t = -(np.sum(x*n, axis = 1)+k)/np.sum(v*n, axis = 1)
        
        t_valid = t >= 0

        if np.all(~t_valid): # particles that are out would result in an infinite loop
            return np.ones(3)*np.nan, np.nan, -1
        
        t = np.where(t_valid, t, np.inf) # update t - (F,)
        
        order = np.argsort(t) # index of order

        in_facet = False

        for fc in order: # for each facet, in order
            tc = t[fc] # get minimum t

            if direction is None:
                v = -n[fc, :]

            xc = x + tc*v # candidate collision positions based on t_min    

            for face in self.mesh.facets[fc]: # for each face in the facet
                    
                tri = self.mesh.vertices[self.mesh.faces[face], :] # vertices of the face
                
                # trying to use 'cross' because 'cramer' was causing infinite loops due to a division by zero
                bar = tm.triangles.points_to_barycentric(np.expand_dims(tri, 0), np.expand_dims(xc, 0), method = 'cramer')

                # check whether the points are between the vertices of the face
                tol = 0
                in_facet = np.all(np.logical_and(bar >= -tol, bar <= 1+tol), axis = 1)

                if in_facet:
                    if direction is None:
                        break # leave the loop with current xc, tc and fc
                    else:
                        # return current xc, tc and fc
                        return xc, tc, fc
            
            if in_facet:
                break

        if direction is None:
            # if the code is looking for the nearest point the edges need to be checked.

            v1 = self.mesh.vertices[self.mesh.edges[:, 0], :]       # reference vertex (E, 3)
            e  = self.mesh.vertices[self.mesh.edges[:, 1], :] - v1  # edge vector      (E, 3)

            x_e  = np.sum( x*e, axis = 1, keepdims = True) #  x . e - (E,1)
            v1_e = np.sum(v1*e, axis = 1, keepdims = True) # v1 . e - (E,1)
            e_e  = np.sum( e*e, axis = 1, keepdims = True) #  e . e - (E,1)

            t_e = (x_e - v1_e)/(e_e) # (E,1)

            t_e = np.where(t_e >= 0, t_e, np.inf) # removing t < 0
            t_e = np.where(t_e <= 1, t_e, np.inf) # removing t > 1

            with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
                xc_cand = t_e*e + v1 # calculate candidate positions - (E, 3)

            dist = np.linalg.norm(x-xc_cand, axis = 1) # get distance from collision (E,)

            d_min = np.nanmin(dist) # get the closest

            if d_min >= tc:
                return xc, tc, fc
            else:
                ec = np.argmax(dist == d_min) # which edge

                t_e = t_e[ec] # get only the t parameter for the closest points

                verts = self.mesh.edges[ec]
                for j, faces in enumerate(self.mesh.facets):
                    fct_v = np.unique(self.mesh.faces[faces, :]) # get unique vertices
                    if np.all(np.in1d(verts, fct_v)): # if both vertices of the edge are in the facet, they are part of the facet
                        fc = j
                        break
            
                return xc, tc, fc

    def get_subvol_connections(self):
        print('Getting subvol connections...')
        
        o = (self.subvol_center+np.expand_dims(self.subvol_center, 1))/2 # interface origins/midpoints (SV, SV, 3)
        n = self.subvol_center-np.expand_dims(self.subvol_center, 1)     # interface normals/directions (SV, SV, 3)
        c_d = np.linalg.norm(n, axis = -1)                               # distances (SV, SV)

        psbl_con = np.ones((self.n_of_subvols, self.n_of_subvols), dtype = bool) # possible connections matrix
        psbl_con[np.arange(self.n_of_subvols), np.arange(self.n_of_subvols)] = False

        sv_con = np.vstack(psbl_con.nonzero()).T # get posible connections
        sv_con = np.sort(sv_con, axis = 1) # remove repetitions
        sv_con = sv_con[np.lexsort((sv_con[:,1], sv_con[:,0]))]
        sv_con = np.unique(sv_con, axis = 0)
        
        # Obs: for some reason mesh.contains(points) fails too often, so I'm using signed_distance.
        contains = tm.proximity.signed_distance(self.mesh, o[sv_con[:, 0], sv_con[:, 1], :]) > 0 
        # contains = self.contains_naive(o[sv_con[:, 0], sv_con[:, 1], :])

        sv_con = sv_con[contains, :]
        x_col = self.subvol_center[sv_con[:, 0], :]
        v_col = n[sv_con[:, 0], sv_con[:, 1], :]
        _, d, _ = self.find_boundary_naive(x = x_col, direction = v_col)

        sv_con = sv_con[d>1, :]

        confirmed = np.zeros(sv_con.shape[0], dtype = bool)
        
        remove = np.zeros(sv_con.shape[0], dtype = bool)
        
        order = np.argsort(c_d[sv_con[:, 0], sv_con[:, 1]]) # setting the order by proximity

        for index, con in enumerate(sv_con[order, :]): # for each connection, in order of distance
                            
            if confirmed[order[index]]:
                pass
            else:
                i_sv = con[0]
                j_sv = con[1]

                # check for i
                i_con = np.any(sv_con == i_sv, axis = 1).nonzero()[0]             # index of possible connections
                
                if np.any(confirmed[i_con]):                                      # if any of these are already confirmed
                    i_con_conf = i_con[confirmed[i_con]]                          # get which of them are
                    end_sv = sv_con[i_con_conf, :][sv_con[i_con_conf, :] != i_sv] # get the confirmed connected subvols
                    for k_sv in end_sv:                                           # for each extra sv
                        d_p = np.sum((o[i_sv, j_sv, :] - o[i_sv, k_sv, :])*n[i_sv, k_sv, :]) # calculate the distance from origin to confirmed plane
                        if d_p >= 0:                                              # if it is not in a proper position
                            remove[order[index]] = True

                # check for j
                if not remove[order[index]]:
                    # check for j
                    j_con = np.any(sv_con == j_sv, axis = 1).nonzero()[0]   # index of possible connections
                    if np.any(confirmed[j_con]):                                      # if any of these are already confirmed and was not already removed
                        j_con_conf = j_con[confirmed[j_con]]                          # get which of them are
                        end_sv = sv_con[j_con_conf, :][sv_con[j_con_conf, :] != j_sv] # get the confirmed connected subvols
                        for k_sv in end_sv:                                           # for each extra sv
                            d_p = np.sum((o[i_sv, j_sv, :] - o[j_sv, k_sv, :])*n[j_sv, k_sv, :]) # calculate the distance from origin to confirmed plane
                            if d_p >= 0:                                              # if it is not in a proper position
                                remove[order[index]] = True
                
                if not remove[order[index]]:
                    confirmed[order[index]] = True
            
        sv_con = sv_con[~remove, :]

        u_sv = np.unique(sv_con) # keep only the ones that are connected

        self.subvol_center = self.subvol_center[u_sv, :]
        
        new_sv_con = np.zeros(sv_con.shape, dtype = int)
        for i, sv in enumerate(u_sv):
            new_sv_con = np.where(sv_con == sv, i, new_sv_con)

        self.subvol_connections = np.copy(new_sv_con)

        self.n_of_subvols = self.subvol_center.shape[0]

        self.n_of_subvol_con = self.subvol_connections.shape[0]
        
        self.save_connections()
        
    def save_connections(self):
        fig, ax = self.plot_facet_boundaries(self.mesh, l_color = 'r')

        ax.scatter(self.subvol_center[:, 0], self.subvol_center[:, 1], self.subvol_center[:, 2], marker = 'o', c = 'b', s = 5)
        for i in range(self.n_of_subvols):
            ax.text(self.subvol_center[i, 0], self.subvol_center[i, 1], self.subvol_center[i, 2], '{:d}'.format(i))

        for rp in self.subvol_connections:
            p = self.subvol_center[rp, :]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], ':', c = 'k')
        
        ax.tick_params(labelsize = 'small')
        plt.savefig(self.args.results_folder + 'subvol_connections.png')
        plt.close(fig)

    def get_facet_centroid(self, f):
        if type(f) == int:
            c = np.zeros(3)
            for tri in self.mesh.facets[f]:
                v = self.mesh.vertices[self.mesh.faces[tri], :]
                A = np.linalg.norm(np.cross(v[1, :]-v[0, :], v[2, :]-v[0, :])/2)
                c += v.mean(axis = 0)*A
        elif type(f) in [np.ndarray, list, tuple]:
            c = np.zeros((f.shape[0], 3))
            for i, i_f in enumerate(f):
                for tri in self.mesh.facets[i_f]:
                    v = self.mesh.vertices[self.mesh.faces[tri], :]
                    A = np.linalg.norm(np.cross(v[1, :]-v[0, :], v[2, :]-v[0, :])/2)
                    c[i, :] += v.mean(axis = 0)*A
            
        c /= self.mesh.facets_area[f]

        return c

    def get_path(self):
        if len(self.args.path_points) > 0:
            if self.args.path_points[0] == 'relative':
                self.path_points = self.scale_positions(self.path_points, inv = True)
            elif self.args.path_points[0] == 'absolute':
                pass
            else:
                raise Warning('Wrong --path_points keyword. Path will be ignored.')
                self.path_points = None
        else:
            self.path_points = None
        
        if self.path_points is not None:
            self.path_kappa = self.snap_path(self.path_points)

    def snap_path(self, points):

        sv_points = np.argmax(self.subvol_classifier.predict(self.scale_positions(points)), axis = 1)

        if np.unique(sv_points).shape[0] == 1:
            raise Warning('Invalid path points. Path conductivity will be turned off.')
        else:
            n_paths = sv_points.shape[0] - 1

            all_paths = [np.array([sv_points[0]])]

            for i_path in range(n_paths):
                sv_start = sv_points[i_path    ] # starting subvolume
                sv_end   = sv_points[i_path + 1] #   ending subvolume

                path = np.array([sv_start, sv_end])

                total_v = self.subvol_center[sv_end, :] - self.subvol_center[sv_start, :]
                total_v /=  np.linalg.norm(total_v)

                local_start = sv_start
                local_end   = sv_end

                while local_start != local_end: # while both points do not meet

                    v = self.subvol_center[local_end, :] - self.subvol_center[local_start, :]
                    v /=  np.linalg.norm(v)

                    psbl_start = np.any(self.subvol_connections == local_start, axis = 1).nonzero()[0] # possible connections
                    psbl_start = self.subvol_connections[psbl_start, :][self.subvol_connections[psbl_start, :] != local_start]

                    if psbl_start.shape[0] > 1:
                        i = (path == local_start).nonzero()[0][-1] # index in path where local_start is
                        psbl_start = np.delete(psbl_start, psbl_start == path[i - 1]) # remove possible connections that were already done

                    psbl_end = np.any(self.subvol_connections == local_end, axis = 1).nonzero()[0] # possible connections
                    psbl_end = self.subvol_connections[psbl_end, :][self.subvol_connections[psbl_end, :] != local_end]

                    if psbl_end.shape[0] > 1:
                        i = (path == local_end).nonzero()[0][-1] # index in path where local_end is
                        psbl_end = np.delete(psbl_end, psbl_end == path[i - 1]) # remove possible connections that were already done

                    local_v_start = self.subvol_center[psbl_start, :] - self.subvol_center[local_start, :]
                    local_v_start /=  np.linalg.norm(local_v_start)
                    
                    local_v_end = self.subvol_center[psbl_end, :] - self.subvol_center[local_end, :]
                    local_v_end /=  np.linalg.norm(local_v_end)

                    dot_start = (local_v_start* v).sum(axis = 1)
                    dot_end   = (local_v_end  *-v).sum(axis = 1)

                    i_start = (dot_start == dot_start.max()).nonzero()[0]
                    
                    if i_start.shape[0] > 1:
                        total_dot_start = (local_v_start[i_start, :]* total_v).sum(axis = 1)
                        i_start = i_start[total_dot_start == total_dot_start.max()][0]
                    else:
                        i_start = i_start[0]
                    best_start = psbl_start[i_start]
                
                    i_end = (dot_end == dot_end.max()).nonzero()[0]
                    # time.sleep(1)
                    if i_end.shape[0] > 1:
                        total_dot_end = (local_v_end[i_end, :]*-total_v).sum(axis = 1)
                        i_end = i_end[total_dot_end == total_dot_end.max()][0]
                    else:
                        i_end = i_end[0]
                    best_end = psbl_end[i_end]

                    if dot_start.max() >= dot_end.max():
                        i = (path == local_start).nonzero()[0][-1]
                        if best_start != local_end:
                            if path[i-1] != best_start or local_start == sv_points[i_start]:
                                path = np.insert(path, i+1, best_start)
                            # else:
                            #     path = np.delete(path, i)
                        local_start = best_start
                    else:
                        i = (path == local_end).nonzero()[0][-1]
                        if best_end != local_start:
                            if path[i-1] != best_end or local_end == sv_points[i_end]:
                                path = np.insert(path, i, best_end)
                            # else:
                            #     path = np.delete(path, i-1, best_end)
                        local_end = best_end
                
                all_paths.append(path)
            
            all_paths = [all_paths[i][1:] if i > 0 else all_paths[i] for i in range(len(all_paths))]
            path = np.concatenate(all_paths)
            
            fig, ax = self.plot_facet_boundaries(self.mesh)
            ax.plot(self.subvol_center[path, 0],
                    self.subvol_center[path, 1],
                    self.subvol_center[path, 2], '--')
            
            for i in path:
                ax.text(self.subvol_center[i, 0], self.subvol_center[i, 1], self.subvol_center[i, 2], '{:d}'.format(i))
            fig.savefig(self.args.results_folder + 'path_kappa.png')
            plt.close(fig)

            return path

class SubvolClassifier():
    def __init__(self, n, xc = None, a = None):
        
        self.n  = n                                  # number of subvolumes

        if xc is None:
            self.a  = a                                  # slicing axis
            self.xc = np.ones((self.n, 3))*0.5
            self.xc[:, self.a] = np.linspace(0, 1-1/n, n) + 1/(2*n) # center positions
        else:
            self.xc = xc

        self.f = NearestNDInterpolator(self.xc, np.eye(self.n, self.n))
        
    def predict(self, x):
        return self.f(x)