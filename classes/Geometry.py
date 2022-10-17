# calculations
from errno import EBADMSG
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import NearestNDInterpolator
from scipy.stats.qmc import Sobol

# geometry
import trimesh as tm
import routines.subvolumes as subvolumes

from shapely.geometry import Polygon, Point

from triangle import triangulate as triangulate_tri
from mapbox_earcut import triangulate_float32 as triangulate_earcut

# other
import sys
import copy

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
        
        # Processing mesh

        self.tol_decimals = 1

        self.load_geo_file(self.shape) # loading
        self.transform_mesh()          # transforming
        self.get_outer_hull(args)      # getting the outer hull, making it watertight and correcting bound_facets
        self.get_mesh_properties()     # defining useful properties
        self.get_bound_facets(args)    # get boundary conditions facets
        self.check_connections(args)   # check if all connections are valid and adjust vertices
        self.get_plane_k()
        # self.get_offset_mesh()
        self.save_reservoir_meshes()   # save meshes of each reservoir after adjust vertices
        self.set_subvolumes()          # define subvolumes and save their meshes and properties
        # self.get_voronoi_diagram()

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

            self.subvol_classifier = slice_classifier(n  = self.n_of_subvols,
                                                  xc = self.scale_positions(self.subvol_center))
            
            # self.subvol_volume = self.calculate_subvol_volume()
            try: # try slicing the mesh first
                self.subvol_volume = self.calculate_subvol_volume()
            except: # if it gives an error, try with quasi monte carlo / sobol sampling
                self.subvol_volume = self.calculate_subvol_volume(algorithm = 'qmc')

        elif self.subvol_type == 'voronoi':
            self.n_of_subvols    = int(self.args.subvolumes[1])
            self.subvol_center = subvolumes.distribute(self.mesh, self.n_of_subvols, self.folder, view = True)
            self.subvol_center = self.subvol_center[np.lexsort((self.subvol_center[:,2],
                                                                self.subvol_center[:,1],
                                                                self.subvol_center[:,0]))] # sort it
            self.subvol_classifier = slice_classifier(n  = self.n_of_subvols,
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

            self.subvol_center = (np.vstack(map(np.ravel, g)).T)*self.bounds.ptp(axis = 0)+self.bounds[0, :]
            self.subvol_center = self.subvol_center[np.lexsort((self.subvol_center[:,2],
                                                                self.subvol_center[:,1],
                                                                self.subvol_center[:,0]))] # sort it

            flag = True
            
            while flag:
                self.n_of_subvols = self.subvol_center.shape[0]

                self.subvol_classifier = slice_classifier(n  = self.n_of_subvols,
                                                          xc = self.scale_positions(self.subvol_center))
                try: # try slicing the mesh first
                    self.subvol_volume = self.calculate_subvol_volume()
                except: # if it gives an error, try with quasi monte carlo / sobol sampling
                    self.subvol_volume = self.calculate_subvol_volume(algorithm = 'qmc', tol = 1e-4, verbose = False)

                non_zero = self.subvol_volume > 0

                mean = self.subvol_volume[non_zero].mean()
                std  = self.subvol_volume[non_zero].std()

                dev = (self.subvol_volume - mean)/std # normalised deviation from the mean
                
                outliers = dev < -2

                self.subvol_center = self.subvol_center[~outliers, :]

                plt.close('all')
                plt.plot(self.subvol_center[:, 0], self.subvol_center[:, 1], 'o')
                # plt.show()

                flag = np.any(outliers)

                print(self.subvol_volume)
                print(mean, std)
                print(dev)
                print(outliers.astype(int))
                
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
                print('Boundary positions ill defined. Check input parameters.')
                quit()
            
            if   args.bound_pos[0] == 'relative':
                    self.bound_pos = self.bound_pos*self.bounds.ptp(axis = 0)+self.bounds[0, :]
            elif args.bound_pos[0] == 'absolute':
                pass
            else:
                print('Please specify the type of position for BC with the keyword "absolute" or "relative".')
                quit()

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

        fig, ax = self.plot_facet_boundaries(self.mesh, number_facets = True)
        for r in self.res_meshes:
            fig, ax = self.plot_facet_boundaries(r, fig, ax, l_color = 'b', linestyle='--')
        
        plt.savefig(self.args.results_folder + 'reservoir_meshes.png')
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
            
        # getting indices
        # ext_i = self.get_external_ring_index(v, ring_list)

        # ind = [len(ring_list[ext_i])-1]
        # ind += [len(ring)-1 for i, ring in enumerate(ring_list) if i != ext_i]
        # ind = np.array(ind).cumsum()

        # new_v = v[ring_list[ext_i][:-1], :]
        # new_v = np.vstack((new_v, *(v[ring[:-1], :] for i, ring in enumerate(ring_list) if i != ext_i)))

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


    def plot_facet_boundaries(self, mesh, fig = None, ax = None, l_color = 'k', linestyle = '-', number_facets = False, m_color = 'r', markerstyle = 'o'):

        if ax is None or fig is None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 200, subplot_kw={'projection':'3d'})
            ax.set_box_aspect( np.ptp(mesh.bounds, axis = 0) )

        for fct in range(len(mesh.facets)):
            for e in mesh.facets_boundary[fct]:
                ax.plot(mesh.vertices[e, 0], mesh.vertices[e, 1], mesh.vertices[e, 2], linestyle = linestyle, color = l_color)
            if number_facets:
                ue = np.unique(mesh.facets_boundary[fct]) # unique vertices
                c  = mesh.vertices[ue, :].mean(axis = 0)  # mean of the vertices
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

    def scale_positions(self, x):
        '''Method to scale positions x to the bounding box coordinates.
            
            x = positions to be scaled;
            
            Returns:
            x_s = scaled positions.'''

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

        face_k = np.zeros(self.mesh.faces.shape[0])
        for i in range(self.n_of_facets):
            facet = self.mesh.facets[i]
            face_k[facet] = self.facet_k[i]
        
        self.face_k = copy.copy(face_k)
        
    def get_voronoi_diagram(self):
        print('Getting voronoi connections')

        # generate samples and classify to subvolumes
        samples = subvolumes.generate_points(self.mesh, int(2**np.ceil(np.log2(1e5))), Sobol(3)) 
        sv_samples = np.argmax(self.subvol_classifier.predict(self.scale_positions(samples)), axis = 1)

        # calculate the distances between each subvol center and all others
        dist = np.linalg.norm(self.subvol_center - np.expand_dims(self.subvol_center, axis = 1), axis = -1)

        ridge_points = np.zeros((0, 2))
       
        for sv in range(self.n_of_subvols):             # for each subvolume
            print('Subvolume {}...'.format(sv))
            i_samples = (sv_samples == sv).nonzero()[0] # which samples are in that subvolume
            
            sorted_i = np.argsort(dist[sv, :])[1:]      # see what are the closest other sv

            sv_mesh = self.mesh                         # get total mesh
            current_vol = sv_mesh.volume                # and total volume

            for i in sorted_i:                          # for each other subvolume
                normal =  self.subvol_center[sv, :] - self.subvol_center[i, :]      # get the normal direction
                origin = (self.subvol_center[sv, :] + self.subvol_center[i, :])/2   # get the plane origin
                
                sv_mesh = tm.intersections.slice_mesh_plane(sv_mesh, normal, origin, cap = True) # cut the mesh with plane
                bodies = sv_mesh.split()
                for b in bodies:
                    b.fill_holes()

                sv_mesh = tm.boolean.union(bodies, engine = 'scad') 
                
                if sv_mesh.volume < current_vol: # if the plane really cuts the mesh, the volume will be reduced
                    bodies = sv_mesh.split() # get separate bodies, if it is the case
                    if len(bodies) > 1: # if there is more than one body
                        n_samples = np.array([b.contains(samples[i_samples, :]).sum() for b in bodies]) # calculate the body with more representation for that subvolume
                        sv_mesh = bodies[np.argmax(n_samples)]                                          # define the new mesh as that body
                    
                    current_vol = sv_mesh.volume
                    ridge_points = np.vstack((ridge_points, np.array([sv, i])))
                else:   # if it does not cut the mesh, they are not connected
                    pass
        
        self.ridge_points = np.unique(np.sort(ridge_points, axis = 1), axis = 0).astype(int)

    #     self.save_voronoi()
        
    # def save_voronoi(self):
    #     fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 100, figsize = (10, 10), subplot_kw ={'projection':'3d'})

    #     ax.scatter(self.subvol_center[:, 0], self.subvol_center[:, 1], self.subvol_center[:, 2], marker = 'o', c = 'b')

    #     for rp in self.ridge_points:
    #         p = self.subvol_center[rp, :]
    #         ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', c = 'k')
        
    #     for f in range(len(self.mesh.facets)):
    #         for e in self.mesh.facets_boundary[f]:
    #             p = self.mesh.vertices[e, :]
    #             ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', c = 'r')
            
    #     # plt.savefig('mesh_last.png')
    #     plt.show()

    #     plt.close(fig)

    #     fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 100, figsize = (10, 10), subplot_kw ={'projection':'3d'})

    #     for f in range(len(self.mesh.facets)):
    #         for e in self.mesh.facets_boundary[f]:
    #             p = self.mesh.vertices[e, :]
    #             ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', c = 'r')
        
    #     ax.scatter(self.subvol_center[:, 0], self.subvol_center[:, 1], self.subvol_center[:, 2], marker = 'o', c = 'b')

    #     for rp in self.ridge_points:
    #         p = self.subvol_center[rp, :]
    #         ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', c = 'k')
            
    #     # plt.savefig('mesh_first.png')
    #     plt.show()

    #     quit()
        
class slice_classifier():
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