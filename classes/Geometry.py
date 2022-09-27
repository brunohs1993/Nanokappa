# calculations
from errno import EBADMSG
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import NearestNDInterpolator
from scipy.stats.qmc import Sobol

# geometry
import trimesh as tm
from trimesh.graph import is_watertight
import routines.subvolumes as subvolumes

from shapely.ops import polygonize_full, unary_union, triangulate

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
        self.save_reservoir_meshes()   # save meshes of each reservoir after adjust vertices
        self.set_subvolumes()          # define subvolumes and save their meshes and properties

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
        
    def facets_plot(self):
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

        self.facets_plot()

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
        
    def save_reservoir_meshes(self):

        faces    = [self.mesh.facets[i] for i in range(self.n_of_facets)] # indexes of the faces that define each facet

        # facets of reservoirs
        self.res_faces = [faces[i] for i in self.res_facets]

        # meshes of the boundary facets to be used for sampling
        self.res_meshes = self.mesh.submesh(faces_sequence = self.res_faces, append = False)

        # Surface area of the reservoirs' facets' meshes. Saving before adjusting so that it does not change the probability of entering with the offset.
        self.res_areas = np.array([mesh.area for mesh in self.res_meshes])

        h = 2*self.offset # margin to adjust - a little more than offset just to be sure

        for r in range(self.n_of_reservoirs):
            mesh = self.res_meshes[r]

            origin = mesh.vertices[0, :]        # plane origin
            normal = mesh.facets_normal[0, :]   # plane normal

            b1 = np.random.rand(3)             # generate random vector
            b1 = b1 - normal*np.sum(normal*b1) # make b1 orthogonal to the normal

            b1 = b1/np.sum(b1**2)**0.5            # normalise b1

            b2 = np.cross(normal, b1)          # generate b2 = n x b1

            plane_coord = np.zeros((0, 2))
            for v in mesh.vertices:         # for each vertex

                B = np.expand_dims((v - origin), 1)     # get relative coord
                A = np.vstack((b1, b2, normal)).T # get x and y bases coefficients
                plane_coord = np.vstack((plane_coord, np.linalg.solve(A, B).T[0, :2])) # store the plane coordinates
            
            lines = tuple(tuple(tuple(plane_coord[p, :]) for p in l) for l in mesh.edges)    # line pairs

            result, _, _, _ = polygonize_full(lines)                        # polygons

            merged = unary_union(result)

            right_offset = merged.exterior.parallel_offset(h, 'right', 2)
            left_offset  = merged.exterior.parallel_offset(h, 'left' , 2)

            if right_offset.length < left_offset.length:
                offset = right_offset.simplify(0.1)
            else: offset = left_offset.simplify(0.1)

            tris = triangulate(offset)

            v_2d = np.vstack([np.vstack(t.exterior.xy).T for t in tris])
            uv_2d = np.unique(v_2d, axis = 0)

            faces = []
            for t in tris:
                v = np.vstack(t.exterior.xy).T[:3, :]       # vertices of the triangle
                f = [np.all(v[0, :] == uv_2d, axis = 1).nonzero()[0][0],
                     np.all(v[1, :] == uv_2d, axis = 1).nonzero()[0][0],
                     np.all(v[2, :] == uv_2d, axis = 1).nonzero()[0][0]]
                faces.append(f)
            
            uv_3d = np.expand_dims(uv_2d[:, 0], 1)*b1 + np.expand_dims(uv_2d[:, 1], 1)*b2 + origin
            
            self.res_meshes[r] = tm.base.Trimesh(vertices = uv_3d, faces = faces)

    def faces_to_facets(self, index_faces):
        ''' get which facet those faces are part of '''
        
        index_facets = np.zeros(index_faces.shape)
        
        for i in range(index_faces.shape[0]):
            face = index_faces[i]
            for j in range(len(self.mesh.facets)):
                facet = self.mesh.facets[j]
                if face in facet:
                    index_facets[i] = j

        return index_facets

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
