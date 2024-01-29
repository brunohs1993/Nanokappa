# calculations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import NearestNDInterpolator
from scipy.stats.qmc import Sobol

# geometry
from classes.Mesh import Mesh
import routines.geo3d as geo3d
import trimesh as tm
import routines.subvolumes as subvolumes
from shapely.geometry import Polygon

# other
import sys
import os
import copy
import time

np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=np.nan)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class that prepares and defines the geometry to be simulated.

#   TO DO
#   
#   - Write docstring for all methods
#   - Clean the code in general

class Geometry:
    def __init__(self, args):
        
        self.args            = args
        self.standard_shapes = ['cuboid', 'box', 'cylinder', 'rod', 'bar', 'star', 'castle', 'zigzag', 'corrugated', 'freewire']
        self.scale           = args.scale
        
        self.shape           = args.geometry[0]
        self.dimensions      = args.dimensions
        
        if len(args.geo_rotation) > 0:
            self.rotation  = np.array(args.geo_rotation[:-1]).astype(float)
            self.rot_order = args.geo_rotation[-1]
        else:
            self.rotation = None
            self.rot_order = None
        
        self.subvol_type     = args.subvolumes[0]
        
        self.folder          = args.results_folder
        
        self.path_points     = np.array(self.args.path_points[1:]).astype(float).reshape(-1, 3)

        # Processing mesh
        self.tol_decimals = 1

        self.load_geo_file(self.shape) # loading
        self.transform_mesh()          # transforming
        self.get_mesh_properties()
        # self.plot_triangulation(linestyle = '-')
        self.get_bound_facets(args)    # get boundary conditions facets
        self.check_facet_connections(args)   # check if all connections are valid and adjust vertices
        self.plot_mesh_bc()
        self.set_subvolumes()          # define subvolumes and save their meshes and properties
        
        self.get_path()

        print('Geometry processing done!')

    def load_geo_file(self, shape):
        '''Load the file informed in --geometry. If an standard geometry defined in __init__, adjust
        the path to load the native file. Else, need to inform the whole path.'''
        
        print('Loading geometry...')

        if shape in self.standard_shapes:
            self.mesh = self.generate_primitives(shape, self.dimensions)
        else:
            prev_mesh = tm.load(shape)
                
            self.mesh = Mesh(np.around(prev_mesh.vertices, decimals = 10), prev_mesh.faces)
        
    def generate_primitives(self, shape, dims):
        if shape in ['cuboid', 'box']:
            vertices = np.array([[0, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 1],
                                 [0, 1, 0],
                                 [1, 0, 0],
                                 [1, 0, 1],
                                 [1, 1, 1],
                                 [1, 1, 0]])*np.array(dims)
            
            faces = np.array([[0, 1, 2],
                              [0, 2, 3],
                              [4, 5, 6],
                              [4, 6, 7],
                              [0, 4, 5],
                              [0, 5, 1],
                              [3, 7, 6],
                              [3, 6, 2],
                              [0, 4, 7],
                              [0, 7, 3],
                              [1, 5, 6],
                              [1, 6, 2]], dtype = int)

        elif shape in ['cylinder', 'rod', 'bar']:
            L = float(dims[0])
            R = float(dims[1])
            N = int(dims[2])
            
            angles = np.arange(N)*2*np.pi/N

            ring = (np.vstack((np.cos(angles), np.sin(angles), np.zeros(N)))*R).T

            vertices = np.vstack((np.zeros((1, 3)),
                                  ring,
                                  np.zeros((1, 3)) + np.array([0, 0, L]),
                                  ring             + np.array([0, 0, L])))

            faces = []
            # lower base
            for i in range(1, N+1):
                if i == N:
                    faces.append([0, i, 1])
                else:
                    faces.append([0, i, i+1])

            # sides
            for i in range(1, N+1):
                if i == N:
                    faces.append([i, i+(N+1), i+2])
                    faces.append([i, 1      , i+2])
                else:
                    faces.append([i, i+(N+1), i+1+(N+1)])
                    faces.append([i, i+1    , i+1+(N+1)])

            faces = np.vstack(faces).astype(int)         # group
            faces = np.vstack((faces, faces[:N, :]+N+1)) # upper base

        elif shape in ['zigzag']:
            L  = float(dims[0]) # section length
            R  = float(dims[1]) # radius
            dx = float(dims[2]) # dislocation x
            dy = float(dims[3]) # dislocation y
            Ns =   int(dims[4]) # number of sides
            Nc =   int(dims[5]) # number of sections

            angles = np.arange(Ns)*2*np.pi/Ns

            ring = (np.vstack((np.cos(angles), np.sin(angles), np.zeros(Ns)))*R).T

            base_faces = np.array([[i, i+1, 0] if i < Ns else [i, 1, 0] for i in range(1, Ns+1)])
            side_faces = np.zeros((2*Ns, 3), dtype = int)
            for i in range(Ns-1):
                side_faces[2*i  , :] = np.array([i, i+1 , i+Ns+1])
                side_faces[2*i+1, :] = np.array([i, i+Ns, i+Ns+1])
            side_faces[-2, :] = np.array([Ns-1,      0, Ns])
            side_faces[-1, :] = np.array([Ns-1, 2*Ns-1, Ns])

            vertices = np.vstack((np.zeros(3), ring))
            faces = np.copy(base_faces)

            for i in range(1, Nc+1):
                vertices = np.vstack((vertices,
                                      ring + np.array([int(i % 2 == 1)*dx, int(i % 2 == 1)*dy, i*L])))
                
                faces = np.vstack((faces,
                                   side_faces + (i-1)*Ns+1))
            
            # closing
            vertices = np.vstack((vertices,
                                     np.array([(Nc % 2 == 1)*dx, (Nc % 2 == 1)*dy, Nc*L])))

            faces = np.vstack((faces,
                               np.where(base_faces == 0, vertices.shape[0]-1, base_faces + vertices.shape[0]-Ns-2)))
            
        elif shape in ['corrugated']:
            L  = float(dims[0]) # section length
            R  = float(dims[1]) # outer radius
            r  = float(dims[2]) # inner radius
            Ns =   int(dims[3]) # number of sides
            Nc =   int(dims[4]) # number of sections

            angles = np.arange(Ns)*2*np.pi/Ns

            outer_ring = (np.vstack((np.cos(angles), np.sin(angles), np.zeros(Ns)))*R).T
            inner_ring = (np.vstack((np.cos(angles), np.sin(angles), np.zeros(Ns)))*r).T

            base_faces = np.array([[i, i+1, 0] if i < Ns else [i, 1, 0] for i in range(1, Ns+1)])
            side_faces = np.zeros((2*Ns, 3), dtype = int)
            for i in range(Ns-1):
                side_faces[2*i  , :] = np.array([i, i+1 , i+Ns+1])
                side_faces[2*i+1, :] = np.array([i, i+Ns, i+Ns+1])
            side_faces[-2, :] = np.array([Ns-1,      0, Ns])
            side_faces[-1, :] = np.array([Ns-1, 2*Ns-1, Ns])

            vertices = np.vstack((np.zeros(3), outer_ring))
            faces = np.copy(base_faces)

            for i in range(1, Nc+1):
                if i % 2 == 1:
                    vertices = np.vstack((vertices,
                                          inner_ring + np.array([0, 0, i*L])))
                else:
                    vertices = np.vstack((vertices,
                                          outer_ring + np.array([0, 0, i*L])))
                
                faces = np.vstack((faces,
                                   side_faces + (i-1)*Ns+1))
            
            # closing
            vertices = np.vstack((vertices,
                                     np.array([0, 0, Nc*L])))

            faces = np.vstack((faces,
                               np.where(base_faces == 0, vertices.shape[0]-1, base_faces + vertices.shape[0]-Ns-2)))

        elif shape in ['castle']:
            L  = float(dims[0]) # large castle length
            l  = float(dims[1]) # small castle length
            R  = float(dims[2]) # outer radius
            r  = float(dims[3]) # inner radius
            Ns = int(dims[4])   # number of sides
            Nc = int(dims[5])   # number of "castles"
            s  = bool(dims[6])  # start with large castle or not

            if R <= r:
                raise Exception('Outer radius smaller or equal to the inner radius. Check parameters.')
            
            outer_angles = np.arange(Ns)*2*np.pi/Ns
            inner_angles = np.arange(Ns)*2*np.pi/Ns

            inner_ring = (np.vstack((np.cos(inner_angles), np.sin(inner_angles), np.zeros(Ns)))*r).T
            outer_ring = (np.vstack((np.cos(outer_angles), np.sin(outer_angles), np.zeros(Ns)))*R).T

            small_lid_faces = np.zeros((Ns, 3), dtype = int)
            for i in range(Ns-1):
                small_lid_faces[i, [1, 2]] = np.array([i+1, i+2])
            small_lid_faces[-1, [1, 2]] =  np.array([Ns, 1])

            ring_lid_faces = np.zeros((2*Ns, 3), dtype = int)
            for i in range(Ns-1):
                ring_lid_faces[2*i  , :] = np.array([i, i+Ns, i+Ns+1])
                ring_lid_faces[2*i+1, :] = np.array([i, i+1, i+Ns+1])
            ring_lid_faces[-2, :] = np.array([Ns-1, 2*Ns-1, Ns])
            ring_lid_faces[-1, :] = np.array([Ns-1,      0, Ns])

            side_faces = np.zeros((2*Ns, 3), dtype = int)
            for i in range(Ns-1):
                side_faces[2*i  , :] = np.array([i, i   +1, i+2*Ns+1])
                side_faces[2*i+1, :] = np.array([i, i+2*Ns+1, i+2*Ns])
            side_faces[-2, :] = np.array([Ns-1, 0, 2*Ns])
            side_faces[-1, :] = np.array([Ns-1, 3*Ns-1, 2*Ns])

            # first section
            if s: # if start with large section
                vertices = np.vstack((np.zeros(3),
                                      inner_ring,
                                      outer_ring,
                                      inner_ring + np.array([0, 0, L]),
                                      outer_ring + np.array([0, 0, L])))
                faces = np.vstack((small_lid_faces           ,
                                    ring_lid_faces        + 1,
                                        side_faces +   Ns + 1,
                                    ring_lid_faces + 2*Ns + 1))
                section = 'small'
                z = L
            else:# if start with small section
                vertices = np.vstack((np.zeros(3),
                                      inner_ring,
                                      inner_ring + np.array([0, 0, l])))
                faces = np.vstack((small_lid_faces,
                                   side_faces+1))
                
                faces = np.where(faces >= 2*Ns+1, faces-Ns, faces)

                section = 'large'
                z = l

            for i in range(1, Nc):
                if section == 'small':
                    z += l
                    vertices = np.vstack((vertices,
                                          inner_ring+np.array([0, 0, z])))
                    
                    faces = np.vstack((faces,
                                       side_faces + vertices.shape[0]-3*Ns ))

                    section = 'large'

                elif section == 'large':

                    vertices = np.vstack((vertices,
                                          outer_ring+np.array([0, 0, z]),
                                          inner_ring+np.array([0, 0, z+L]),
                                          outer_ring+np.array([0, 0, z+L])))
                    
                    faces = np.vstack((faces,
                                       ring_lid_faces + vertices.shape[0]-4*Ns,
                                           side_faces + vertices.shape[0]-3*Ns,
                                       ring_lid_faces + vertices.shape[0]-2*Ns))
                    
                    section = 'small'
                    z += L

            # closing
            vertices = np.vstack((vertices,
                                  np.array([0, 0, z])))
            if section == 'small':
                faces = np.vstack((faces,
                                   np.where(small_lid_faces == 0, vertices.shape[0]-1, small_lid_faces+vertices.shape[0]-2*Ns-2)))
            if section == 'large':
                faces = np.vstack((faces,
                                   np.where(small_lid_faces == 0, vertices.shape[0]-1, small_lid_faces+vertices.shape[0]-Ns-2)))

        elif shape in ['star']:
            H = float(dims[0]) # height 
            R = float(dims[1]) # outer radius
            r = float(dims[2]) # inner radius
            N = int(dims[3])   # number of points

            if R <= r:
                raise Exception('Outer radius smaller or equal to the inner radius. Check parameters.')

            outer_angles = np.arange(N)*2*np.pi/N
            inner_angles = (np.arange(N)-0.5)*2*np.pi/N

            inner_ring = (np.vstack((np.cos(inner_angles), np.sin(inner_angles), np.zeros(N)))*r).T
            outer_ring = (np.vstack((np.cos(outer_angles), np.sin(outer_angles), np.zeros(N)))*R).T

            vertices = np.vstack((np.zeros(3),
                                  inner_ring,
                                  outer_ring))
            vertices = np.vstack((vertices,
                                  vertices + np.array([0, 0, H])))
            
            lid  = np.zeros((0, 3), dtype = int) # faces of the base 
            side = np.zeros((0, 3), dtype = int) # faces of the sides 
            for i in range(N):
                if i == N-1:
                    lid = np.vstack((lid,
                                     np.array([    0, i+1, 1]),
                                     np.array([i+1+N, i+1, 1])))
                    
                    side = np.vstack((side,
                                      np.array([[i+1  , i+  N+1, i+2*N+2],
                                                [i+N+1, i+2*N+2, i+3*N+2],
                                                [    1, i+  N+1,   2*N+2],
                                                [i+N+1,   2*N+2, i+3*N+2]])))
                    
                else:
                    lid = np.vstack((lid,
                                     np.array([    0, i+1, i+2]),
                                     np.array([i+1+N, i+1, i+2])))

                    side = np.vstack((side,
                                      np.array([[i+1  , i+  N+1, i+2*N+2],
                                                [i+N+1, i+2*N+2, i+3*N+2],
                                                [i+2  , i+  N+1, i+2*N+3],
                                                [i+N+1, i+2*N+3, i+3*N+2]])))
            
            faces = np.vstack((lid,
                               side,
                               lid + 2*N+1))
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1, subplot_kw={'projection':'3d'})

            for f in faces:
                ax.plot(vertices[f[[0, 1, 2, 0]], 0],
                        vertices[f[[0, 1, 2, 0]], 1],
                        vertices[f[[0, 1, 2, 0]], 2])
            
        elif shape in ['freewire']:
            
            R = np.array([dims[i] for i in range(0, len(dims)-1, 2)], dtype = float) # radii
            L = np.array([dims[i] for i in range(1, len(dims)-1, 2)], dtype = float) # section lengths
            N = int(dims[-1]) # number of sides

            angles = np.arange(N)*2*np.pi/N

            ring = (np.vstack((np.cos(angles), np.sin(angles), np.zeros(N)))).T

            base_faces = np.array([[i, i+1, 0] if i < N else [i, 1, 0] for i in range(1, N+1)])
            side_faces = np.zeros((2*N, 3), dtype = int)
            for i in range(N-1):
                side_faces[2*i  , :] = np.array([i, i+1 , i+N+1])
                side_faces[2*i+1, :] = np.array([i, i+N, i+N+1])
            side_faces[-2, :] = np.array([N-1,      0, N])
            side_faces[-1, :] = np.array([N-1, 2*N-1, N])

            vertices = np.vstack((np.zeros(3), ring*R[0]))
            faces = np.copy(base_faces)

            for i, r in enumerate(R[1:]):
                vertices = np.vstack((vertices,
                                        ring*r + np.array([0, 0, L[:i+1].sum()])))
                
                faces = np.vstack((faces,
                                   side_faces + i*N+1))
            
            # closing
            vertices = np.vstack((vertices,
                                  np.array([0, 0, L[:(R.shape[0]-1)].sum()])))

            faces = np.vstack((faces,
                               np.where(base_faces == 0, vertices.shape[0]-1, base_faces + vertices.shape[0]-N-2)))
            
        return Mesh(vertices, faces)

    def transform_mesh(self):
        '''Builds transformation matrix and aplies it to the mesh.'''

        print('Transforming geometry...')

        self.mesh.rezero()  # brings mesh to origin such that all vertices are on the positive octant

        self.mesh.vertices *= self.scale

        if self.rotation is not None or self.rot_order is not None:

            R = rot.from_euler(self.rot_order, self.rotation, degrees = True)
            self.mesh.vertices = R.apply(self.mesh.vertices)

            self.mesh.rezero() # brings mesh to origin back again to avoid negative coordinates
        
        

        self.mesh.update_mesh_properties()
        self.mesh.rezero()

    def get_mesh_properties(self):
        self.faces          = self.mesh.faces
        self.facets         = self.mesh.facets
        self.n_of_faces     = self.mesh.n_of_faces
        self.n_of_facets    = self.mesh.n_of_facets
        self.bounds         = self.mesh.bounds
        self.facet_centroid = self.mesh.facet_centroid
        self.volume         = self.mesh.volume
        self.facets_normal  = self.mesh.facets_normal
        self.facets_area    = self.mesh.facets_area

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

            self.subvol_classifier = SubvolClassifier(n  = self.n_of_subvols,
                                                      xc = self.subvol_center)
            self.subvol_volume = self.calculate_subvol_volume(algorithm = 'mc', tol = 1e-4, return_centers = False)

            self.get_subvol_connections()

        elif self.subvol_type == 'voronoi':
            self.n_of_subvols    = int(self.args.subvolumes[1])
            self.subvol_center = subvolumes.distribute(self.mesh, self.n_of_subvols, self.folder, view = True)

            inside = self.mesh.contains(self.subvol_center)
            self.subvol_center = self.subvol_center[inside, :]

            self.subvol_center = self.subvol_center[np.lexsort((self.subvol_center[:,2],
                                                                self.subvol_center[:,1],
                                                                self.subvol_center[:,0]))] # sort it
            
            self.get_subvol_connections()

            self.subvol_classifier = SubvolClassifier(n  = self.n_of_subvols,
                                                      xc = self.subvol_center)

            try: # try slicing the mesh first
                self.subvol_volume = self.calculate_subvol_volume()
            except: # if it gives an error, try with quasi monte carlo / sobol sampling
                self.subvol_volume = self.calculate_subvol_volume(algorithm = 'mc')
        
        elif self.subvol_type == 'grid':
            self.grid = np.array(self.args.subvolumes[1:4]).astype(int)

            nx = int(self.grid[0])
            ny = int(self.grid[1])
            nz = int(self.grid[2])
            
            dx = 1/nx
            dy = 1/ny
            dz = 1/nz

            xx = np.linspace(dx/2, 1-dx/2, nx)
            yy = np.linspace(dy/2, 1-dy/2, ny)
            zz = np.linspace(dz/2, 1-dz/2, nz)

            g = np.meshgrid(xx, yy, zz)

            self.subvol_center = (np.vstack(list(map(np.ravel, g))).T)*self.bounds.ptp(axis = 0)+self.bounds[0, :] # create centers

            passed = self.mesh.closest_point(self.subvol_center)[1] > 0

            self.subvol_center = self.subvol_center[passed, :]

            self.subvol_center = self.subvol_center[np.lexsort((self.subvol_center[:,2],
                                                                self.subvol_center[:,1],
                                                                self.subvol_center[:,0]))] # sort them

            self.n_of_subvols = self.subvol_center.shape[0]

            self.get_subvol_connections()

            self.subvol_classifier = SubvolClassifier(n  = self.n_of_subvols,
                                                      xc = self.subvol_center)

            self.subvol_volume = self.calculate_subvol_volume(algorithm = 'mc', tol = 1e-4, verbose = False)

        else:
            print('Invalid subvolume type!')
            print('Stopping simulation...')
            quit()
        
    def calculate_subvol_volume(self, algorithm = 'mc', tol = 1e-4, return_centers = False, verbose = False):
        if verbose:
            print('Calculating volumes... Algorithm:', algorithm)
        # calculating subvol cover and volume

        if self.subvol_type in ['slice', 'grid'] and self.shape in ['cuboid', 'box']:
            subvol_volume = self.volume*np.ones(self.n_of_subvols)/self.n_of_subvols

        elif algorithm == 'qmc':
            ################# RANDOM SAMPLES ########################
            
            cover = np.zeros(self.n_of_subvols)
            err   = np.ones(self.n_of_subvols)

            n_t  = 0

            ns = int(2**10)
            gen = Sobol(3)

            cnt = 1
            samples        = np.zeros((0, 3))
            scaled_samples = np.zeros((0, 3))
            while err.max() > tol:
                
                new_samples = gen.random(ns)*self.bounds.ptp(axis = 0)+self.bounds[0, :]

                new_samples_in = np.nonzero(self.mesh.contains(new_samples))[0]

                new_samples = new_samples[new_samples_in, :]
                
                samples = np.vstack((samples, new_samples))
                scaled_samples = np.vstack((scaled_samples, self.scale_positions(new_samples)))

                n_t = samples.shape[0]
                
                r = self.subvol_classifier.predict(scaled_samples)

                u_r, counts = np.unique(r, return_counts = True)

                new_cover = np.zeros(self.n_of_subvols)
                new_cover[u_r] = counts.sum()/counts
                
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    err = np.absolute((new_cover - cover)/cover)
                    err[np.isnan(err)] = 1
                if verbose:
                    print('{:4d} - Samples: {:>8.2e} - Max error: {:>8.2e}'.format(cnt, n_t, err.max()))

                cnt += 1
                cover = copy.copy(new_cover)

            subvol_volume = cover*self.volume

            if return_centers:
                r = np.argmax(self.subvol_classifier.predict(scaled_samples), axis = 1)
                subvol_center = np.zeros((self.n_of_subvols, 3))
                for sv in range(self.n_of_subvols):
                    subvol_center[sv, :] = np.mean(samples[ r == sv, :], axis = 0)
            
        elif algorithm == 'mc':
            
            cover = np.zeros(self.n_of_subvols)
            err   = np.ones(self.n_of_subvols)

            nt  = 0
            ns = int(2**10)

            cnt = 1
            samples        = np.zeros((0, 3))
            scaled_samples = np.zeros((0, 3))
            while err.max() > tol:
                
                new_samples = self.mesh.sample_volume(ns)

                samples = np.vstack((samples, new_samples))
                
                r = self.subvol_classifier.predict(new_samples)

                nr = np.array([(r == i).sum(dtype = int) for i in range(self.n_of_subvols)])

                new_cover = (cover*nt + nr)/(nt+ns)

                nt += ns

                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    err = np.absolute((new_cover - cover)/cover)
                    err[np.isnan(err)] = 1
                if verbose:
                    print('{:4d} - Samples: {:>8.2e} - Max error: {:>8.2e}'.format(cnt, nt, err.max()))

                cnt += 1
                cover = copy.copy(new_cover)

            subvol_volume = cover*self.volume

            if return_centers:
                r = self.subvol_classifier.predict(samples)
                subvol_center = np.zeros((self.n_of_subvols, 3))
                for sv in range(self.n_of_subvols):
                    subvol_center[sv, :] = np.mean(samples[ r == sv, :], axis = 0)

        if return_centers:
            return subvol_volume, subvol_center
        else:
            return subvol_volume

    def get_bound_facets(self, args):

        # initialize boundary condition array with the last one
        self.bound_cond = np.array([args.bound_cond[-1] for _ in range(self.n_of_facets)])

        # correct for specified positions
        try:
            self.bound_pos = np.array(args.bound_pos[1:]).reshape(-1, 3).astype(float)
        except:
            raise Exception('Boundary positions ill defined. Check input parameters.')
        
        if   args.bound_pos[0] == 'relative':
                self.bound_pos = self.scale_positions(self.bound_pos, True)
        elif args.bound_pos[0] == 'absolute':
            pass
        else:
            raise Exception('Please specify the type of position for BC with the keyword "absolute" or "relative".')

        self.bound_facets, _, _ = self.mesh.closest_facet(self.bound_pos)

        for j, i in enumerate(self.bound_facets):
            self.bound_cond[i] = args.bound_cond[j]

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
        
        bound_indices = [-1 for _ in range(len(self.bound_cond))]
        i = 0
        for f, facet in enumerate(self.bound_facets):
            if self.bound_cond[facet] != 'P':
                bound_indices[f] = i
                i += 1

        # saving values
        for i, bound_facet in enumerate(self.bound_facets):          # for each specified facet
            
            if bound_facet in self.res_facets:                       # if it is a reservoir
                j = self.res_facets == bound_facet                   # get where it is
                self.res_values[j] = args.bound_values[bound_indices[i]]            # save the value in res array
                
            elif bound_facet in self.rough_facets:                   # if it is a rough facet
                j = self.rough_facets == bound_facet                 # get the facet location
                self.rough_facets_values[j] = args.bound_values[bound_indices[i]]   # save roughness (eta)
        
    def check_facet_connections(self, args):

        print('Checking connected faces...')

        self.connected_facets = np.zeros((0, 2))

        if len(args.connect_pos) > 0:
            points = np.array(args.connect_pos[1:], dtype = float).reshape(-1, 3)
            if args.connect_pos[0] == 'relative':
                points = self.scale_positions(points, True)
            elif args.connect_pos[0] == 'absolute':
                pass
            else:
                raise Exception("Wrong option in --connect_pos. Choose between 'relative' or 'absolute'.")
            
            self.connected_facets = self.mesh.closest_facet(points)[0].reshape(-1, 2)

        for i in range(self.connected_facets.shape[0]):
            normal_1 = self.facets_normal[self.connected_facets[i, 0], :]
            normal_2 = self.facets_normal[self.connected_facets[i, 1], :]
            normal_check = np.all(np.absolute(normal_1+normal_2) < 10**-self.tol_decimals)

            if normal_check:

                faces_1 = self.facets[self.connected_facets[i, 0]]
                faces_2 = self.facets[self.connected_facets[i, 1]]

                mesh_1 = Mesh(vertices = self.mesh.vertices, faces = self.mesh.faces[faces_1, :])
                mesh_2 = Mesh(vertices = self.mesh.vertices, faces = self.mesh.faces[faces_2, :])

                mesh_1.rezero()
                mesh_2.rezero()
                
                ring_1 = self.get_boundary_rings(mesh_1)
                ring_2 = self.get_boundary_rings(mesh_2)

                vertices_1_2d, b1, b2 = geo3d.transform_3d_to_2d(mesh_1.vertices, normal_1, np.zeros(3))
                vertices_2_2d, _, _ = geo3d.transform_3d_to_2d(mesh_2.vertices, normal_1, np.zeros(3), b1 = b1, b2 = b2)

                vertices_1_2d, ring_1 = self.remove_midpoints_from_ring(vertices_1_2d, ring_1)
                vertices_2_2d, ring_2 = self.remove_midpoints_from_ring(vertices_2_2d, ring_2)

                poly_1 = self.save_polygon(vertices_1_2d, ring_1)
                poly_2 = self.save_polygon(vertices_2_2d, ring_2)

                poly_1.simplify(1)
                poly_2.simplify(1)

                vertex_check = poly_1.equals(poly_2)

                if vertex_check:
                    print('Connection {:d} OK!'.format(i))
                else:
                    Exception('Connection {:d} is wrong! Check arguments!'.format(i))
            else:
                Exception('Connected facets normals do not agree!!')
            
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

        legend_elements = [Patch(facecolor='w', edgecolor='k', linestyle = '-', label='Roughness'),
                           Patch(facecolor='w', edgecolor='b', linestyle = '-', label='Reservoir'),
                           Patch(facecolor='w', edgecolor='r', linestyle = ':', label='Periodic')]
        
        ax.legend(handles=legend_elements, loc='lower right')

        plt.savefig(os.path.join(self.folder, 'BC_plot.png'))
        plt.close(fig)

    def remove_midpoints_from_ring(self, v, r, tol = 1e-3):
        '''From a list of vertices (any dimension), removes midpoints
           by comparing the cosine of the angle between previous and
           next vertices in the sequence to a tolerance, and adjust
           ring sequence.'''

        d = v.shape[1]
        new_v = np.zeros((0, d))
        new_r = []

        for ring in r:
            v_ord = v[ring, :]

            diff = v_ord[1:, :] - v_ord[:-1, :]
            diff /= np.linalg.norm(diff, axis = 1, keepdims = True)
            diff = np.vstack((diff[-1, :], diff))

            dot = np.sum(diff[:-1, :]*diff[1:, :], axis = 1)
            
            v_ord = v_ord[:-1, :][1-dot > tol, :]

            rr = [i+new_v.shape[0] for i in range(v_ord.shape[0])]
            rr += [new_v.shape[0]]

            new_r.append(rr)

            new_v = np.vstack((new_v, v_ord))
        
        return new_v, new_r

    def get_boundary_rings(self, mesh, fct = 0):
        '''Returns a list of lists, where each inner list is a ring.
        For each ring, the correspondent list contains the indexes of
        the vertices ordered by connection, beginning from the lowest
        indexed vertex in the ring.
        
        If fct is not informed, the first facet of the mesh is considered.'''
        
        # ordering the vertices for line string
        bound_edges = mesh.edges[mesh.facets_boundary[fct], :]
        boundary_vertices = np.unique(bound_edges)

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
            fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 200, subplot_kw={'projection':'3d'}, layout = 'constrained')
            ax.set_box_aspect( np.ptp(mesh.bounds, axis = 0) )
            ax.tick_params(labelsize = 5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        if facets is None:
            facets = np.arange(len(mesh.facets))
        
        for fct in facets:
            for e in mesh.facets_boundary[fct]:
                ax.plot(mesh.vertices[mesh.edges[e, :], 0], mesh.vertices[mesh.edges[e, :], 1], mesh.vertices[mesh.edges[e, :], 2], linestyle = linestyle, color = l_color)
            if number_facets:
                c = self.facet_centroid[int(fct), :]
                ax.scatter(c[0], c[1], c[2], marker = markerstyle, c = m_color)
                ax.text(c[0], c[1], c[2], s = fct)
        
        plt.tight_layout()

        return fig, ax

    def faces_to_facets(self, index_faces):
        ''' get which facet those faces are part of '''
        
        if isinstance(index_faces, (int, float)): # if only one index is passed
            face = int(index_faces)
            for j in range(len(self.facets)):
                facet = self.facets[j]
                if face in facet:
                    return int(j)
        elif isinstance(index_faces, (tuple, list, np.ndarray)):
            index_faces = np.array(index_faces)
            index_facets = np.zeros(index_faces.shape)
            for i in range(index_faces.shape[0]):
                face = index_faces[i]
                for j in range(len(self.facets)):
                    facet = self.facets[j]
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

    def get_subvol_connections(self):
        print('Getting subvol connections...')

        o = (self.subvol_center+np.expand_dims(self.subvol_center, 1))/2 # interface origins/midpoints (SV, SV, 3)
        n = self.subvol_center-np.expand_dims(self.subvol_center, 1)     # interface normals/directions (SV, SV, 3)
        c_d = np.linalg.norm(n, axis = -1)                               # distances (SV, SV)

        if self.subvol_type == 'slice':
            sv_con = np.vstack((np.arange(self.n_of_subvols-1), np.arange(self.n_of_subvols-1))).T
            sv_con[:, 1] += 1
            self.subvol_connections = sv_con
            self.n_of_subvol_con = self.subvol_connections.shape[0]
            self.subvol_con_vectors = self.subvol_center[self.subvol_connections[:, 1], :] - self.subvol_center[self.subvol_connections[:, 0], :]
            self.save_subvol_connections()
            return

        psbl_con = np.ones((self.n_of_subvols, self.n_of_subvols), dtype = bool) # possible connections matrix
        psbl_con[np.arange(self.n_of_subvols), np.arange(self.n_of_subvols)] = False

        sv_con = np.vstack(psbl_con.nonzero()).T # get posible connections
        sv_con = np.sort(sv_con, axis = 1) # remove repetitions
        sv_con = sv_con[np.lexsort((sv_con[:,1], sv_con[:,0]))]
        sv_con = np.unique(sv_con, axis = 0)
        
        contains = self.mesh.contains(o[sv_con[:, 0], sv_con[:, 1], :])

        sv_con = sv_con[contains, :]
        x_col = self.subvol_center[sv_con[:, 0], :]
        v_col = n[sv_con[:, 0], sv_con[:, 1], :]
        _, d, _ = self.mesh.find_boundary(x_col, v_col)

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

        self.subvol_con_vectors = self.subvol_center[self.subvol_connections[:, 1], :] - self.subvol_center[self.subvol_connections[:, 0], :]
        
        self.save_subvol_connections()
        
    def save_subvol_connections(self):
        fig, ax = self.plot_facet_boundaries(self.mesh, l_color = 'r')

        ax.scatter(self.subvol_center[:, 0], self.subvol_center[:, 1], self.subvol_center[:, 2], marker = 'o', c = 'b', s = 5)
        for i in range(self.n_of_subvols):
            ax.text(self.subvol_center[i, 0], self.subvol_center[i, 1], self.subvol_center[i, 2], '{:d}'.format(i))

        for rp in self.subvol_connections:
            p = self.subvol_center[rp, :]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], ':', c = 'k')
        
        ax.tick_params(labelsize = 'small')
        plt.savefig(os.path.join(self.folder, 'subvol_connections.png'))
        
        plt.close(fig)

    def get_path(self):
        if len(self.args.path_points) > 0:
            if self.args.path_points[0] == 'relative':
                self.path_points = self.scale_positions(self.path_points, inv = True)
            elif self.args.path_points[0] == 'absolute':
                pass
            else:
                self.path_points = None
                raise Warning('Wrong --path_points keyword. Path will be ignored.')
        else:
            self.path_points = None
        
        if self.path_points is not None:
            self.path_kappa = self.snap_path(self.path_points)

    def snap_path(self, points):

        sv_points = self.subvol_classifier.predict(points)
        
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
                with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
                    total_v /=  np.linalg.norm(total_v)

                local_start = sv_start
                local_end   = sv_end
                
                start = time.time()
                warn_flag = True
                while local_start != local_end: # while both points do not meet

                    if time.time() - start > 10 and warn_flag:
                        print('Path is taking too long to be found. Please try again with another points.')
                        warn_flag = False

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
                        local_start = best_start
                    else:
                        i = (path == local_end).nonzero()[0][-1]
                        if best_end != local_start:
                            if path[i-1] != best_end or local_end == sv_points[i_end]:
                                path = np.insert(path, i, best_end)
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
            fig.savefig(os.path.join(self.folder, 'path_kappa.png'))
            plt.close(fig)

            return path

    def plot_triangulation(self, mesh = None, fig = None, ax = None, l_color = 'k', linestyle = '-', dpi = 200):
        if mesh is None:
            mesh = self.mesh
        fig, ax = self.mesh.plot_triangulation(fig = fig, ax = ax, l_color = l_color, linestyle = linestyle, dpi = dpi)
        
        fig.savefig(os.path.join(self.folder, 'triangulation.png'))
        plt.close(fig)

class SubvolClassifier():
    def __init__(self, n, xc = None, a = None):
        
        self.n  = n                                  # number of subvolumes

        if xc is None:
            self.a  = a                                  # slicing axis
            self.xc = np.ones((self.n, 3))*0.5
            self.xc[:, self.a] = np.linspace(0, 1-1/n, n) + 1/(2*n) # center positions
        else:
            self.xc = xc

        self.f = NearestNDInterpolator(self.xc, np.arange(self.n, dtype = int))
        
    def predict(self, x):
        return self.f(x).astype(int)