import numpy as np
from scipy.spatial.transform import Rotation as rot
import routines.geo3d as geo3d
import matplotlib.pyplot as plt
from itertools import combinations
import os

from scipy.spatial import Delaunay

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class representing the triangular mesh without any BC or material data.

#   TO DO
#   - Complete the implementation of interfaces for interfaces with more than one face (probably using adjacency)
#   - Better triangulation to avoid holes or simplices outside boundaries (gmsh seems to be a good option)

class Mesh:
    def __init__(self, vertices, faces, remove_unref = True, triangulate_volume = True):
        self.vertices = self.correct_vertices_input(vertices)
        self.faces = faces.astype(int)
        
        self.tol = 1e-10 # tolerance for proximity checks

        self.update_mesh_properties(remove_unref, triangulate_volume)

    def correct_vertices_input(self, vertices):

        if vertices.shape[1] == 2:
            z = np.zeros((vertices.shape[0], 1))
            vertices = np.hstack((z, vertices))
        elif vertices.shape[1] == 3:
            pass
        else:
            Exception('Wrong dimensionality of passed vertices. 2D and 3D points are accepted only.')
        
        return vertices.astype(float)

    def rezero(self):
        dx = np.min(self.vertices, axis = 0)

        self.vertices         -= dx
        self.face_centroid    -= dx
        self.facet_centroid   -= dx
        self.face_origins     -= dx
        self.facets_origin    -= dx
        self.center_mass      -= dx
        self.bounds           -= dx
        self.face_bounds      -= dx
        if self.n_of_simplices > 0:
            self.simplices_points -= dx
            self.simplices_bounds -= dx

        self.get_face_k()
        self.get_facets_k()

    def rotate(self, rot_order, rotation, degrees = True):
        R = rot.from_euler(rot_order, rotation, degrees = degrees)
        self.vertices = R.apply(self.vertices)

    def update_mesh_properties(self, remove_unref = True, triangulate_volume = True):
        
        self.n_of_vertices = self.vertices.shape[0]
        self.bounds  = np.vstack((self.vertices.min(axis = 0), self.vertices.max(axis = 0)))
        self.extents = self.bounds.ptp(axis = 0)

        if remove_unref:
            self.remove_unref_vertices(update = False)
        self.get_faces_properties()  # faces_centroids, face_normals, face_area, adjacency
        self.get_edges()
        self.get_face_adjacency()
        self.get_facets_properties() # facets, facet_normals, facet_centroid, facet_area
        self.get_interfaces()
        self.check_winding()
        if triangulate_volume:
            self.get_volume_properties()

    def get_edges(self):
        '''Identify self.edges and self.face_edges from self.faces.'''

        self.edges = np.zeros((0, 2), dtype = int)

        for f in self.faces:
            self.edges = np.vstack((self.edges, np.array([[f[0], f[1]],
                                                          [f[0], f[2]],
                                                          [f[1], f[2]]])))

        self.edges = np.sort(self.edges, axis = 1)
        self.edges = np.unique(self.edges, axis = 0)
        self.edges = self.edges[np.lexsort((self.edges[:, 1], self.edges[:, 0])), :]

        self.n_of_edges = self.edges.shape[0]
        
        self.face_edges = -np.ones(self.faces.shape, dtype = int) # which edges compose each face
        for i, f in enumerate(self.faces):
            e = np.array([[f[0], f[1]],
                          [f[0], f[2]],
                          [f[1], f[2]]]) # (3, 2)
            e = np.expand_dims(np.sort(e, axis = 1), 1) # (3, 1, 2)

            equal = np.all(self.edges == e, axis = 2).T # (E, 3)

            f_e = np.argmax(equal, axis = 0) # (3,)

            self.face_edges[i, :] = np.copy(f_e)
        
        self.edges_faces = [] # which faces are composed by each edge

        for i in range(self.n_of_edges):
            self.edges_faces.append(np.any(self.face_edges == i, axis = 1).nonzero()[0])

    def check_winding(self):

        # find collision positions first        
        o = np.copy(self.face_centroid)
        n = np.copy(self.face_normals)

        # Fo = origin faces, Fh = hit faces
        #     (Fh, 3) (Fh, 3)        (Fo, 1, 3)         
        t = np.sum(n*(o - np.expand_dims(o, axis = 1)), axis = 2) # (Fo, Fh)
        #                     (Fo, 1, 3)        (Fh, 3)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            t /=  np.sum(np.expand_dims(n, axis = 1)*n, axis = 2)  # (Fo, Fh)
        t = np.where(np.isnan(t), 0, t)
        t = np.where(np.isinf(t), 0, t)
        t[np.arange(t.shape[0]), np.arange(t.shape[0])] = 0 # (Fo, Fh)

        hits = t > 0   # (Fo, Fh)

        hits[:, self.interfaces] = False
        
        # (Fo, 3)     (Fh, Fo, 1)  (Fo, 3)
        x = o + np.expand_dims(t.T, 2)*n # (Fh, Fo, 3)

        new_hits = np.copy(hits)

        for f, face in enumerate(self.faces):  # for each origin face
            for i, h in enumerate(hits[f, :]): # for each other face it may hit
                if h: # if the normal hits the plane and it is not
                    bar = geo3d.cartesian_to_barycentric(x[i, f, :], self.vertices[self.faces[i, :], :]) # calculate the barycentric coordinates of the collision
                    bar = np.around(bar, decimals = 10)
                    if np.any(bar < 0) or np.any(bar > 1): # if it is out of the triangle
                        new_hits[f, i] = False

        n_hits = new_hits.sum(axis = 1) # number of true hits

        # If a hit is right on an edge, it may be registered twice.
        # The following tries to ensure that it will not be repeated.
        for f, nh in enumerate(n_hits): # for each number of hits
            if nh > 1: # if the number of hits is more than one
                n_hits[f] = np.unique(np.around(x[new_hits[f, :], f, :], decimals = 8), axis = 0).shape[0]

        for f, nh in enumerate(n_hits):
            if nh % 2 == 1:
                self.faces[f, [0, 1]] = self.faces[f, [1, 0]]
        
        self.face_normals[n_hits % 2 == 1, :] *= -1
        self.get_facets_properties()

    def get_face_adjacency(self):
        '''Get pairs of adjacent faces from self.face_edges'''
        
        self.face_adjacency = np.zeros((0, 2), dtype = int)
        
        for i in range(self.n_of_edges):
            pair = np.any(self.face_edges == i, axis = 1)
            if pair.sum() == 2: # if edge is shared by two faces, store the pair
                self.face_adjacency = np.vstack((self.face_adjacency, pair.nonzero()[0]))
            elif pair.sum() > 2: # if it is shared by more than two triangles, store their combinations
                indices = pair.nonzero()[0]
                self.face_adjacency = np.vstack((self.face_adjacency, np.array(tuple(combinations(indices, 2)))))
        
        self.face_adjacency = np.sort(self.face_adjacency, axis = 1)
        self.face_adjacency = self.face_adjacency[np.lexsort((self.face_adjacency[:, 1], self.face_adjacency[:, 0])), :]

    def get_facet_adjacency(self):
        
        all_ngb = []
        for fct in self.facets:
            indices = np.any(np.isin(self.face_adjacency, fct), axis = 1)
            fct_ngb = np.unique(self.face_adjacency[indices, :]) # get adjacencent faces to the composing faces
            fct_ngb = fct_ngb[~np.isin(fct_ngb, fct)]        # remove those that compose the facet

            fct_ngb = np.unique(self.faces_to_facets(fct_ngb)) # transform in facets and get the unique

            all_ngb.append(fct_ngb.astype(int))
        
        self.facets_neighbours = all_ngb

        all_adj = np.zeros((0, 2))

        for f, n in enumerate(self.facets_neighbours):
            new_adj = np.zeros((len(n), 2))
            new_adj[:, 0] = f
            new_adj[:, 1] = np.array(n)
            all_adj = np.vstack((all_adj, new_adj))
        
        all_adj = np.sort(all_adj, axis = 1)
        all_adj = np.unique(all_adj, axis = 0)

        self.facets_adjacency = all_adj

    def get_faces_properties(self):
        '''Get:
           - self.face_areas
           - self.face_centroid
           - self.face_normals'''

        self.n_of_faces = self.faces.shape[0]

        self.face_areas    = np.zeros(self.n_of_faces)
        
        self.face_centroid = np.zeros((self.n_of_faces, 3))

        for i, f in enumerate(self.faces):
            v = self.vertices[f, :]

            self.face_areas[i] = np.linalg.norm(np.cross(v[1, :]-v[0, :], v[2, :]-v[0, :]))/2
            self.face_centroid[i, :] = v.mean(axis = 0)

        self.area = np.sum(self.face_areas)

        self.face_basis = np.concatenate((np.expand_dims(self.vertices[self.faces[:, 1], :] - self.vertices[self.faces[:, 0], :], 0),
                                          np.expand_dims(self.vertices[self.faces[:, 2], :] - self.vertices[self.faces[:, 0], :], 0)), axis = 0)

        self.face_normals = np.cross(self.face_basis[0, :, :], self.face_basis[1, :, :])
        self.face_normals /= np.linalg.norm(self.face_normals, axis = 1, keepdims = True)
        
        self.face_basis_matrix = np.concatenate((self.face_basis, np.expand_dims(self.face_normals, axis = 0)), axis = 0) # (3, F, 3)
        self.face_basis_matrix = np.transpose(self.face_basis_matrix, (1, 2, 0)) # (3, F, 3)

        self.face_origins = self.vertices[self.faces[:, 0], :]

        self.get_face_k()

        self.face_bounds = np.concatenate((np.expand_dims(self.vertices[self.faces, :].min(axis = 1), 0),
                                           np.expand_dims(self.vertices[self.faces, :].max(axis = 1), 0)), axis = 0)
        
    def get_facets_properties(self, tol = None):
        '''Get:
           - self.facets
           - self.facets_normal
           - self.facets_area
           - self.facet_centroid
           - self.facets_edges
           - self.facets_boundaries'''

        if tol is None:
            tol = self.tol

        n = self.face_normals                  # normals
        o = self.vertices[self.faces[:, 0], :] # origins
        k = -np.sum(n*o, axis = 1)             # plane constant

        coplanar = np.zeros(self.face_adjacency.shape[0], dtype = bool)
        for i, a in enumerate(self.face_adjacency):
            # s_n = np.linalg.norm(n[a[0], :]-n[a[1], :]) < 1e-10 # same normal
            # s_k = np.absolute(k[a[0]] - k[a[1]]) < 1e-10        # same constant

            s_n = np.absolute((n[a[0], :]*n[a[1], :]).sum()) > (1 - tol)
            s_k = np.absolute(k[a[0]]) - np.absolute(k[a[1]]) < tol        # same constant

            if s_n and s_k: # if coplanar
                coplanar[i] = True
        
        adj_cop = self.face_adjacency[coplanar, :]

        self.facets = []
        for f in range(self.n_of_faces):
            saved = np.any([np.isin(f, i) for i in self.facets])
            if not saved:
                if f in adj_cop and not saved:
                    in_facet = np.unique(adj_cop[np.any(adj_cop == f, axis = 1), :]) # getting all first adjacents
                    found_all = False
                    while not found_all:
                        n_faces = in_facet.shape[0]
                        in_facet = np.unique(adj_cop[np.any(np.isin(adj_cop, in_facet), axis = 1), :])
                        if n_faces == in_facet.shape[0]:
                            found_all = True
                    self.facets.append(in_facet.astype(int))
                else:
                    self.facets.append(np.array([f], dtype = int)) # if it doesn't have any coplanar adjacent faces, it is a facet of one face only

        self.facets = [np.sort(fct) for fct in self.facets] # sorting to avoid duplicates

        self.n_of_facets = len(self.facets)
        
        self.facets_normal  = np.array([self.face_normals[fct[0], :] for fct in self.facets])
        self.facets_area    = np.array([self.face_areas[fct].sum() for fct in self.facets])
        self.facet_centroid = np.array([np.sum(self.face_centroid[fct, :]*self.face_areas[fct].reshape(-1, 1), axis = 0)/self.facets_area[i]
                                         for i, fct in enumerate(self.facets)])
        self.facet_vertices = [self.vertices[np.unique(self.faces[self.facets[i], :]), :] for i in range(self.n_of_facets)]
        
        edges_count = [np.unique(self.face_edges[fct, :].ravel(), return_counts = True) for fct in self.facets]
        
        self.facets_edges    = [i[0].astype(int) for i in edges_count] # edges composing the facet
        self.facets_boundary = [i[0][i[1] == 1].astype(int) for i in edges_count] # edges of the facet that are note shared in the facet

        self.facets_origin = np.array([self.vertices[self.faces[i[0], 0], :] for i in self.facets])

        self.get_facets_k()

        self.get_faces_facets()

        self.get_facet_adjacency()

    def faces_to_facets(self, index_faces):
        ''' get which facet those faces are part of '''
        return self.face_facets[index_faces]
    
    def get_faces_facets(self):

        self.face_facets = -np.ones(self.n_of_faces, dtype = int)

        for f in range(self.n_of_faces):
            for i, fct in enumerate(self.facets):
                if f in fct:
                    self.face_facets[f] = i

    def get_face_k(self):
        self.face_k = -np.sum(self.face_normals*self.face_origins, axis = 1)

    def get_facets_k(self):
        self.facets_k = -np.sum(self.facets_normal*self.facets_origin, axis = 1)

    def get_interfaces(self):
        '''Checks if there are internal facets (interfacets) dividing the domain, and keep track of them as interfaces.
           OBS.: For know, only interfacets (i.e., interfaces composed of coplanar adjacent triangles) are supported,
           hence all its boundary edges should be shared with external faces. Consequently, no internal surface can have
           concavities.'''

        # this should be done before checking winding, because the collision detection needs to take into account that these faces
        # are not defining boundaries.

        interfacet_boundaries = []
        for e, ef in enumerate(self.edges_faces):
            if len(ef) > 2:                     # edge that connects more than two faces is
                interfacet_boundaries.append(e)  # the boundary of an interface

        self.interfacets = []
        for fct, b in enumerate(self.facets_boundary):
            if np.all(np.isin(b, interfacet_boundaries)):
                self.interfacets.append(fct)

        self.interfacets = np.array(self.interfacets, dtype= int) # which facets are internal
        if len(self.interfacets) > 0:
            self.interfaces = np.unique(np.concatenate([self.facets[i] for i in self.interfacets]).astype(int)) # which faces are internal
        else:
            self.interfaces = np.array([], dtype = int)
    
    def triangulate_volume(self, max_edge_div = 100, sample_volume = 100, sample_surface = 100):
        '''Delaunay triangulation of the mesh volume, keeping the simplices for volume calculations.
           Arguments:
           - max_edge_div (bool) : maximum length between points to subdivde mesh edges;
           - sample_volume (bool, int) : number of random points to be created in the volume;
           - sample_surface (bool, int) : number of random points to be created on the surface.'''

        if self.n_of_facets == 1: # if the mesh is consituted of only one plane
            self.simplices = np.zeros((0, 4))
            self.simplices_bounds = np.zeros((0, 3))
            self.contains_matrices = np.zeros((0, 3, 3))
            self.simplices_volumes = np.zeros(0)
            self.n_of_simplices = 0
        else:
            inv_failed = True
            while inv_failed:
                self.simplices_points = np.zeros((0, 3))
                for e in self.edges:
                    n = int(np.ceil(np.linalg.norm(self.vertices[e[1], :] - self.vertices[e[0], :])/max_edge_div))
                    new = self.vertices[e[0], :] + (self.vertices[e[1], :] - self.vertices[e[0], :])*np.arange(1, n).reshape(-1, 1)/n
                    self.simplices_points = np.vstack((self.simplices_points, new))

                self.simplices_points = np.vstack((self.simplices_points, self.vertices))
                if bool(sample_volume):
                    self.simplices_points = np.vstack((self.simplices_points, self.sample_volume_naive(sample_volume)))
                if bool(sample_surface):
                    self.simplices_points = np.vstack((self.simplices_points, self.sample_surface(sample_surface)))

                options = 'Qt Qbb Qc' # Qt = merged coincident points; Qbb = rescaling to [0, max]
                tri = Delaunay(self.simplices_points, qhull_options = options)
                tri.close()

                # checking simplex volume
                v1 = self.simplices_points[tri.simplices[:, 1], :] - self.simplices_points[tri.simplices[:, 0], :]
                v2 = self.simplices_points[tri.simplices[:, 2], :] - self.simplices_points[tri.simplices[:, 0], :]
                v3 = self.simplices_points[tri.simplices[:, 3], :] - self.simplices_points[tri.simplices[:, 0], :]

                X = np.cross(v1, v2, axis = 1)

                A = np.linalg.norm(X, axis = 1)/2
                
                with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
                    n = X/np.linalg.norm(X, axis = 1, keepdims = True)

                h = np.absolute(np.sum(v3*n, axis = 1))

                V = A*h/3

                to_del = V <= 1e-6

                c = self.simplices_points[tri.simplices].mean(axis = 1)
                
                contains = self.contains_naive(c)

                to_del = np.logical_or(to_del, ~contains)

                self.simplices = np.delete(tri.simplices, to_del, axis = 0) # keep the simplices
                
                self.n_of_simplices = self.simplices.shape[0]

                self.simplices_bounds = np.concatenate((np.expand_dims(self.simplices_points[self.simplices, :].min(axis = 1), 0),
                                                        np.expand_dims(self.simplices_points[self.simplices, :].max(axis = 1), 0)), axis = 0) # and their bounds

                # Setting simplices baricentric coordinates to speedup "contains" queries.
                v1 = self.simplices_points[self.simplices[:, 1], :] - self.simplices_points[self.simplices[:, 0], :]
                v2 = self.simplices_points[self.simplices[:, 2], :] - self.simplices_points[self.simplices[:, 0], :]
                v3 = self.simplices_points[self.simplices[:, 3], :] - self.simplices_points[self.simplices[:, 0], :]


                A = np.concatenate((np.expand_dims(v1, 1),
                                    np.expand_dims(v2, 1),
                                    np.expand_dims(v3, 1)), axis = 1) # (S, 3v, 3d)
                try:
                    self.contains_matrices = np.linalg.inv(A) # (S, 3v, 3d)
                    inv_failed = False
                except:
                    pass

            # calculating simplices volumes for volumetric sampling
            X = np.cross(v1, v2, axis = 1)
            A = np.linalg.norm(X, axis = 1)/2
            n = X/np.linalg.norm(X, axis = 1, keepdims = True)
            h = np.absolute(np.sum(v3*n, axis = 1))
            self.simplices_volumes = A*h/3

    def plot_triangulation(self, fig = None, ax = None, l_color = 'k', linestyle = '-', dpi = 200):

        if ax is None or fig is None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = dpi, subplot_kw={'projection':'3d'}, layout = 'constrained')
            ax.set_box_aspect( np.ptp(self.bounds, axis = 0) )
            ax.tick_params(labelsize = 5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        
        simp = np.sort(self.simplices, axis = 1)

        lines = np.vstack((np.unique(simp[:, [0, 1]], axis = 0),
                           np.unique(simp[:, [0, 2]], axis = 0),
                           np.unique(simp[:, [0, 3]], axis = 0),
                           np.unique(simp[:, [1, 2]], axis = 0),
                           np.unique(simp[:, [1, 3]], axis = 0),
                           np.unique(simp[:, [2, 3]], axis = 0)))
        
        lines = np.unique(lines, axis = 0)

        for l in lines:
            ax.plot(self.simplices_points[l, 0],
                    self.simplices_points[l, 1],
                    self.simplices_points[l, 2], color = l_color, linestyle = linestyle, linewidth = 1)
        
        return fig, ax
    
    def contains(self, x):
        '''Check if one or more points are inside the mesh by
           calculating their barycentric coordinates in relation
           to the volume simplices.'''
        
        # (P, S)                                  (P, 1, 3)         (S, 3)
        contained = np.logical_and(np.all(np.expand_dims(x, 1) >= self.simplices_bounds[0, :, :], axis = 2),
                                   np.all(np.expand_dims(x, 1) <= self.simplices_bounds[1, :, :], axis = 2))

        in_p, in_s = contained.nonzero()
        
        b = x[in_p, :] - self.simplices_points[self.simplices[in_s, 0], :] # (C, 3)

        #                    (C, 3, 3)                   (C, 3, 1)
        bar = np.sum(self.contains_matrices[in_s, :, :]*np.expand_dims(b, 2), axis = 1) # (C, (b, c, d))
        bar = np.concatenate((1 - np.sum(bar, axis = 1, keepdims = True), bar), axis = 1) # adding a

        in_simplex = np.all(np.logical_and(bar >= 0, bar <= 1), axis = 1) # (P, S)
        contained[in_p, in_s] = in_simplex

        return np.any(contained, axis = 1)

    def get_volume_properties(self):
        '''Calculate volume and center of mass.'''

        # self.triangulate_volume()
        self.triangulate_volume(max_edge_div=1000, sample_volume = 0, sample_surface=0)
        if self.simplices.shape[0] == 0:
            self.volume = 0
            self.center_mass = self.facet_centroid[0, :]
        else:
            v1 = self.simplices_points[self.simplices[:, 1], :] - self.simplices_points[self.simplices[:, 0], :]
            v2 = self.simplices_points[self.simplices[:, 2], :] - self.simplices_points[self.simplices[:, 0], :]
            v3 = self.simplices_points[self.simplices[:, 3], :] - self.simplices_points[self.simplices[:, 0], :]

            X = np.cross(v1, v2, axis = 1)

            A = np.linalg.norm(X, axis = 1, keepdims = True)/2

            n = X/np.linalg.norm(X, axis = 1, keepdims = True)

            h = np.absolute(np.sum(v3*n, axis = 1, keepdims = True))

            V = A*h/3

            self.volume = np.sum(V)

            c = (self.simplices_points[self.simplices[:, 0], :] +
                 self.simplices_points[self.simplices[:, 1], :] + 
                 self.simplices_points[self.simplices[:, 2], :] +
                 self.simplices_points[self.simplices[:, 3], :])/4

            self.center_mass = (c*V).sum(axis = 0)/V.sum()
    
    def remove_vertices(self, indices, update = True):

        all_i    = np.arange(self.n_of_vertices, dtype = int)
        keep_i   = np.delete(all_i, indices) # indices that will be kept
        remove_i = (~np.isin(all_i, keep_i)).nonzero()[0]
        remove_i = np.sort(remove_i)

        self.vertices = self.vertices[keep_i, :]

        # removing faces that used removed vertices
        faces_to_remove = np.any(np.isin(self.faces, remove_i), axis = 1)
        new_faces = self.faces[~faces_to_remove, :]

        # correcting the vertices' indices in self.faces
        self.faces = np.zeros(new_faces.shape, dtype = int)
        for i, v in enumerate(keep_i):
            self.faces[new_faces == v] = i
        
        if update:
            self.update_mesh_properties()
        
    def remove_faces(self, indices, update = True):
        all_i    = np.arange(self.n_of_faces, dtype = int)
        keep_i   = np.delete(all_i, indices) # indices that will be kept

        self.faces = self.faces[keep_i, :]

        if update:
            self.update_mesh_properties()

    def remove_duplicates(self, tol = None):
        '''Removes any duplicate vertices or faces that the mesh may have according to a tolerance.'''

        if tol is None:
            tol = self.tol
        
        # removing repeated vertices. Done with a while loop to keep self.vertices free to be changed
        flag = True
        while flag:
            found = False
            i = 0
            while not found:
                if i == self.n_of_vertices:
                    flag = False
                    break

                same_v = np.linalg.norm(self.vertices - self.vertices[i, :], axis = 1) < tol
                same_v[i] = False
                if np.any(same_v) and i < self.n_of_vertices:
                    found = True
                else:
                    i += 1

            same_v = same_v.nonzero()[0]
            for sv in same_v:
                self.faces[self.faces == sv] = i
            
            self.remove_vertices(same_v)
        
        # removing repeated faces
        _, inverse, counts = np.unique(np.sort(self.faces, axis = 1), axis = 0, return_counts = True, return_inverse = True)

        same_f = np.array([], dtype = int)
        for i, c in enumerate(counts):
            if c > 1:
                same_f = np.concatenate((same_f, (inverse == i).nonzero()[0][1:]))
            
        self.remove_faces(same_f)

    def remove_unref_vertices(self, update = True):
        ref = np.isin(np.arange(self.n_of_vertices), np.unique(self.faces))
        unref = (~ref).nonzero()[0]

        self.remove_vertices(unref, update)
    
    def plot_triangles(self, fig = None, ax = None, l_color = 'k', linestyle = '-', numbers = False, m_color = 'r', markerstyle = 'o', dpi = 200):
        
        if ax is None or fig is None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = dpi, subplot_kw={'projection':'3d'})
            ax.set_box_aspect( np.ptp(self.bounds, axis = 0) )
        
        for e in self.edges:
            ax.plot(self.vertices[e, 0], self.vertices[e, 1], self.vertices[e, 2], linestyle = linestyle, color = l_color)
        if numbers:
            for i, f in enumerate(self.faces):
                c = self.vertices[f, :].mean(axis = 0)  # mean of the vertices
                ax.scatter(c[0], c[1], c[2], marker = markerstyle, c = m_color)
                ax.text(c[0], c[1], c[2], s = i)
       
        return fig, ax
    
    def plot_facet_boundaries(self, fig = None, ax = None, facets = None, l_color = 'k', linestyle = '-', number_facets = False, m_color = 'r', markerstyle = 'o', dpi = 200):

        if ax is None or fig is None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = dpi, subplot_kw={'projection':'3d'}, layout = 'constrained')
            ax.set_box_aspect( np.ptp(self.bounds, axis = 0) )
            ax.tick_params(labelsize = 5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        if facets is None:
            facets = np.arange(self.n_of_facets)
        
        for fct in facets:
            for e in self.facets_boundary[fct]:
                ax.plot(self.vertices[self.edges[e, :], 0], self.vertices[self.edges[e, :], 1], self.vertices[self.edges[e, :], 2], linestyle = linestyle, color = l_color)
            if number_facets:
                # c = self.get_facet_centroid(int(fct))
                c = self.facet_centroid[int(fct), :]
                ax.scatter(c[0], c[1], c[2], marker = markerstyle, c = m_color)
                ax.text(c[0], c[1], c[2], s = fct)
        
        plt.tight_layout()

        return fig, ax
    
    def closest_face(self, x):
        '''finds the closest face to one or more points x onto which
           the points can be projected. If there is more than one it
           chooses the face if smallest index.'''
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        
        n_p = x.shape[0]
        
        #        (P, 1, 3)                (F, 3)                  (F, 3)               (P, 1, 3)        (F, 3)
        pj = np.expand_dims(x, 1) - self.face_normals*np.expand_dims(np.sum(self.face_normals*(np.expand_dims(x, 1)-self.face_origins), axis = -1), 2) #(P, F, 3)
        
        valid = np.logical_and(np.all(pj >= self.face_bounds[0, :, :]-self.tol, axis = 2), # (P, F)
                               np.all(pj <= self.face_bounds[1, :, :]+self.tol, axis = 2)) # with tolerance margin to avoid numerical errors 
        
        in_p, in_f = valid.nonzero()

        A = self.face_basis_matrix[in_f, :, :]
        o = self.face_origins[in_f, :]

        bar = np.linalg.solve(A, pj[in_p, in_f, :] - o)[:, :2]
        bar = np.concatenate((bar, 1 - bar.sum(axis=1, keepdims = True)), axis = 1)

        valid[in_p, in_f] = np.all(np.logical_and(bar >= 0-self.tol, bar <= 1+self.tol), axis = -1)

        in_p, in_f = valid.nonzero()
        
        d             = np.ones((n_p, self.n_of_faces))*np.inf
        d[in_p, in_f] = np.linalg.norm(pj[in_p, in_f, :] - x[in_p, :], axis = 1)
        
        d_min = d.min(axis = 1)
        f = np.argmax(d == np.expand_dims(d_min, 1), axis = 1)
        f[np.isinf(d_min)] = -1

        return f.astype(int), d[np.arange(n_p), f], pj[np.arange(n_p), f, :]

    def closest_facet(self, x):
        f, d, xc = self.closest_face(x)

        valid = f >= 0

        f[valid] = np.array(list(map(self.face_to_facet, f[valid])))

        return f, d, xc
        
    def face_to_facet(self, f):
        '''Finds which facet the face f is part of.'''
        for i, fct in enumerate(self.facets):
            if f in fct:
                return i
        
        # base case
        return -1

    def closest_edge(self, x, edges = None):
        '''Finds the closest point on the edges of the geometry for
           one or more points in x.'''
        if edges is None:
            edges = self.edges
        else:
            edges = self.edges[edges, :]

        if len(x.shape) == 1:
            x = x.reshape(1, 3)
        
        v1 = np.expand_dims(self.vertices[edges[:, 0], :], 1) # (E, 1, 3)
        v2 = np.expand_dims(self.vertices[edges[:, 1], :], 1) # (E, 1, 3)
        dv = v2 - v1                                               # (E, 1, 3)

        t = np.sum((x - v1)*dv, axis = 2)/np.sum(dv*dv, axis = 2) # (E, P)
        
        t[t<=0] = 0
        t[t>=1] = 1

        c = v1+np.expand_dims(t, 2)*dv # (E, P, 3)

        d = np.linalg.norm(c - x, axis = 2) # (E, P)

        e = np.argmax(d == d.min(axis = 0), axis = 0)

        c = c[e, np.arange(x.shape[0]), :]

        d = d[e, np.arange(x.shape[0])]

        return e, d, c

    def closest_point(self, x):
        '''Finds the closest point on the mesh from
           on or more points x.'''
        _, df, xf = self.closest_face(x)
        _, de, xe = self.closest_edge(x)
        
        xc = np.copy(xf)
        xc[de<df, :] = xe[de<df, :]

        dc = np.where(de < df, de, df)

        return xc, dc

    def contains_naive(self, x):

        v = np.mean(self.bounds, axis = 0) - x
        v /= np.linalg.norm(v, axis = 1, keepdims = True)
        # _, _, f = self.find_boundary(x, v)
        f, _, p = self.closest_face(x)

        contains = f >= 0
        contains[contains] = np.sum(self.face_normals[f[contains], :]*(p[contains, :] - x[contains, :]), axis = 1) > 0

        # contains[contains] = np.sum(self.facets_normal[f[contains], :]*v[contains, :], axis = 1) >= 0

        return contains

    def find_boundary(self, x, v):
        # c . n + k = 0
        # c = x + t v
        # Thus:
        # x . n + t (v . n) + k = 0
        # t = - [(x . n) + k]/(v . n)

        if len(x.shape) == 1:
            x = x.reshape(1, 3)

        with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
        #                    sum((P, 1, 3)*(F, 3)) --> (P, F)             + (F,)              sum((P, 1, 3) * (F, 3)) ---> (P, F)
            t = -(np.sum(np.expand_dims(x, 1)*self.face_normals, axis = 2)+self.face_k)/np.sum(np.expand_dims(v, 1)*self.face_normals, axis = 2) # (P, F)

        possible = t >= self.tol
        possible = np.logical_and(possible, ~np.isnan(t))
        possible = np.logical_and(possible, ~np.isinf(np.absolute(t)))

        in_p, in_f = possible.nonzero()

        c = x[in_p, :] + np.expand_dims(t[in_p, in_f], 1)*v[in_p, :]

        in_bounds = np.logical_and(np.all(c >= self.face_bounds[0, in_f, :]-self.tol, axis = 1),
                                   np.all(c <= self.face_bounds[1, in_f, :]+self.tol, axis = 1)) # with tolerance margin to avoid numerical errors

        possible[in_p, in_f] = in_bounds

        c = c[in_bounds, :]

        in_p, in_f = possible.nonzero()

        A = self.face_basis_matrix[in_f, :, :]
        o = self.face_origins[in_f, :]

        bar = np.linalg.solve(A, c - o)[:, :2]
        bar = np.concatenate((bar, 1 - bar.sum(axis=1, keepdims = True)), axis = 1)

        possible[in_p, in_f] = np.all(np.logical_and(bar >= 0-self.tol, bar <= 1+self.tol), axis = -1) # (P, 2)

        t = np.where(possible, t, np.inf)
        
        tc = np.min(t, axis = 1)

        fc = self.faces_to_facets(np.argmax(t == np.expand_dims(tc, 1), axis = -1))

        fc[tc == np.inf] = -1
        fc = fc.astype(int)

        xc = x + np.expand_dims(tc, 1)*v

        return xc, tc, fc

    def sample_volume(self, n):
        if self.n_of_simplices == 0:
            raise Exception('Number of simplices is zero. The mesh may be a plane and has no volume to sample from.')
        
        s = np.random.choice(self.n_of_simplices, size = n, p = self.simplices_volumes/self.simplices_volumes.sum()) # from which simplex to draw the points

        v = self.simplices_points[self.simplices[s, :], :] # (N, 4, 3)

        a = np.random.rand(n, 4, 1) # (N, 4, 1)
        a = -np.log(a)
        a /= a.sum(axis = 1, keepdims = True) # (N, 1, 1)

        x = np.sum(a*v, axis = 1)

        return x

    def sample_volume_naive(self, n):
        '''Sample volume with rejection sampling using contains_naive.
           This method is slower, but does not depend on simplices.'''

        points = np.random.rand(n, 3)*self.extents + self.bounds[0, :]

        contains = self.contains_naive(points)

        points = points[contains, :]

        while points.shape[0] < n:
            new_points = np.random.rand(n-points.shape[0], 3)*self.extents + self.bounds[0, :]
            contains   = self.contains_naive(new_points)
            points = np.vstack((points, new_points[contains, :]))

        return points

    def sample_surface(self, n, faces = None, facets = None):
        if self.n_of_faces == 0:
            Exception('Number of faces is 0. There is no surface to sample from.')

        if facets is None and faces is None:
            faces = np.arange(self.n_of_faces, dtype = int)
        else:
            try:
                int(facets)
                facets = np.array([facets])
            except: pass
            face_list = [self.facets[f] for f in facets]
            faces = np.concatenate(face_list)

        f = np.random.choice(faces, size = n, p = self.face_areas[faces]/self.face_areas[faces].sum()) # from which faces to draw the points

        v = self.vertices[self.faces[f, :], :] # (N, 3, 3)

        a = np.random.rand(n, 3, 1) # (N, 3, 1)
        a = -np.log(a)
        a /= a.sum(axis = 1, keepdims = True)

        x = np.sum(a*v, axis = 1) # (N, 3)

        return x

    def export_stl(self, name, path = None):
        '''Saves an ASCII stl file with the mesh.'''
        
        if path is None:
            path = os.getcwd()

        name.replace('.stl', '')

        content = 'solid {:s}\n'.format(name)

        for f in range(self.n_of_faces):
            content += 'facet normal {:.6e} {:.6e} {:.6e}\n'.format(self.face_normals[f, 0], self.face_normals[f, 1], self.face_normals[f, 2])
            content += '    outer loop\n'
            for v in range(3):
                content += '        vertex {:.6e} {:.6e} {:.6e}\n'.format(self.vertices[self.faces[f, v], 0],
                                                                          self.vertices[self.faces[f, v], 1],
                                                                          self.vertices[self.faces[f, v], 2])
            content += '    endloop\n'
            content += 'endfacet\n'
        content += 'endsolid {:s}'.format(name)
        
        with open(path+'/'+name+'.stl', 'w') as file:
            file.write(content)