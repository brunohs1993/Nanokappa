import numpy as np

def project_point_on_plane(p, n, o):
    '''Project point p on a plane with normal n and origin o.'''
    return p - n*np.sum(n*(p-o)) # projecting on plane

def cartesian_to_barycentric(p, v):
    '''Calculate barycentric coordinates of the projection of point p
        in triangle defined by an array of vertices v.'''

    b1 = v[1, :] - v[0, :]
    b2 = v[2, :] - v[0, :]

    n = np.cross(b1, b2)
    n /= np.linalg.norm(n)

    p = project_point_on_plane(p, n, v[0, :])
    bar, _, _ = transform_3d_to_2d(p, n, origin = v[0, :], b1 = b1, b2 = b2)

    bar = np.vstack((bar[:, 0], bar[:, 1], 1 - bar.sum(axis = 1))).T

    return bar

def plane_constant(n, o):
    '''Gets the plane constant from its normal and origin point.'''
    # finds and stores the constant in the plane equation:
    # n . x + k = 0
    return -np.sum(n*o) # k for each facet

def line_plane_intersection(lo, ld, pn, po, return_t = False):
    '''Finds the point of intersection between
       line with origin lo and direction ld
       and the plane with normal pn and origin po.'''
    
    if np.sum(pn*ld) == 0:
        if return_t:
            return np.ones(3)*np.inf, np.inf
        else:
            return np.ones(3)*np.inf
    else:
        t = np.sum(pn*(po - lo))/(np.sum(pn*ld))
        if return_t:
            return lo + t*ld, t
        else:
            return lo + t*ld

def plane_plane_interesection(n1, o1, n2, o2):
    '''Find an origin and a direction for the line that
       intersects two planes. Line origin will have z = 0.'''

    d = np.cross(n1, n2)
    A = np.vstack((n1[0, 1], n2[0, 1]))
    b = np.array([np.sum(n1*o1), np.sum(n2*o2)])

    o = np.linalg.solve(A, b) # A.o = b
    o = np.concatenate((o, [0]))

    return d, o

def closest_points_line_line(o1, d1, o2, d2, return_t = False):
    '''Finds the closest points in each line for which
       the distance is the smallest.'''

    if np.absolute(np.sum(d1*d2)) == 1:
        # if lines are parallel, all points have the same distance
        if return_t:
            return o1, o2, 0, 0
        else:
            o1, o2

    d1d1 = np.sum(d1*d1)
    d2d2 = np.sum(d2*d2)
    d1d2 = np.sum(d1*d2)
    o1d1 = np.sum(o1*d1)
    o1d2 = np.sum(o1*d2)
    o2d1 = np.sum(o2*d1)
    o2d2 = np.sum(o2*d2)

    A1 = d1d1 - d1d2*d1d2/d2d2
    A2 = d2d2 - d1d2*d1d2/d1d1

    B1 = (o1d1 - o2d1) - (o2d2 - o1d2)*d1d2/d2d2
    B2 = (o2d2 - o1d2) - (o1d1 - o2d1)*d1d2/d1d1

    t1 = B1/A1
    t2 = B2/A2

    x1 = o1 + t1*d1
    x2 = o2 + t2*d2

    if return_t:
        return x1, x2, t1, t2
    else:
        return x1, x2

def triangle_area(v):
    return np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]))

def triangle_normal(v):
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return n/np.linalg.norm(n)

def line_triangle_intersection(lo, ld, v, return_t = False):
    '''Finds the intersection points between line with
       origin lo and direction ld and triangle defined
       by vertices v.'''

    n = triangle_normal(v) # get normal
    if return_t:
        c, t = line_plane_intersection(lo, ld, n, v[0, :], return_t = True) # find intersection point
    else:
        c = line_plane_intersection(lo, ld, n, v[0, :]) # find intersection point
    b = cartesian_to_barycentric(c, v)

    if np.any(b<0) or np.any(b > 1):
        return np.ones(3)*np.nan, np.inf
    else:
        if return_t:
            return c, t
        else:
            return c

def transform_2d_to_3d(v, b1, b2, o):
    '''v = vertices
       b1 and b2 = basis
       o = origin'''
    return np.expand_dims(v[:, 0], 1)*b1 + np.expand_dims(v[:, 1], 1)*b2 + o

def transform_3d_to_2d(points, normal, origin = None, b1 = None, b2 = None):
    '''Transform 3d points to the 2d projection on a plane
    defined by normal and origin. The other two base vectors are chosen as
    the projection of unit x (or unit y if the normal is already equal to
    unit x) in the plane and the cross product between normal and b1.
    
    Returns the coordinates of the points projected onto the plane and the base vectors.
    '''
    if len(points.shape) == 1:
        points = np.expand_dims(points, 0)

    if origin is None:
        origin = points[0, :]

    if b1 is None or b2 is None:
        b = np.eye(3)
        is_normal = 1 - np.sum(b*np.absolute(normal), axis = 1) < 1e-3

        if np.any(is_normal):
            b = b[~is_normal, :]

        # if b1 is parallel to b1
        b1 = b[0, :]
        b1 = b1 - normal*np.sum(normal*b1)  # make b1 orthogonal to the normal
        b1 = b1/np.sum(b1**2)**0.5          # normalise b1

        b2 = np.cross(normal, b1)           # generate b2 = n x b1
    
    A = np.vstack((b1, b2, normal)).T   # get x and y bases coefficients
    
    plane_coord = np.zeros((0, 2))
    for p in points:             # for each vertex

        B = np.expand_dims((p - origin), 1) # get relative coord
        
        plane_coord = np.vstack((plane_coord, np.linalg.solve(A, B).T[0, :2])) # store the plane coordinates
    
    return plane_coord, b1, b2

def triangulate_plane(v, b = None):
    '''Triangulate given vertices by taking the first three vertices as reference of the plane.
       The other points will be projected onto this plane.
       Returns the new vertices and the vertex indices for each triangle.
       b defines the boundaries (rings) of the geometry. If b is not informed, no restriction will be imposed.'''
    
    n = triangle_normal(v[0:3, :]) # get normal
    o = v[0, :]                    # get origin

    for i in range(v.shape[0]): # for each point
        v[i, :] = project_point_on_plane(v[i, :], n, o) # project on plane

    tri = triangulate_points(v)
    
    return v

def triangulate_points(v):
    '''Delaunay triangulation of points in 3D space.'''

    

    