# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# get_ipython().run_line_magic('matplotlib', 'ipympl')

import trimesh as tm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import copy
import time

def generate_points(geo, n):
    points = tm.sample.volume_mesh(geo, n)
    while points.shape[0]<n:
        new_points = tm.sample.volume_mesh(geo, n-points.shape[0])
        points = np.concatenate((points, new_points), axis = 0)
    return points

def get_outer_hull(geo):

    # bounding box mesh
    box = tm.creation.box(extents = geo.extents+0.2)
    box.rezero()
    box.vertices -= 0.1

    # if mesh is composed of more than one mesh, add them up
    if isinstance(geo, tm.Scene):
        print(geo.dump(concatenate=True))
    else:
        submesh = geo

    # get external faces of the mesh to make it watertight
    exponent = 4
    known_faces = np.empty(0)
    i = 0

    print('Number of bodies: ', geo.body_count)

    while (not submesh.is_watertight) and i<1000:
        
        i = i+1
        print('Try', i)

        n_points = 10**exponent

        # sampling points on bounding box
        start_points, _ = tm.sample.sample_surface(box, n_points)
        end_points  , _ = tm.sample.sample_surface(box, n_points)
        directions = end_points - start_points

        # detecting hit faces
        tri, _ = geo.ray.intersects_id(ray_origins = start_points, ray_directions = directions, multiple_hits = False)

        # get all unique faces that were hit
        valid_faces = np.unique(tri[tri >= 0])

        # add detected faces to the external ones
        known_faces = np.unique(np.concatenate((known_faces, valid_faces))).astype(int)

        # generate submesh with known faces
        submesh = tm.util.submesh(mesh = geo, faces_sequence = [known_faces], append = True)
        
        # submesh.show()

        # Try to make submesh watertight
        try:
            submesh.process(validate=True)
            submesh.fill_holes()
            submesh.fix_normals()
        except:
            pass
        print(known_faces.shape[0], len(geo.faces), submesh.is_watertight)

    # finalise submesh
    submesh.process(validate=True)
    submesh.fill_holes()
    submesh.fix_normals()
    submesh.show()

    # substitute geometry with submesh
    geo = submesh

    return geo

def scene_to_mesh(scene):
    if isinstance(scene, tm.Scene):
        if len(scene.geometry) == 0:
                mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = tm.util.concatenate(
                tuple(tm.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene.geometry.values()))
    else:
        assert(isinstance(scene, tm.Trimesh))
        mesh = scene
    return mesh

def get_covered_volume(geo, spheres):
    # get intersections
    subvols = [scene_to_mesh(sphere.intersection(geo)) for sphere in spheres]

    # check if all subvolumes are meshes
    for i in range(len(subvols)):
        subvol = subvols[i]
        if subvol.body_count > 1:   # get only the larger body of the subvolume
            bodies = subvol.split()
            body_vol = np.array([body.volume for body in bodies])
            subvols[i] = bodies[np.argmax(body_vol)]
    
    print('Subvols are WT:', [subvol.is_watertight for subvol in subvols])

    total_spheres = tm.boolean.union(spheres, engine = 'blender')
    total_volume = tm.boolean.intersection([geo, total_spheres], engine = 'blender')
    # total_volume = tm.boolean.union(subvols, engine = 'blender')
    
    # i = 1
    # while not total_volume.is_watertight:
    #     print('Try {} to WT total vol'.format(i))
    #     total_volume = get_outer_hull(total_volume)
    #     total_volume.process(validate = True)
    #     total_volume.fill_holes()
    #     i = i+1

    
    total_volume = scene_to_mesh(total_volume)

    print('Total Vol is WT:', total_volume.is_watertight, 'Total Vol:', total_volume.volume, 'Geo Vol:',geo.volume)
    # total_volume.show()

    cover = total_volume.volume/geo.volume

    return total_volume, subvols, cover # returns the total covered volume as a mesh and a list of subvolumes' meshes

def get_unique_subvols(subvols):
    unique_subvols = []
    for i in range(n_spheres):
        subvol = subvols[i]
        other_subvols = copy.copy(subvols)              # gather all subvolumes
        other_subvols.pop(i)                            # remove current subvolume
        other_subvols = tm.boolean.union(other_subvols, engine = 'blender') # sum up all other subvolumes
        
        intersection  = tm.boolean.intersection([subvol, other_subvols], engine = 'blender')
        unique_subvol = tm.boolean.difference([subvol, intersection], engine = 'blender')

        unique_subvol = scene_to_mesh(unique_subvol)
        
        unique_subvols.append(unique_subvol) # get the unique volume covered by the sphere
    
    print('Unique subvols are WT:', [unique_subvol.is_watertight for unique_subvol in unique_subvols])

    return unique_subvols

def update_centers(centers, subvols, extents):
    n_spheres = centers.shape[0]
    
    alpha_A = 1e-2
    alpha_R = 1e-2 #1e-3/n_spheres
    beta = 3
    max_force = 0.1

    centers = centers/extents   # normalisation - PAY ATTENTION TO REZERO THE DOMAIN

    centers_of_mass = np.array([subvol.center_mass if subvol.is_watertight else subvol.centroid for subvol in subvols])/extents
    
    d_cm = centers_of_mass - centers
    d_cm = (d_cm**2).sum(axis = 1)
    d_cm = d_cm**(1/2)
    
    force_A = alpha_A*(centers_of_mass - centers)/(d_cm.reshape(-1, 1))**beta
    
    d_sp = centers.reshape(n_spheres, 1, 3) - centers
    d_sp = d_sp**2
    d_sp = d_sp.sum(axis = 2)
    d_sp = d_sp.reshape(n_spheres, n_spheres, 1)

    force_R = alpha_R* (centers.reshape(n_spheres, 1, 3) - centers)/d_sp**beta
    force_R = np.where(np.isnan(force_R), 0, force_R)
    force_R = force_R.sum(axis = 0)

    force = force_A - force_R
    force[force>max_force] = max_force
    force[force<-max_force] = -max_force

    new_centers = centers + force
    new_centers *= extents

    print(new_centers - centers)

    return new_centers

def update_radius(radius, subvols, cover):

    volumes = np.array([subvol.volume for subvol in subvols])
    volume_ratio = volumes/volumes.mean()

    beta = (volume_ratio**2+cover**2)**0.5
    alpha = 1-np.exp(-(2**0.5))+np.exp(-beta)
    radius *= alpha

    print(cover, volume_ratio)

    return radius

# %%
# Load Geometry
f = 'std_geo/icosahedron.obj'
print('Mesh:', f)

geo = tm.load_mesh(f)
geo.rezero()

geo = get_outer_hull(geo)

# %%
# optimisation process

n_spheres = 3
n_samples = 1e3
iterations = 20

centers_conv = []

# initialise spheres
print('Initialising spheres')
radius = np.ones(n_spheres)*geo.extents.max()/n_spheres
centers = generate_points(geo, n_spheres)

# main loop
cover = 0
for _ in range(iterations):
    
    # print centers and radius of the generated spheres
    # print(centers)
    # print(radius)

    centers_conv.append(centers)

    # generate spheres
    spheres = [tm.creation.icosphere(subdivisions = 1, radius = radius[i]).apply_translation(centers[i, :])
                for i in range(n_spheres)]

    # scene = tm.Scene([geo]+spheres)
    # scene.show()

    # get subvolumes = intersection between geometry and sphere
    # print('Detecting intersections with geometry.')
    total_volume, subvols, cover = get_covered_volume(geo, spheres)
    
    # get which parts are convered uniquely by each sphere
    unique_subvols = get_unique_subvols(subvols)

    # updating centers
    centers = update_centers(centers, unique_subvols, geo.extents)
    radius = update_radius(radius, subvols, cover)

    # print('Is watertight:', [subvol.is_watertight for subvol in subvols])

    


# %%
scene = tm.Scene(subvols)
scene.show()

centers_conv = np.array(centers_conv)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(n_spheres):
    ax.plot(centers_conv[:, i, 0], centers_conv[:, i, 1], centers_conv[:, i, 2])
    ax.scatter(centers_conv[-1, i, 0], centers_conv[-1, i, 1], centers_conv[-1, i, 2], c='k')

points = generate_points(geo, 500)

ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha = 0.2)

plt.show()
