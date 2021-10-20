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
import copy

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

def get_which_spheres(samples, spheres):
    
    n_samples = samples.shape[0]
    n_subvols = len(spheres)

    in_subvol = np.ones((n_samples, n_subvols))

    for i in range(n_subvols):
        in_subvol[:, i] = spheres[i].contains(samples)

    return in_subvol

def get_cover(in_subvol, intersec_tol = None):

    contained_any = np.any(in_subvol, axis = 1)

    cover = contained_any.mean()

    volumes = in_subvol.mean(axis = 0)
    
    volume_ratio = volumes/volumes.mean()
    volume_ratio = np.nan_to_num(volume_ratio, nan=0)

    if intersec_tol is None:
        ideal_ratio = None
    else:
        ideal_volume = (1+intersec_tol)/n_subvols
        ideal_ratio = volumes/ideal_volume
    
    return cover, volumes, volume_ratio, ideal_ratio

def update_centers(samples, in_subvols, centers, noise, geo):

    n_subvols = in_subvols.shape[1]

    how_many_spheres = in_subvols.astype(int).sum(axis = 1)

    new_centers = np.zeros((n_subvols, 3))  # initialise new_centers

    for i in range(n_subvols):                                  # check for each sphere
        indexes = in_subvols[:, i].astype(bool)
        unique_indexes = how_many_spheres[indexes] == 1

        in_points = samples[indexes, :]

        #=#=#=#=#=#=#=#= CONSIDERING INTERSECTION CENTROID =#=#=#=#=#=#=#=#=#=#=#=#
        # if indexes.sum() == 0:  # if there is no point inside
            
        #     new_centers[i, :] = centers[i, :]   # keep the place
            
        # elif indexes.sum() == unique_indexes.sum():   # if there is no intersection (all in_points are unique)
            
        #     unique_centroid = in_points[unique_indexes].mean(axis = 0)  # calculate centroid of points
        #     new_centers[i, :] = unique_centroid                         # and use it as new center

        # elif indexes.sum() > unique_indexes.sum():  # if there are intersections
            
        #     unique_centroid = in_points[unique_indexes].mean(axis = 0)          # get unique centroid
        #     intersection_centroid = in_points[~unique_indexes].mean(axis = 0)   # get intersection centroid
        #     displacement = (unique_centroid - intersection_centroid)               # define displacement

        #     new_centers[i, :] = centers[i, :] + displacement*0.1

        #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

        if in_points[unique_indexes].shape[0]>0:    # if there are points inside
            new_centers[i, :] = in_points[unique_indexes].mean(axis = 0) # to  calculate their centroid
        else:
            new_centers[i, :] = centers[i, :]

    noise = noise*(np.random.rand(*new_centers.shape)-0.5)

    new_centers = new_centers+noise*geo.extents

    domain = tm.creation.box(extents = geo.extents)
    domain.rezero()
    in_domain = domain.contains(new_centers)

    new_centers[~in_domain, :] = generate_points(geo, (~in_domain).sum())

    return new_centers

def update_radius(radius, cover, volume_ratio, noise, noise_size = 1, ideal_ratio = None):

    if ideal_ratio is None:
        beta = (volume_ratio**2+cover**2)**0.5
        alpha = 1-np.exp(-(2**0.5))+np.exp(-beta)
    else:
        beta = (volume_ratio**2+cover**2+ideal_ratio**2)**0.5
        alpha = 1-np.exp(-(3**0.5))+np.exp(-beta)
    
    radius *= np.where(volume_ratio == 0, 1.1, alpha)

    radius += noise*noise_size*(np.random.rand(*radius.shape)-0.5)

    radius = np.where(radius <= 0, noise_size, radius)

    return radius

def update_noise(initial_noise, iterations, counter):
    noise = initial_noise*(1 - counter/iterations)
    return noise

def generate_subvols(shape, n_subvols, radius, centers):
    if shape == 'sphere':
        # generate spheres
        subvols = [tm.creation.icosphere(subdivisions = 1, radius = radius[i]).apply_translation(centers[i, :])
                    for i in range(n_subvols)]
    elif shape == 'box':
        subvols = [tm.creation.box(extents = np.ones(3)*radius[i]).apply_translation(centers[i, :])
                    for i in range(n_subvols)]
    
    return subvols

def obj_function(volumes, intersection, cover, radius):
    v = 0 # volumes.std()*volumes.shape[0]
    return 10*(cover<1)+(v**2+intersection**2+radius.mean()**2)**0.5

# %%
# Load Geometry
f = 'std_geo/cuboid.stl'
print('Mesh:', f)

geo = tm.load_mesh(f)
geo.rezero()

geo.vertices *= np.array([5, 1, 1])

geo = get_outer_hull(geo)

geo.show()
# %%
# optimisation process

n_subvols = 20
n_samples = int(1e4)
n_samples_max = int(1e4)
intersec_tol = 0.2  # if shape = 'sphere' this has no effect
iterations = 500
shape = 'sphere'
initial_noise = 0

centers_conv = []
radius_conv = []
volume_conv = []
cover_conv = []

# main loop

if shape == 'sphere':
        intersec_tol = None

tries = 0
counter = 0
solution_found = False
best_intersection = 1
best_obj = np.inf
best_vol_std = np.inf
best_cover = 0
n_acc = 50

obj = np.inf

while not solution_found:
    tries +=1
    print('No solution yet! Optimising...')
    new_radius = np.ones(n_subvols)*geo.extents.max()/(2*n_subvols)
    new_centers = generate_points(geo, n_subvols)

    T = 0.1
    acc = np.zeros(0).astype(bool)

    cover = 0
    volume_ratio = np.ones(n_subvols)
    noise = initial_noise
    
    while counter < tries*iterations:

        subvols = generate_subvols(shape, n_subvols, new_radius, new_centers)

        sample_points       = generate_points(geo, n_samples)           # sample geometry volume
        in_subvol           = get_which_spheres(sample_points, subvols) # get which spheres contain the samples
        cover, volumes, volume_ratio, ideal_ratio = get_cover(in_subvol, intersec_tol)                      # get what percentage of the volume each sphere contains

        empty_subvols = np.any(~np.any(in_subvol, axis = 0)) # check if there is any empty subvol
        intersection = (in_subvol.sum(axis = 1)>1).mean()
        new_obj = obj_function(volumes, intersection, cover, new_radius)

        prob = 1 # np.exp(-(new_obj-obj)/T)

        if (new_obj < obj) or (np.random.rand() < prob):
            centers = copy.copy(new_centers)
            radius = copy.copy(new_radius)
            obj = new_obj

            acc = np.append(acc, True)[-n_acc:]
            
        else:
            acc = np.append(acc, False)[-n_acc:]

        noise = noise*(0.02*acc.mean()+0.99)

        if (new_obj < best_obj):# and (cover == 1) and (not empty_subvols):
        
            solution_found = True
            best_obj = new_obj
            best_vol_std = volumes.std()/volumes.mean()
            best_cover = cover
            
            best_intersection = intersection
            
            solution_centers = copy.copy(centers)
            solution_radius  = copy.copy(radius)
            solution_iteration = counter
        
        centers_conv.append(copy.copy(centers))
        radius_conv.append(copy.copy(radius))
        volume_conv.append(in_subvol.mean(axis = 0))
        cover_conv.append(copy.copy(cover))

        print('{:4d} - Cover: {:6.2f}%, Obj: {:>6.3f}, Best Obj: {:>6.3f}, Best Int: {:.3f}, Best Vol Std. {:.3f}, Samples: {:.2e}'.format(counter, cover*100, new_obj, best_obj, best_intersection, best_vol_std, n_samples))

        counter += 1

        new_centers = update_centers(sample_points, in_subvol, centers, noise, geo)  # update centers
        new_radius  = update_radius(radius, cover, volume_ratio, noise, ideal_ratio = ideal_ratio)   # update radius

        T = T*0.99 # update_noise(initial_noise, iterations, counter)

        
        
    

# %%
# Visualising results
print('Generating final meshes...')
final_subvols = generate_subvols(shape, n_subvols, solution_radius, solution_centers)

print([subvol.is_watertight for subvol in final_subvols])

scene = tm.Scene(final_subvols)
scene.show()

centers_conv = np.array(centers_conv)
radius_conv  = np.array( radius_conv)
volume_conv  = np.array( volume_conv)
cover_conv   = np.array(  cover_conv)

fig = plt.figure()
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

sample_points = generate_points(geo, n_samples_max)
in_subvol = get_which_spheres(sample_points, subvols)

unique_indexes       = in_subvol.sum(axis = 1) == 1
intersection_indexes = in_subvol.sum(axis = 1) > 1
out_indexes          = in_subvol.sum(axis = 1) == 0

print('Total points:', n_samples)
print('Out: {} -> {:.2f}%'.format(out_indexes.sum(), out_indexes.mean()*100))
print('Unique: {} -> {:.2f}%'.format(unique_indexes.sum(), unique_indexes.mean()*100))


# ax1.scatter(sample_points[intersection_indexes, 0],
#            sample_points[intersection_indexes, 1],
#            sample_points[intersection_indexes, 2], c = 'y', alpha = 0.1)

ax1.scatter(sample_points[out_indexes, 0],
            sample_points[out_indexes, 1],
            sample_points[out_indexes, 2], c = 'r', alpha = 0.1)

unique_samples = sample_points[unique_indexes, :]

for i in range(n_subvols):
    indexes = in_subvol[unique_indexes, i].astype(bool)
    
    # ax1.scatter(unique_samples[indexes, 0],
    #            unique_samples[indexes, 1],
    #            unique_samples[indexes, 2], alpha = 0.1)

    ax1.plot(centers_conv[:, i, 0], centers_conv[:, i, 1], centers_conv[:, i, 2])
    ax1.scatter(centers_conv[-1, i, 0], centers_conv[-1, i, 1], centers_conv[-1, i, 2], c='k')
    ax2.plot(np.arange(radius_conv.shape[0]), radius_conv[:, i], '.')
    ax3.plot(np.arange(volume_conv.shape[0]), volume_conv[:, i], '.')

ax4.plot(np.arange(cover_conv.shape[0]), cover_conv)
# points = generate_points(geo, 500)
ax2.plot([solution_iteration, solution_iteration], ax2.get_ylim(), 'r')
ax3.plot([solution_iteration, solution_iteration], ax3.get_ylim(), 'r')
ax4.plot([solution_iteration, solution_iteration], ax4.get_ylim(), 'r')


ax2.set_title('Radius/Edge size')
ax3.set_title('Relative Volume')
ax4.set_title('Cover')

# ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha = 0.2)
plt.show()