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

def get_which_spheres(samples, spheres):
    
    n_samples = samples.shape[0]
    n_subvols = len(spheres)

    in_subvol = np.ones((n_samples, n_subvols))

    for i in range(n_subvols):
        in_subvol[:, i] = tm.proximity.signed_distance(spheres[i], samples) > 0

    return in_subvol

def get_cover(in_subvol):

    contained_any = np.any(in_subvol, axis = 1)

    cover = contained_any.mean()

    volumes = in_subvol.mean(axis = 0)
    
    volume_ratio = volumes/volumes.mean()
    volume_ratio = np.nan_to_num(volume_ratio, nan=0)

    return cover, volume_ratio

def update_centers(samples, in_subvols, centers):

    n_subvols = in_subvols.shape[1]

    how_many_spheres = in_subvols.astype(int).sum(axis = 1)

    new_centers = np.zeros((n_subvols, 3))  # initialise new_centers

    for i in range(n_subvols):                                  # check for each sphere
        indexes = in_subvols[:, i].astype(bool)
        unique_indexes = how_many_spheres[indexes] == 1

        in_points = samples[indexes, :]

        if in_points[unique_indexes].shape[0]>0:    # if there are points inside
            new_centers[i, :] = in_points[unique_indexes].mean(axis = 0) # to  calculate their centroid
        else:
            new_centers[i, :] = centers[i, :]

    return new_centers

def update_radius(radius, cover, volume_ratio):

    beta = (volume_ratio**2+cover**2)**0.5
    alpha = 1-np.exp(-(2**0.5))+np.exp(-beta)
    
    alpha = alpha**2    
    radius *= np.where(volume_ratio == 0, 1.1, alpha)

    return radius

def generate_subvols(shape, n_subvols, radius, centers):
    if shape == 'sphere':
        # generate spheres
        subvols = [tm.creation.icosphere(subdivisions = 1, radius = radius[i]).apply_translation(centers[i, :])
                    for i in range(n_subvols)]
    elif shape == 'box':
        subvols = [tm.creation.box(extents = np.ones(3)*radius[i]).apply_translation(centers[i, :])
                    for i in range(n_subvols)]
    
    return subvols

def distribute(geo, n_subvols, shape, folder, view = True):

    # geo = get_outer_hull(geo)   # ensure that geometry is watertight

    n_samples = int(1e3)        # initial number of points to test
    n_samples_max = int(1e5)    # maximum number of points to test
    iterations = 10            # number of iterations per try

    # initialising convergence lists
    centers_conv = []
    radius_conv = []
    volume_conv = []
    cover_conv = []

    # initialising variables
    tries = 0
    counter = 0
    solution_found = False
    best_intersection = 1
    best_spec_intersect = 1

    radius = np.ones(n_subvols)*geo.extents.max()/(2*n_subvols)
    centers = generate_points(geo, n_subvols)

    cover = 0
    volume_ratio = np.ones(n_subvols)

    # main loop
    while not solution_found:
        tries +=1
        while counter < tries*iterations:

            subvols = generate_subvols(shape, n_subvols, radius, centers)

            sample_points       = generate_points(geo, n_samples)           # sample geometry volume
            in_subvol           = get_which_spheres(sample_points, subvols) # get which spheres contain the samples
            cover, volume_ratio = get_cover(in_subvol)                      # get what percentage of the volume each sphere contains

            empty_subvols = np.any(~np.any(in_subvol, axis = 0)) # check if there is any empty subvol
            intersection = (in_subvol.sum(axis = 1)>1).mean()

            centers_conv.append(copy.copy(centers))
            radius_conv.append(copy.copy(radius))
            volume_conv.append(in_subvol.mean(axis = 0))
            cover_conv.append(copy.copy(cover))

            if (cover == 1) and (not empty_subvols):
            
                spec_intersect = intersection/n_samples

                if spec_intersect < best_spec_intersect:
                    best_intersection = intersection
                    best_spec_intersect = spec_intersect
                    solution_centers = copy.copy(centers)
                    solution_radius  = copy.copy(radius)
                    solution_iteration = counter
                
                radius = radius*(np.exp(-intersection/10))**(1/3)

            if cover == 1:
                if n_samples < n_samples_max:
                    n_samples = int(n_samples*2)
                if n_samples > n_samples_max:
                    n_samples = n_samples_max
                    solution_found = True
                    break

            counter += 1
            
            centers = update_centers(sample_points, in_subvol, centers)  # update centers
            radius  = update_radius(radius, cover, volume_ratio)   # update radius

        print('{:4d} - Cover: {:6.2f}%, Int: {:>.3f}, Best Int: {:.3f}, Samples: {:.2e}'.format(counter, cover*100, intersection, best_intersection, n_samples))

    print('Generating final meshes...')

    np.savetxt(fname = folder + 'subvolumes.txt',
               X = np.hstack((radius.reshape(-1, 1), centers)),
               fmt = '%.3f', delimiter = ',',
               header = 'Distribution of subvolumes. Type: '+shape+'\n Radius/Edge, Center x, Center y, Center z')
    
    # generating shapes
    subvols = generate_subvols(shape, n_subvols, solution_radius, solution_centers)

    # getting their intersection with the geometry
    final_subvols = [geo.intersection(subvol) for subvol in subvols]

    # check if all of them are watertight
    watertight = np.array([subvol.is_watertight for subvol in final_subvols])

    if np.any(~watertight):
        print('Subvolumes {} are not watertight! Try another configuration.'.format(np.where(~watertight)[0]))
        print('Interrupting simulation...')
        quit()
    else:
        print('All subvolumes are watertight. Continuing simulation...')
    
    if view:
        view_subvols(geo,
                     subvols,
                     final_subvols,
                     centers_conv,
                     radius_conv,
                     volume_conv,
                     cover_conv,
                     n_samples_max,
                     solution_centers,
                     solution_iteration)
    
    return final_subvols

def view_subvols(geo,
                 subvols,
                 final_subvols,
                 centers_conv,
                 radius_conv,
                 volume_conv,
                 cover_conv,
                 n_samples_max,
                 solution_centers,
                 solution_iteration):

    n_subvols = len(final_subvols)

    # visualising geometry
    geo.show()

    # visualising shapes
    scene = tm.Scene(subvols)
    scene.show()

    # visualising subvols
    scene = tm.Scene(final_subvols)
    scene.show()

    # plotting convergence
    centers_conv = np.array(centers_conv)
    radius_conv  = np.array( radius_conv)
    volume_conv  = np.array( volume_conv)
    cover_conv   = np.array(  cover_conv)

    fig = plt.figure(figsize = (20, 20), dpi = 200)
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    sample_points = generate_points(geo, n_samples_max)
    in_subvol = get_which_spheres(sample_points, final_subvols)

    unique_indexes       = in_subvol.sum(axis = 1) == 1
    out_indexes          = in_subvol.sum(axis = 1) == 0

    print('Total points:', n_samples_max)
    print('Out: {} -> {:.2f}%'.format(out_indexes.sum(), out_indexes.mean()*100))
    print('Unique: {} -> {:.2f}%'.format(unique_indexes.sum(), unique_indexes.mean()*100))
    print('Shape centers are in black. Out points in transparent red.')

    ax1.scatter(sample_points[out_indexes, 0],
                sample_points[out_indexes, 1],
                sample_points[out_indexes, 2], c = 'r', alpha = 0.1)

    for i in range(n_subvols):

        ax1.plot(centers_conv[:, i, 0], centers_conv[:, i, 1], centers_conv[:, i, 2])
        ax1.scatter(solution_centers[i, 0], solution_centers[i, 1], solution_centers[i, 2], c='k')
        ax2.plot(np.arange(radius_conv.shape[0]), radius_conv[:, i], '.')
        ax3.plot(np.arange(volume_conv.shape[0]), volume_conv[:, i], '.')

    ax4.semilogy(np.arange(cover_conv.shape[0]), cover_conv)
    ax2.plot([solution_iteration, solution_iteration], ax2.get_ylim(), 'r')
    ax3.plot([solution_iteration, solution_iteration], ax3.get_ylim(), 'r')
    ax4.plot([solution_iteration, solution_iteration], ax4.get_ylim(), 'r')

    ax2.set_title('Radius or Edge size')
    ax3.set_title('Relative Volume')
    ax4.set_title('Cover')

    plt.savefig('subvolumes.png')