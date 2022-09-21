# from statistics import LinearRegression
import trimesh as tm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

from scipy.stats.qmc import Sobol

# AI classifier models
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import OneHotEncoder

def generate_points(geo, n, gen = None):

    if gen is None:
        points = tm.sample.volume_mesh(geo, n)
        while points.shape[0]<n:
            new_points = tm.sample.volume_mesh(geo, n-points.shape[0])
            points = np.concatenate((points, new_points), axis = 0)
        return points
    else:
        points = gen.random(n)*geo.bounds.ptp(axis = 0) + geo.bounds[0, :]
        i = geo.contains(points)
        points = points[i, :]
        while points.shape[0]<n:
            new_points = gen.random(n-points.shape[0])*geo.bounds.ptp(axis = 0) + geo.bounds[0, :]
            i = geo.contains(new_points)
            points = np.concatenate((points, new_points[i, :]), axis = 0)
        return points

def get_regions(x_s, x_r):
    
    n_r = x_r.shape[0] # number of regions

    d = x_s - x_r.reshape(n_r, 1, 3)
    d = np.sum(d**2, axis = 2)**0.5

    min_d = np.amin(d, axis = 0, keepdims = True)

    r = np.argmax(d == min_d, axis = 0)

    return r

def get_cover(r, n_r):

    n_s = r.shape[0] # number of samples

    cvr = np.zeros(n_r)             # initialise cover array
    for i in range(n_r):            # for each region:
        ind = np.nonzero(r == i)[0] # get samples in region
        cvr[i] = ind.shape[0]/n_s   # save fraction of total

    return cvr

def update_centers(x_s, r, x_r):

    n_r = x_r.shape[0] # number of regions

    x_r_new = np.zeros((n_r, 3))  # initialise new_centers

    for i in range(n_r):                           # check for each region
        ind = np.nonzero(r == i)[0]
        x_r_new[i, :] = x_s[ind, :].mean(axis = 0) # to  calculate their centroid

    return x_r_new

def normalise(x, geo):
    return (x - geo.bounds[0, :])/np.ptp(geo.bounds, axis = 0)


# def train_model(x_r, n_s, geo):

#     n_r = x_r.shape[0]
#     x_s = generate_points(geo, n_s)
    
#     r = get_regions(x_s, x_r)
    
#     print('Enconding...')
    
#     ohe = OneHotEncoder(sparse = False)
#     y_train = ohe.fit_transform(r.reshape(-1, 1))

#     print('Training classifier...')

#     hl_size = int(5*n_r)

#     model = MLPClassifier(hidden_layer_sizes = (hl_size, hl_size, hl_size),
#                           activation         = 'logistic'   ,
#                           solver             = 'adam'       ,
#                           verbose            = 1            ,
#                           max_iter           = 1000         ,
#                           learning_rate      = 'adaptive'   )

#     x_train = normalise(x_s, geo)

#     model.fit(x_train, y_train) # train model

#     x_test = generate_points(geo, n_s) # generate test samples
#     r_test = get_regions(x_test, x_r)  # get their regions
    
#     x_test = normalise(x_test, geo)         # normalise

#     y_test = ohe.transform(r_test.reshape(-1, 1))

#     score = model.score(x_test, y_test) # calculate model score

#     print('Classifier trained! Estimated score: {:.3e}'.format(score)) # exhibit score

#     return model


def distribute(geo, n_r, folder, view = True):

    # geo = get_outer_hull(geo)   # ensure that geometry is watertight

    n_s = int(1e3)     # initial number of points to test
    n_s_max = int(1e6) # maximum number of points to test
    it = 10            # number of iterations per try
    criterion = 1e-8

    # initialising convergence lists
    centers_conv = []
    cover_conv = []
    displacement_conv = []

    # initialising variables
    tries = 0
    counter = 0
    solution_found = False
    
    gen = Sobol(3)

    x_r = generate_points(geo, n_r, gen) # regions coordinates
    x_s = generate_points(geo, n_s, gen) # samples coordinates

    # main loop
    while not solution_found:
        tries +=1
        while counter < tries*it:
            
            r   = get_regions(x_s, x_r) # get which spheres contain the samples
            cvr = get_cover(r, n_r)     # get what percentage of the volume each sphere contains

            x_r_new = update_centers(x_s, r, x_r)  # update centers

            dx_r = x_r_new - x_r

            centers_conv.append(copy.copy(x_r))
            cover_conv.append(copy.copy(cvr))
            displacement_conv.append(copy.copy(dx_r))

            # comparison = np.absolute(dx_r).max()
            comparison = np.linalg.norm(dx_r, axis = 1).max()

            if n_s < n_s_max and comparison < criterion:
                n_s = int(n_s*2)
                x_s = generate_points(geo, n_s, gen)
            if n_s >= n_s_max:
                n_s = n_s_max
                solution_found = comparison < criterion
                break

            counter += 1
            
            x_r = x_r_new
            
        print('{:4d} - Samples: {:.2e} - Max dx_r = {:.3e}'.format(counter, n_s, np.max(dx_r)))
    
    if view: # visualise results if requested
        view_subvols(geo, folder,
                     centers_conv,
                     cover_conv,
                     displacement_conv)
    
    # model = train_model(x_r, n_s_max, geo)

    # x_s = normalise(x_s, geo)

    # r = np.argmax(model.predict(x_s), axis = 1)

    # cvr = get_cover(r, n_r)

    # np.savetxt(fname = folder + 'subvolumes.txt',
    #            X = np.hstack((x_r, cvr.reshape(-1, 1))),
    #            fmt = '%.3f', delimiter = ',',
    #            header = 'Distribution of subvolumes. \n Center x, Center y, Center z')

    # return model, cvr, x_r
    return x_r

def view_subvols(geo, folder,
                 centers_conv,
                 cover_conv,
                 displacement_conv):

    x_r = centers_conv[-1]

    n_r = x_r.shape[0]
    
    # visualising geometry
    # geo.show()

    # plotting convergence
    centers_conv      = np.array(centers_conv)
    cover_conv        = np.array(  cover_conv)
    displacement_conv = np.array(displacement_conv)
    disp = np.linalg.norm(displacement_conv, axis = 2)

    fig = plt.figure(figsize = (20, 10), dpi = 200)
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # x_s = generate_points(geo, n_s_max)
    # r   = get_regions(x_s, x_r)

    for i in range(n_r):
        ax1.plot(centers_conv[:, i, 0], centers_conv[:, i, 1], centers_conv[:, i, 2])
        ax1.scatter(x_r[i, 0], x_r[i, 1], x_r[i, 2], c='k')
        ax2.plot(np.arange(cover_conv.shape[0]), cover_conv[:, i])
        ax3.plot(np.arange(disp.shape[0]), disp[:, i])
    
    ax1.legend(np.arange(n_r))
    ax2.legend(np.arange(n_r))
    ax3.legend(np.arange(n_r))

    ax1.set_title('Center coordinates')
    ax2.set_title('Relative cover volume')
    ax3.set_title('Center Displacement')

    plt.savefig(folder+'subvolumes_dist.png')

    plt.close(fig)