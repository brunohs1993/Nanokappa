# THIS FILE IS JUST A DUMMY FILE TO TEST GENERAL THINGS BEFORE APPLYING TO THE CODE
# EXAMPLES: VISUALISE DATA, TEST FUNCTIONS, TRY DIFFERENT CALCULATION METHODS

import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm

import h5py
from thermal_cond_module import *

hdf5_file = 'kappa-m20206.hdf5'
poscar_file = 'POSCAR-unitcell'

phonons = Phonon()
phonons.load_properties(hdf5_file, poscar_file)
energy = phonons.calculate_energy(100, phonons.omega)

print(phonons.omega)
print(energy)

print(ct.hbar)




# fig = plt.figure()
# cmap = matplotlib.cm.get_cmap('viridis')

# for i in range(1,19):
#     p = i-1
#     ax = fig.add_subplot(3,6,i, projection='3d')

#     # norm_freq_matrix = (omega[:,p].max()-omega[:,p])/(omega[:,p].max()-omega[:,p].min())
#     norm_freq_matrix = (omega.max()-omega)/(omega.max()-omega.min())

#     ax.scatter(x, y, z, color = cmap(norm_freq_matrix[:,p]))

#     ax.set_xlim(0, 0.5)
#     ax.set_ylim(0, 0.5)
#     ax.set_zlim(0, 0.5)


# grid = Grid()
# grid.set_cells(5, 1, 1)
# grid.get_allowed_k()

# omega = phonons.get_frequency(grid.k_grid, 0)

# print(omega)

# RANDOM POINTS

# no_points = 10000
# filename = 'kappa-m20206.hdf5'

# data = h5py.File(filename,'r')

# q_points = np.array(data['qpoint'])

# x = q_points[:,0]
# y = q_points[:,1]
# z = q_points[:,2]

# omega = np.array(data['frequency'])*2*np.pi

# fig = plt.figure()
# cmap = matplotlib.cm.get_cmap('viridis')

# for i in range(1,19):
#     p = i-1
#     ax = fig.add_subplot(3,6,i, projection='3d')

#     # norm_freq_matrix = (omega[:,p].max()-omega[:,p])/(omega[:,p].max()-omega[:,p].min())
#     norm_freq_matrix = (omega.max()-omega)/(omega.max()-omega.min())

#     ax.scatter(x, y, z, color = cmap(norm_freq_matrix[:,p]))

#     ax.set_xlim(0, 0.5)
#     ax.set_ylim(0, 0.5)
#     ax.set_zlim(0, 0.5)

# random_points = np.zeros( (no_points, 4) )
# random_points[:,0] = lengths*np.cos(angles)
# random_points[:,1] = lengths*np.sin(angles)
# random_points[:,2] = np.random.rand(no_points)*0.5
# random_frequencies = np.zeros(no_points)

# for i in range(no_points):
#     k = random_points[i, 0:3]
#     random_frequencies[i] = phonons.get_frequency(k, p)

# random_frequencies = np.nan_to_num(random_frequencies)

# random_colors = (random_frequencies.max() - random_frequencies)/(random_frequencies.max()-random_frequencies.min())

# ax.scatter(random_points[:,0], random_points[:,1], random_points[:,2], color = cmap(random_colors))




# plt.tight_layout()
# plt.show()