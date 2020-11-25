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

filename = 'kappa-m20206.hdf5'

data = h5py.File(filename,'r')

q_points = np.array(data['qpoint'])

x = q_points[:,0]
y = q_points[:,1]
z = q_points[:,2]

omega = np.array(data['frequency'])*2*np.pi

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

p = 0

cmap = matplotlib.cm.get_cmap('viridis')
norm_freq_matrix = (omega[:,p].max()-omega[:,p])/(omega[:,p].max()-omega[:,p].min())

ax.scatter(x, y, z, color = cmap(norm_freq_matrix))

phonons = Phonon()
phonons.load_properties(filename)

no_points = 10000

lengths = np.random.rand(no_points)
angles = np.random.rand(no_points)*np.pi/2

random_points = np.zeros( (no_points, 4) )
random_points[:,0] = lengths*np.cos(angles)
random_points[:,1] = lengths*np.sin(angles)
random_points[:,2] = np.random.rand(no_points)*0.5
random_frequencies = np.zeros(no_points)

for i in range(no_points):
    k = random_points[i, 0:3]
    random_frequencies[i] = phonons.get_frequency(k, p)

random_frequencies = np.nan_to_num(random_frequencies)

random_colors = (random_frequencies.max() - random_frequencies)/(random_frequencies.max()-random_frequencies.min())

print(random_frequencies)
print(omega)


ax.scatter(random_points[:,0], random_points[:,1], random_points[:,2], color = cmap(random_colors))


plt.tight_layout()
plt.show()