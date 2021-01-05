# THIS FILE IS JUST A DUMMY FILE TO TEST GENERAL THINGS BEFORE APPLYING TO THE CODE
# EXAMPLES: VISUALISE DATA, TEST FUNCTIONS, TRY DIFFERENT CALCULATION METHODS

import numpy as np
import scipy.stats as st
import copy
import time

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm

import h5py
from thermal_cond_module import *

import trimesh
import pyiron.vasp.structure as pistr

phonons = h5py.File('kappa-m20206.hdf5')

gamma = np.array(phonons['gamma'])

tau = np.where(gamma>0, 1/(4*np.pi*gamma), 0)

print( (gamma == 0).sum(), (gamma.shape[0]*gamma.shape[1]*gamma.shape[2]) )

quit()



body = trimesh.load_mesh('std_geo/untitled.stl')    # load mesh

atoms = pistr.read_atoms('POSCAR-unitcell')


n = 100

point    = np.random.rand(n,3)*4-2

velocity = np.random.rand(n,3)-0.5

faces, rays = body.ray.intersects_id(point, velocity, multiple_hits = False)    # check colisions

print(faces)
print(rays )

is_in = body.contains(point)    # check inside

in_ind  = np.where(is_in == True)[0]
out_ind = np.where(is_in == False)[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(point[in_ind ,0], point[in_ind ,1], point[in_ind ,2], marker='o', color='r')
ax.scatter(point[out_ind,0], point[out_ind,1], point[out_ind,2], marker='o', color='b')
ax.scatter(body.vertices[:,0], body.vertices[:,1], body.vertices[:,2], marker='o', color='g')

plt.tight_layout()
plt.show()
