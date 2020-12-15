# THIS FILE IS JUST A DUMMY FILE TO TEST GENERAL THINGS BEFORE APPLYING TO THE CODE
# EXAMPLES: VISUALISE DATA, TEST FUNCTIONS, TRY DIFFERENT CALCULATION METHODS

import numpy as np
import scipy.stats as st
# from scipy.interpolate import RegularGridInterpolator as rg
import copy
import time

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm

import h5py
import dpdata
from thermal_cond_module import *

import pymesh as pm
import pywavefront as pwf
from pywavefront import visualization

body = pm.load_mesh('std geo/untitled.stl')
body.enable_connectivity()


for i in range(10):

    n = 10**i

    start = time.time()

    point = np.random.rand(n,3)*6-3

    wind = pm.compute_winding_number(body, point)

    end = time.time()

    

    print('10^{} - {:.3f} sec'.format(i, end-start))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(closest_pts[:,0], closest_pts[:,1], closest_pts[:,2], marker='o', color='b')
# ax.scatter(point[:,0], point[:,1], point[:,2], marker='o', color='r')
# ax.scatter(body.vertices[:,0], body.vertices[:,1], body.vertices[:,2], marker='o', color='g')

# plt.tight_layout()
# plt.show()
