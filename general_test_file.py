# THIS FILE IS JUST A DUMMY FILE TO TEST GENERAL THINGS BEFORE APPLYING TO THE CODE
# EXAMPLES: VISUALISE DATA, TEST FUNCTIONS, TRY DIFFERENT CALCULATION METHODS

import numpy as np
import scipy.stats as st
# from scipy.interpolate import RegularGridInterpolator as rg
import copy

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm

import h5py
import dpdata
from thermal_cond_module import *

hdf5_file = 'kappa-m20206.hdf5'
poscar_file = 'POSCAR-unitcell'

data_poscar = dpdata.System(poscar_file, fmt = 'vasp/poscar') 

phonon = Phonon()
phonon.load_properties(hdf5_file, poscar_file)

population = Population(10000)
population.atribute_modes(phonon)
population.atribute_properties(phonon)

v_old = copy.deepcopy(population.velocities)

population.randomize_drift_directions()

v_new = copy.deepcopy(population.velocities)

print(np.any( np.abs(v_new-v_old)>0) )

branches = phonon.frequency.shape[1]
q_points = phonon.frequency.shape[0]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(population.indexes[:,0], bins = q_points)

plt.tight_layout()
plt.show()