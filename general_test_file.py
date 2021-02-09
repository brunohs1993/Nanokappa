# THIS FILE IS JUST A DUMMY FILE TO TEST GENERAL THINGS BEFORE APPLYING TO THE CODE
# EXAMPLES: VISUALISE DATA, TEST FUNCTIONS, TRY DIFFERENT CALCULATION METHODS

import numpy as np
import sys
from scipy.interpolate import interp1d
# from thermal_cond_module import *
np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=np.nan)


modes = 10
branches = 5
particles = 1000
temperatures = 29
dT = 0.1
sample = 100

temperature_array = np.arange(10, 10+temperatures, dT)

mode_omega = np.random.rand(modes, branches)*100

mode_occupation = 1/( np.exp( mode_omega/temperature_array.reshape(-1, 1, 1) ) - 1)

mode_energy = mode_omega*(0.5+mode_occupation)

particle_modes = np.random.rand(particles, 2)

particle_modes[:,0] *= modes
particle_modes[:,1] *= branches
particle_modes = np.floor(particle_modes).astype(int)

particle_omega = mode_omega[particle_modes[:,0], particle_modes[:,1]]

particle_T = np.random.rand(particles)*temperature_array.ptp()+temperature_array.min()
particle_occupation = 1/( np.exp( particle_omega/particle_T ) - 1)

particle_energy = particle_omega*(0.5+particle_occupation)

indexes = np.floor(np.random.rand(sample)*particles).astype(int)

selected_modes = particle_modes[indexes, :]

selected_energies = mode_energy[:, selected_modes[:,0], selected_modes[:,1]]

mean_energies_T = selected_energies.sum(axis = 1)

mean_energies_particles = particle_energy[indexes].sum()

interpolated_T = interp1d(mean_energies_T, temperature_array, kind = 'cubic')(mean_energies_particles) 

# print('Original energies and T:')
# print(mean_energies_T, mean_energies_particles)
# print(temperature_array, interpolated_T)


particle_occupation[indexes] = 1/( np.exp( particle_omega[indexes]/interpolated_T ) - 1)
particle_energy[indexes] = particle_omega[indexes]*(0.5+particle_occupation[indexes])

print('Initial energy:', mean_energies_particles)
print('Corrected mean energy:', particle_energy[indexes].sum())

print('Error:', (particle_energy[indexes].sum() - mean_energies_particles)*100/mean_energies_particles, '%')





