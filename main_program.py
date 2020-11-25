import os
from datetime import datetime
import numpy as np
from thermal_cond_module import *

# getting start time

start_time = datetime.now()

print(' ---------- o ----------- o ------------- o ------------')
print("Year: {:<4d}, Month: {:<2d}, Day: {:<2d}".format(start_time.year, start_time.month, start_time.day))
print("Start at: {:<2d} h {:<2d} min {:<2d} s".format(start_time.hour, start_time.minute, start_time.second))	
print(' ---------- o ----------- o ------------- o ------------')

# taking user input

filename = 'kappa-m20206.hdf5' #input("Data filename: ")

T_c = float(input("HOT temperature in K: ") )
T_f = float(input("COLD temperature in K: ") )

(L_x, L_y, L_z) = input("Dimensions L_x, L_y, L_z: ").split(',')

L_x = float(L_x)
L_y = float(L_y)
L_z = float(L_z)

(N_x, N_y, N_z) = input("Number of cells N_x, N_y, N_z: ").split(',')

N_x = int(N_x)
N_y = int(N_y)
N_z = int(N_z)

dt = float(input("Timestep: "))

N_dt = int(input("Number of timesteps: "))

# initialising grid

grid = Grid()

grid.set_dimensions(L_x, L_y, L_z)
grid.set_cells(N_x, N_y, N_z)
grid.set_boundaries()
grid.set_temperature(T_c, T_f)

# opening file

phonons = Phonon()
phonons.load_properties(filename)

k = np.random.rand(3)

print(k, phonons.get_frequency(k, 0))




