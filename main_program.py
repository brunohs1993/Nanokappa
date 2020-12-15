import os
from datetime import datetime
import numpy as np
from thermal_cond_module import *
import argparse

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Setting arguments:

parser = argparse.ArgumentParser()

parser.add_argument('--geometry'   , '-g' , help='Geometry of the domain.'                             , default = 'cuboid'          , type = str  , nargs = 1  ) # SET SOME STANDARD SIMPLE GEOMETRIES (POLYHEDRA, CIRCLE, HEMISPHERE, ETC), AND MAYBE THINK ABOUT IMPORTING EXTERNAL GEOMETRIES, DEFINED BY AN OBJ FILE, FOR EXAMPLE. THE PYMESH MODULE COULD HELP WITH THAT.
parser.add_argument('--dimensions' , '-d' , help='Dimensions in angstroms according to given geometry.', default = [1, 1, 1]         , type = float, nargs = '*') # number of dimensions may vary with geometry
parser.add_argument('--particles'  , '-p' , help='Number of particles.'                                , default = 100               , type = int  , nargs = 1  )
parser.add_argument('--timestep'   , '-t' , help='Timestep size in seconds'                            , default = 0.01              , type = float, nargs = 1  )
parser.add_argument('--iterations' , '-i' , help='Number of timesteps (iterations) to be run'          , default = 1000              , type = int  , nargs = 1  )
parser.add_argument('--bound_cond' , '-bc', help='Set boundary conditions'                             , default = [(0,290),(5, 300)], type = int  , nargs = '*') # THOUGHT ABOUT GIVING NUMBERS FOR IMPOSED TEMPERATURE, 'ISO' FOR ISOLATED AND 'PER' FOR PERIODIC. STUDY WICH TYPES OF BOUNDARY CONDITIONS TO USE. NEED TO INDICATE FACES. THIS COULD GIVE FLEXIBILITY IF LOADING A CUSTOM GEOMETRY.

parser.add_argument('--poscar_file', '-pf', help='Set the POSCAR file to be read.', required = True, type = str) # lattice properties
parser.add_argument('--hdf_file'   , '-hf', help='Set the hdf5 file to be read.'  , required = True, type = str) # phonon properties of the material

args = parser.parse_args()

# getting start time

start_time = datetime.now()

print(' ---------- o ----------- o ------------- o ------------')
print("Year: {:<4d}, Month: {:<2d}, Day: {:<2d}".format(start_time.year, start_time.month, start_time.day))
print("Start at: {:<2d} h {:<2d} min {:<2d} s".format(start_time.hour, start_time.minute, start_time.second))	
print(' ---------- o ----------- o ------------- o ------------')

# initialising grid

geo = Geometry(args)

geo.set_dimensions()

# opening file

phonons = Phonon(args)
phonons.load_properties()

pop = Population(args, geo)

print(phonons.group_vel.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

band = 0

ax.scatter(np.abs(phonons.group_vel[:,band,0]), np.abs(phonons.group_vel[:,band,1]), np.abs(phonons.group_vel[:,band,2]), cmap='flag')

plt.tight_layout()
plt.show()