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

# SET SOME STANDARD SIMPLE GEOMETRIES (POLYHEDRA, CIRCLE, HEMISPHERE, ETC), AND MAYBE THINK ABOUT IMPORTING EXTERNAL GEOMETRIES, DEFINED BY AN OBJ FILE, FOR EXAMPLE. THE PYMESH MODULE COULD HELP WITH THAT.
parser.add_argument('--geometry'   , '-g' , default = 'cuboid'          , type = str  , nargs = 1  , help='Geometry of the domain. Standard shapes are cuboid, cylinder, cone and capsule')
parser.add_argument('--dimensions' , '-d' , default = [1, 1, 1]         , type = float, nargs = '*', help='Dimensions as asked by trimesh.creation primitives. 3 for box, 2 for others. Radius first.')
parser.add_argument('--scale'      , '-s' , default = [1, 1, 1]         , type = float, nargs = 3  , help='Scaling factors (x, y, z) to be applied to given geometry.')
parser.add_argument('--rotation'   , '-r' , default = [0, 0, 0]         , type = float, nargs = 3  , help='Euler angles in degrees to be applied to given geometry (see scipy.rotation.from_euler).')
parser.add_argument('--rot_order'  , '-ro', default = 'xyz'             , type = str  , nargs = 1  , help='Order of rotation to be applied to given geometry (see scipy.rotation.from_euler).')
parser.add_argument('--particles'  , '-p' , default = 1e4               , type = int  , nargs = 1  , help='Number of particles.')
parser.add_argument('--timestep'   , '-ts', default = 0.01              , type = float, nargs = 1  , help='Timestep size in seconds')
parser.add_argument('--iterations' , '-i' , default = 1000              , type = int  , nargs = 1  , help='Number of timesteps (iterations) to be run')
parser.add_argument('--slices'     , '-sl', default = (10, 0)           , type = int  , nargs = 2  , help='Number of slices and slicing axis (x = 0, y = 1, z = 2)')
parser.add_argument('--temperatures','-t' , default = [310, 290]        , type = float, nargs = 2  , help='Set first and last slice temperatures to be imposed.') 
parser.add_argument('--temp_dist'  ,'-td' , default = 'random'          , type = str  , nargs = 1  , help='Set how to distribute initial temperatures.') 

# THOUGHT ABOUT GIVING NUMBERS FOR IMPOSED TEMPERATURE, 'ISO' FOR ISOLATED AND 'PER' FOR PERIODIC. STUDY WICH TYPES OF BOUNDARY CONDITIONS TO USE. NEED TO INDICATE FACES. THIS COULD GIVE FLEXIBILITY IF LOADING A CUSTOM GEOMETRY.
# parser.add_argument('--bound_cond' , '-bc', default = [(0,290),(5, 300)], type = int  , nargs = '*', help='Set boundary conditions') 

parser.add_argument('--poscar_file', '-pf', required = True, type = str, help='Set the POSCAR file to be read.') # lattice properties
parser.add_argument('--hdf_file'   , '-hf', required = True, type = str, help='Set the hdf5 file to be read.'  ) # phonon properties of the material

args = parser.parse_args()

# getting start time

start_time = datetime.now()

print(' ---------- o ----------- o ------------- o ------------')
print("Year: {:<4d}, Month: {:<2d}, Day: {:<2d}".format(start_time.year, start_time.month, start_time.day))
print("Start at: {:<2d} h {:<2d} min {:<2d} s".format(start_time.hour, start_time.minute, start_time.second))	
print(' ---------- o ----------- o ------------- o ------------')

# initialising grid

geo = Geometry(args)

# geo.mesh.show()

# opening file

phonons = Phonon(args)
phonons.load_properties()

pop = Population(args, geo, phonons)

pop.plot_population()