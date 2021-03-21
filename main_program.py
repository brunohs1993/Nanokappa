
# Just a testing command ready to copy and paste on terminal:
# python main_program.py -hf Pb2_I4/kappa-m20206.hdf5 -pf Pb2_I4/POSCAR-unitcell

import os
from datetime import datetime
import numpy as np
from mcphonon import *
import argparse

# Setting arguments:

parser = argparse.ArgumentParser()

parser.add_argument('--geometry'    , '-g' , default = 'cuboid'           , type = str  , nargs = 1  , help='Geometry of the domain. Standard shapes are cuboid, cylinder, cone and capsule')
parser.add_argument('--dimensions'  , '-d' , default = [20e3, 1e3, 1e3]   , type = float, nargs = 3  , help='Dimensions in angstroms as asked by trimesh.creation primitives. 3 for box, 2 for others. Radius first.')
parser.add_argument('--scale'       , '-s' , default = [1, 1, 1]          , type = float, nargs = 3  , help='Scaling factors (x, y, z) to be applied to given geometry.')
parser.add_argument('--rotation'    , '-r' , default = [0, 0, 0]          , type = float, nargs = 3  , help='Euler angles in degrees to be applied to given geometry (see scipy.rotation.from_euler).')
parser.add_argument('--rot_order'   , '-ro', default = 'xyz'              , type = str  , nargs = 1  , help='Order of rotation to be applied to given geometry (see scipy.rotation.from_euler).')
parser.add_argument('--particles'   , '-p' , default = [1]                , type = float, nargs = 1  , help='Number of particles per mode, per slice.')
parser.add_argument('--timestep'    , '-ts', default = [1e-12]            , type = float, nargs = 1  , help='Timestep size in seconds')
parser.add_argument('--iterations'  , '-i' , default = [10000]            , type = int  , nargs = 1  , help='Number of timesteps (iterations) to be run')
parser.add_argument('--slices'      , '-sl', default = (10, 0)            , type = int  , nargs = 2  , help='Number of slices and slicing axis (x = 0, y = 1, z = 2)')
parser.add_argument('--temperatures', '-t' , default = [310, 290]         , type = float, nargs = 2  , help='Set first and last slice temperatures to be imposed.') 
parser.add_argument('--temp_dist'   ,'-td' , default = ['constant_cold']  , type = str  , nargs = 1  , help='Set how to distribute initial temperatures.')
parser.add_argument('--bound_cond'  ,'-bc' , default = 'periodic'         , type = str  , nargs = 1  , help='Set behaviour of the other faces. It can be "periodic", ...')

parser.add_argument('--rt_plot'     ,'-rp' , default = []                 , type = str  , nargs = '*', help='Set which property you want to see in the real time plot during simulation. Choose between T, omega, e, n and None (random colors).')
parser.add_argument('--fig_plot'    ,'-fp' , default = ['T', 'omega', 'e'], type = str  , nargs = '*', help='Save figures with properties at the end. Standard is T, omega and energy.')
parser.add_argument('--colormap'    ,'-cm' , default = 'viridis'          , type = str  , nargs = 1  , help='Set matplotlib colormap to be used on all plots. Standard is viridis.')

# THOUGHT ABOUT GIVING NUMBERS FOR IMPOSED TEMPERATURE, 'ISO' FOR ISOLATED AND 'PER' FOR PERIODIC. STUDY WICH TYPES OF BOUNDARY CONDITIONS TO USE. NEED TO INDICATE FACES. THIS COULD GIVE FLEXIBILITY IF LOADING A CUSTOM GEOMETRY.
# parser.add_argument('--bound_cond' , '-bc', default = [(0,290),(5, 300)], type = int  , nargs = '*', help='Set boundary conditions') 

parser.add_argument('--poscar_file', '-pf', required = True        , type = str, help='Set the POSCAR file to be read.') # lattice properties
parser.add_argument('--hdf_file'   , '-hf', required = True        , type = str, help='Set the hdf5 file to be read.'  ) # phonon properties of the material
parser.add_argument('--conv_file'  , '-cf', default  ='convergence', type = str, help='Set the convergence file name.' ) # convergence file name

args = parser.parse_args()

# saving arguments on file
folder = 'final_result'
if folder not in os.listdir():
    os.mkdir(folder)

args_filename = folder + '/arguments.txt'

f = open(args_filename, 'w')

for key in vars(args).keys():
    f.write( '{} = {} \n'.format(key, vars(args)[key]) )

f.close()

# getting start time

start_time = datetime.now()

print(' ---------- o ----------- o ------------- o ------------')
print("Year: {:<4d}, Month: {:<2d}, Day: {:<2d}".format(start_time.year, start_time.month, start_time.day))
print("Start at: {:<2d} h {:<2d} min {:<2d} s".format(start_time.hour, start_time.minute, start_time.second))	
print(' ---------- o ----------- o ------------- o ------------')

# initialising grid

geo = Geometry(args)

# opening file

phonons = Phonon(args)
phonons.load_properties()

pop = Population(args, geo, phonons)

# pop.plot_figures(geo, property_plot = ['T', 'n', 'omega', 'e'])

while pop.current_timestep < args.iterations[0]:
    
    pop.run_timestep(geo, phonons)

pop.write_final_state()

pop.f.close()

end_time = datetime.now()

total_time = end_time - start_time

print(' ---------- o ----------- o ------------- o ------------')
print("Start at: {:<2d} h {:<2d} min {:<2d} s".format(start_time.hour, start_time.minute, start_time.second))	
print("Finish at: {:<2d} h {:<2d} min {:<2d} s".format(end_time.hour, end_time.minute, end_time.second))

hours = total_time.seconds//3600
minutes = (total_time.seconds//60)%60

seconds = total_time.seconds - 3600*hours - 60*minutes

print("Total time: {:<2d} h {:<2d} min {:<2d} s".format(hours, minutes, seconds))
print(' ---------- o ----------- o ------------- o ------------')