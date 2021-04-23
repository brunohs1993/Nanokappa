from classes.Geometry import Geometry
from classes.Phonon import Phonon
from classes.Population import Population

import os
import argparse
import sys

# Imports classes used and sets the work folder

# changing the working directory to be the same as the mcphonon folder

def initialise_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file'   , '-ff', default  = ''                , type = str, help='Import arguments from file.'     ) # arguments file name to be imported

    parser.add_argument('--geometry'    , '-g' , default = ['cuboid']         , type = str  , nargs = 1  , help='Geometry of the domain. Standard shapes are cuboid, cylinder, cone and capsule')
    parser.add_argument('--dimensions'  , '-d' , default = [20e3, 1e3, 1e3]   , type = float, nargs = 3  , help='Dimensions in angstroms as asked by trimesh.creation primitives. 3 for box, 2 for others. Radius first.')
    parser.add_argument('--scale'       , '-s' , default = [1, 1, 1]          , type = float, nargs = 3  , help='Scaling factors (x, y, z) to be applied to given geometry.')
    parser.add_argument('--rotation'    , '-r' , default = [0, 0, 0]          , type = float, nargs = 3  , help='Euler angles in degrees to be applied to given geometry (see scipy.rotation.from_euler).')
    parser.add_argument('--rot_order'   , '-ro', default = ['xyz']            , type = str  , nargs = 1  , help='Order of rotation to be applied to given geometry (see scipy.rotation.from_euler).')
    parser.add_argument('--particles'   , '-p' , default = [1]                , type = float, nargs = 1  , help='Number of particles per mode, per slice.')
    parser.add_argument('--part_dist'   , '-pd', default = ['random_domain']  , type = str  , nargs = 1  , help='How to distribute particles. random/center _ domain/slice')
    parser.add_argument('--empty_slices', '-es', default = []                 , type = int  , nargs = '*', help='Slices indexesto keep empty at initialisation.')

    parser.add_argument('--timestep'    , '-ts', default = [1]                , type = float, nargs = 1  , help='Timestep size in picoseconds')
    parser.add_argument('--iterations'  , '-i' , default = [10000]            , type = int  , nargs = 1  , help='Number of timesteps (iterations) to be run')
    parser.add_argument('--slices'      , '-sl', default = [10, 0]            , type = int  , nargs = 2  , help='Number of slices and slicing axis (x = 0, y = 1, z = 2)')
    parser.add_argument('--temperatures', '-t' , default = [310, 290]         , type = float, nargs = 2  , help='Set first and reservoirs temperatures to be imposed.') 
    parser.add_argument('--temp_dist'   ,'-td' , default = ['constant_cold']  , type = str  , nargs = '*', help='Set how to distribute initial temperatures.')
    parser.add_argument('--bound_cond'  ,'-bc' , default = ['periodic']       , type = str  , nargs = 1  , help='Set behaviour of the other faces. It can be "periodic", ...')

    parser.add_argument('--rt_plot'     ,'-rp' , default = []                 , type = str  , nargs = '*', help='Set which property you want to see in the real time plot during simulation. Choose between T, omega, e, n and None (random colors).')
    parser.add_argument('--fig_plot'    ,'-fp' , default = ['T', 'omega', 'e'], type = str  , nargs = '*', help='Save figures with properties at the end. Standard is T, omega and energy.')
    parser.add_argument('--colormap'    ,'-cm' , default = ['viridis']        , type = str  , nargs = 1  , help='Set matplotlib colormap to be used on all plots. Standard is viridis.')

    # THOUGHT ABOUT GIVING NUMBERS FOR IMPOSED TEMPERATURE, 'ISO' FOR ISOLATED AND 'PER' FOR PERIODIC. STUDY WICH TYPES OF BOUNDARY CONDITIONS TO USE. NEED TO INDICATE FACES. THIS COULD GIVE FLEXIBILITY IF LOADING A CUSTOM GEOMETRY.
    # parser.add_argument('--bound_cond' , '-bc', default = [(0,290),(5, 300)], type = int  , nargs = '*', help='Set boundary conditions') 

    parser.add_argument('--poscar_file'     , '-pf', required = True   , type = str, help='Set the POSCAR file to be read.' ) # lattice properties
    parser.add_argument('--hdf_file'        , '-hf', required = True   , type = str, help='Set the hdf5 file to be read.'   ) # phonon properties of the material
    parser.add_argument('--results_folder'  , '-rf', default  = ''     , type = str, help='Set the results folder name.'    ) # 
    parser.add_argument('--results_location', '-rl', default  = 'local', type = str, help='Set the results folder location.') # 


    return parser



def read_args():

    # if a file is specified
    if ('-ff' in sys.argv) or ('--from_file' in sys.argv):
        
        # set filename
        if '-ff' in sys.argv:
            filename = sys.argv[sys.argv.index('-ff') +1]
        elif '--from_file' in sys.argv:
            filename = sys.argv[sys.argv.index('--from_file') +1]

        parser = initialise_parser()
        
        f = open(filename, 'r')
        f.seek(0)
        
        # read arguments from file
        args = parser.parse_args( f.read().split() )
        
        f.close()

        args.from_file = filename

    # else, read from command line
    else:
        parser = initialise_parser()
        args = parser.parse_args()

    return args

def generate_results_folder(args):

    # get results location
    loc = args.results_location

    if loc == 'local':
        pass                                    # stay in the same folder
    elif loc == 'main':
        file_path = os.path.realpath(__file__)  # get main_program.py path
        file_dir = os.path.dirname(file_path)   # get main_program.py folder
        args.results_location = file_dir
    else:
        os.chdir(loc)                           # otherwise, change do the specified path
    
    # get results folder name
    folder = args.results_folder

    if folder != '':    # if a folder name is specified

        folders = os.listdir(args.results_location) # get folders in working directory
    
        valid_folders = [i for i in folders if i.startswith(folder+'_')] # get folders that start with the same name

        if len(valid_folders) == 0: # if there is no folder with desired name, create the first (zeroth) one
            folder = folder+'_0'
        else:                       # if there is, create the next one
            numbers = [int(i[ len(folder)+1: ]) for i in valid_folders ]
            
            folder = folder + '_{:d}'.format(max(numbers)+1)

        args.results_folder = args.results_location + '/' + folder

        os.mkdir(args.results_folder)    # create folder

        args.results_folder += '/'
    
    elif folder == '':  # if not specified
        args.results_folder = args.results_location + '/'

    return args