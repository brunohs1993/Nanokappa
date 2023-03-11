import os
import argparse
import sys

# NOTES:
# I propose to remove from user access the following parameters:
# --energy_norm --> let the standard be "mean" and leave "fixed" only for debugging. More stable and lower uncertainty for the same result.
# --offset      --> 1e-3 works fine and there is no need to change.
# --reservoir_gen --> one-to-one should be used only for debugging and fixed-rate should be the norm.
# --bound_facets and --connect_facets --> facets are unstable and can change depending on how the mesh is processed. Positions do not depend on the mesh,
# and the facets are adjusted accordingly.
# --particle_dist --> another parameters that can be left only for debugging, with random_subvol as standard.
# --subvol_material --> this will be probably changed and will never be used;
# 


def initialise_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file'      , '-ff', default = '',
                        type = str  , nargs = 1   , help    = 'Import arguments from file.'     ) # arguments file name to be imported

    parser.add_argument('--geometry'       , '-g' , default = ['cuboid'],
                        type = str  , nargs = 1   , help    = 'Geometry of the domain. Standard shapes are cuboid, cylinder, cone and capsule')
    parser.add_argument('--dimensions'     , '-d' , default = [10e3, 1e3, 1e3],
                        type = float, nargs = '*' , help    = 'Dimensions in angstroms as asked by trimesh.creation primitives. 3 for box, 2 for others. Radius first.')
    parser.add_argument('--scale'          , '-s' , default = [1, 1, 1],
                        type = float, nargs = 3   , help    = 'Scaling factors (x, y, z) to be applied to given geometry.')
    parser.add_argument('--geo_rotation'   , '-gr' , default = [0, 0, 0, 'xyz'],
                                      nargs = '*'  , help    = 'Euler angles in degrees to be applied to given geometry (see scipy.rotation.from_euler) and the order to ' +
                                                              'be applied (see scipy.rotation.from_euler).')
    parser.add_argument('--mat_rotation'   , '-mr', default = [0, 0, 0, 'xyz'],
                                      nargs = '*' , help    = 'Material index, Euler angles in degrees to be applied to given material and ' +
                                                              'the order to be applied (see scipy.rotation.from_euler).')
    parser.add_argument('--particles'      , '-p' , default = ['pmps', 1],
                                      nargs = 2   , help    = 'Number of particles. First argument is a string: "total" for total number, "pmps" for number per mode, per ' +
                                                              'subvolume, "pv" for particles per cubic angstrom. Second is the number.')
    parser.add_argument('--part_dist'      , '-pd', default = ['random_subvol'],
                        type = str  , nargs = 1   , help    = 'How to distribute particles. It can be used any combination of random/center _ domain/subvol, or input an external file' +\
                                                              'of particle data. The file should have the same structure as particle_data.txt given in the results.')
    parser.add_argument('--empty_subvols'  , '-es', default = [],
                        type = int  , nargs = '*' , help    = 'Subvolumes indexes to keep empty at initialisation.')
    parser.add_argument('--subvol_material', '-sm', default = [],
                        type = int  , nargs = '*' , help    = 'Material index of each subvolume, according to the order given at -pf and -hf.')

    parser.add_argument('--timestep'       , '-ts', default = [1],
                        type = float, nargs = 1   , help    = 'Timestep size in picoseconds')
    parser.add_argument('--iterations'     , '-i' , default = [10000],
                        type = int  , nargs = 1   , help    = 'Number of timesteps (iterations) to be run')
    parser.add_argument('--max_sim_time'   , '-mt', default = ['1-00:00:00'],
                        type = str  , nargs = 1   , help    = 'Maximum simulation time. If the iterations are not done when -mt is reached, simulation stops and final data is saved. ' +
                                                              ' Declared as D-HH:MM:SS. Useful to avoid losing data in cluster simulations.')
    parser.add_argument('--subvolumes'     , '-sv', default = [],
                                      nargs = '*' , help    = 'Type of subvolumes, number of subvolumes and slicing axis when the case (x = 0, y = 1, z = 2). ' +
                                                              'Accepts "slice", "grid" and "voronoi" as subvolume types.')
    parser.add_argument('--reference_temp' , '-rt', default = [0],
                        type = float, nargs = 1   , help    = 'Set reference temperature to be considered in the system, in Kelvin.') 
    parser.add_argument('--temp_dist'      , '-td', default = ['cold'], choices = ['cold', 'hot', 'linear', 'mean', 'random', 'custom'],
                        type = str  , nargs = '*' , help    = 'Set how to distribute initial temperatures.')
    parser.add_argument('--subvol_temp'    , '-st', default = [],
                        type = float, nargs = '*' , help    = 'Set subvolumes temperatures when custom profile is selected.')
    parser.add_argument('--bound_cond'     , '-bc', default = ['T', 'T', 'P'], choices = ['T', 'P', 'F', 'R'],
                        type = str  , nargs = '*' , help    = 'Set boundary conditions to each specific facet. Choose between "T" for temperature, "F" for flux, '+
                                                              '"R" for roughness or "P" for periodic. The respective values need to be set in --bound_values '+
                                                              '(not for periodic boundary condition).')
    parser.add_argument('--bound_facets'  , '-bf' , default = [],
                        type = int  , nargs = '*' , help    = 'Set the FACETS on which to apply the specific boundary conditions. Nargs depends on what was specified on --bound_cond. '+
                                                             'If --bound_cond/--bound_values has more values than --bound_facets, the last boundary condition will be applied to all non-specified facets.')
    parser.add_argument('--bound_pos'     , '-bp'    , default = [],
                                      nargs = '*' , help    = 'Set the POSITIONS from which to find the closest facet to apply the specific boundary conditions. Nargs depends on what was specified on --bound_cond.' + 
                                                             'First value is a keyword "relative" - considers all points in the mesh between 0 and 1 - or "absolute" - direct positions. Set points as kw x1 y1 z1 x2 y2 z2 etc.' +
                                                             'If --bound_cond/--bound_values has more values than --bound_pos, the last boundary condition will be applied to all non-specified facets.')
    parser.add_argument('--bound_values'  , '-bv' , default = [],
                        type = float, nargs = '*' , help    = 'Set boundary conditions values to be imposed (temperature [K], flux [W/m^2] or roughness [angstrom]).')
    parser.add_argument('--connect_facets', '-cf' , default = [],
                        type = int  , nargs = '*' , help    = 'Set the facets that are connected to apply periodicity. Faces are connected in pairs, or 0-1, 2-3, and so on.')
    parser.add_argument('--connect_pos'   , '-cp' , default = [],
                                      nargs = '*' , help    = 'Set the POSITIONS from which to find the closest facet to apply the connections between facets. Nargs depends on what was specified on --bound_cond.' + 
                                                             'First value is a keyword "relative" - considers all points in the mesh between 0 and 1 - or "absolute" - direct positions. Set points as kw x1 y1 z1 x2 y2 z2 etc.' +
                                                             'The facets are connected in pairs, the same way as declared on --connect_facets.')
    parser.add_argument('--reservoir_gen' , '-gn' , default = ['fixed_rate'], choices = ['fixed_rate', 'one_to_one', 'constant'],
                        type = str  , nargs = '*' , help    = 'Set the type of generation of particles in the reservoir. "fixed_rate" means the generation is independent from the particles leaving the domain. '+
                                                              '"one_to_one" means that a particle will be generated only when a particle leaves the domain (one leaves, one enters).')
    parser.add_argument('--offset'        , '-os' , default = [1e-3],
                        type = float, nargs = 1   , help    = 'The offset margin from facets to avoid problems with Trimesh collision detection. Default is 2*trimesh.tol.merge = 2e-8.')

    parser.add_argument('--path_points'  , '-pp'  , default = [],
                                      nargs = '*' , help    = 'Set the points where the path to calculate kappa should go through. Declared the same way as --bound_pos')

    parser.add_argument('--energy_normal' , '-en' , default = ['fixed'],
                        type = str  , nargs = 1   , help    = 'Set the energy normalisation in subvolume. "fixed" is divided by the expected number of particles in the subvolume (standard).'+
                                                             ' "mean" is the aritmetic mean.')
                        
    parser.add_argument('--rt_plot'       , '-rp' , default = [],
                        type = str  , nargs = '*' , help    = 'Set which property you want to see in the real time plot during simulation. Choose between T, omega, e, n and None (random colors).')
    parser.add_argument('--fig_plot'      , '-fp' , default = ['T', 'omega', 'e'],
                        type = str  , nargs = '*' , help    = 'Save figures with properties distributions. Standard is T, omega and energy.')
    parser.add_argument('--colormap'      , '-cm' , default = ['viridis'],
                        type = str  , nargs = 1   , help    = 'Set matplotlib colormap to be used on all plots. Standard is viridis.')
    parser.add_argument('--theme'         , '-th' , default = ['light'], choices = ['light', 'dark'],
                        type = str  , nargs = 1   , help    = 'Set theme color for all plots.')
    parser.add_argument('--n_mean'        , '-nm' , default  = [100], 
                        type = int  , nargs = 1   , help    = 'The number of datapoints to consider when calculating mean and stdev values. Each datapoint = 10 iterations. Default is 100.')

    parser.add_argument('--conv_crit'     , '-cc' , default = [1e-2, 10],
                        type = float, nargs = 2   , help    = 'Value of convergence criteria and number of checks to keep it under criteria to consider convergence.')

    parser.add_argument('--output'        , '-op' , default = 'file',
                        type = str  , nargs = 1   , help    = 'Where to print the output. "file" to save it in outuput.txt. "screen" to print on terminal.')

    parser.add_argument('--mat_folder'      , '-mf', default  = ['']   ,  type = str, nargs = '*', help     = 'Set folder with material data.'  ) # lattice properties
    parser.add_argument('--poscar_file'     , '-pf', required = True   , type = str, nargs = '*', help     = 'Set the POSCAR file to be read.' ) # lattice properties
    parser.add_argument('--hdf_file'        , '-hf', required = True   , type = str, nargs = '*', help     = 'Set the hdf5 file to be read.'   ) # phonon properties of the material
    parser.add_argument('--pickled_mat'     , '-pm', default  = []     , type = int, nargs = '*', help     = 'Inform if any material can be loaded from pickled object.')
    parser.add_argument('--mat_names'       , '-mn', required = True   , type = str, nargs = '*', help     = 'Set the names of each material.' ) #

    parser.add_argument('--results_folder'  , '-rf', default  = []     , type = str,nargs = '*', help     = 'Set the results folder name.'    ) # 
    parser.add_argument('--results_location', '-rl', default  = 'local', type = str, help     = 'Set the results folder location.') # 
    
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
    
    # define folder separator symbol and adjust folder path according to opperating system
    loc, folder_sep = correct_folder_separator(loc)

    if loc == 'local':
        args.results_location = os.getcwd()     # stay in the same folder
    elif loc == 'main':
        file_path = os.path.realpath(__file__)  # get main_program.py path
        file_dir = os.path.dirname(file_path)   # get main_program.py folder
        args.results_location = file_dir
    else:
        if not os.path.isabs(loc):                  # if the path is not absolute
            loc = os.getcwd() + folder_sep + loc          # add the current work folder to the path
        if not os.path.isdir(loc):                  # if it does not exist
            os.mkdir(loc)                           # create
        args.results_location = loc

    # get results folder name
    if len(args.results_folder)>0:    # if a folder name is specified

        folder = args.results_folder[0]

        folder, folder_sep = correct_folder_separator(folder)

        subfolders = folder.split(folder_sep)

        if len(subfolders) > 1: # if there are subfolders
            args.results_location += folder_sep + folder_sep.join(subfolders[:-1]) # add subfolders to results_location path
            os.makedirs(args.results_location, exist_ok=True)                      # create subfolders if they don't exist
            folder = subfolders[-1]                                                # save new folder name

        folders = os.listdir(args.results_location) # get folders in working directory
    
        valid_folders = [i for i in folders if i.startswith(folder+'_')] # get folders that start with the same name

        if len(valid_folders) == 0: # if there is no folder with desired name, create the first (zeroth) one
            folder = folder+'_0'
        else:                       # if there is, create the next one
            numbers = [int(i[ len(folder)+1: ]) for i in valid_folders ]
            
            folder = folder + '_{:d}'.format(max(numbers)+1)

        args.results_folder = args.results_location + folder_sep + folder

        os.mkdir(args.results_folder)    # create folder

        args.results_folder += folder_sep
    
    else:  # if not specified
        args.results_folder = args.results_location + folder_sep

    return args

def correct_folder_separator(path):

    if sys.platform == 'win32':
        folder_sep = '\\'
        path = path.replace('/', folder_sep)
    elif sys.platform in ['linux', 'linux2', 'darwin']:
        folder_sep = '/'
        path = path.replace('\\', folder_sep)
    
    return path, folder_sep