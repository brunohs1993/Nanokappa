import os
import argparse
import sys
import copy

def initialise_parser(debug_flag):

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
    parser.add_argument('--mat_rotation'   , '-mr', default = [],
                                      nargs = '*' , help    = 'Material index, Euler angles in degrees to be applied to given material and ' +
                                                              'the order to be applied (see scipy.rotation.from_euler).')
    parser.add_argument('--isotope_scat'   , '-is', default = [],
                        type = int,   nargs = '*' , help    = 'Which materials need to consider mass scattering. Default is none.')
    
    parser.add_argument('--particles'      , '-p' , default = ['pmps', 1],
                                      nargs = 2   , help    = 'Number of particles. First argument is a string: "total" for total number, "pmps" for number per mode, per ' +
                                                              'subvolume, "pv" for particles per cubic angstrom. Second is the number.')
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
    
    parser.add_argument('--temp_dist'      , '-td', default = ['cold'], choices = ['cold', 'hot', 'linear', 'mean', 'random', 'custom'],
                        type = str  , nargs = '*' , help    = 'Set how to distribute initial temperatures.')
    parser.add_argument('--temp_interp'    , '-ti', default = ['nearest'], choices = ['nearest', 'linear', 'radial'],
                        type = str  , nargs = 1   , help    = 'How to interpolate temperatures for particles located between subvolumes. Choose among "nearest", "linear" and "radial". ' +
                                                              'The "linear" option only works with slices and defaults to "radial" when used with other types of subvolumes.')
    parser.add_argument('--subvol_temp'    , '-st', default = [],
                        type = float, nargs = '*' , help    = 'Set subvolumes temperatures when custom profile is selected.')
    parser.add_argument('--bound_cond'     , '-bc', default = [], choices = ['T', 'P', 'R'],
                        type = str  , nargs = '*' , help    = 'Set boundary conditions to each specific facet. Choose between "T" for temperature,'+
                                                              '"R" for roughness or "P" for periodic. The respective values need to be set in --bound_values '+
                                                              '(not for periodic boundary condition).')
    parser.add_argument('--bound_pos'     , '-bp'    , default = [],
                                      nargs = '*' , help    = 'Set the positions from which to find the closest facet to apply the specific boundary conditions. Nargs depends on what was specified on --bound_cond.' + 
                                                             'First value is a keyword "relative" - considers all points in the mesh between 0 and 1 - or "absolute" - direct positions. Set points as kw x1 y1 z1 x2 y2 z2 etc.' +
                                                             'If --bound_cond/--bound_values has more values than --bound_pos, the last boundary condition will be applied to all non-specified facets.')
    parser.add_argument('--bound_values'  , '-bv' , default = [],
                        type = float, nargs = '*' , help    = 'Set boundary conditions values to be imposed (temperature [K] or roughness [angstrom]).')
    parser.add_argument('--connect_pos'   , '-cp' , default = [],
                                      nargs = '*' , help    = 'Set the POSITIONS from which to find the closest facet to apply the connections between facets. Nargs depends on what was specified on --bound_cond.' + 
                                                             'First value is a keyword "relative" - considers all points in the mesh between 0 and 1 - or "absolute" - direct positions. Set points as kw x1 y1 z1 x2 y2 z2 etc.' +
                                                             'The facets are connected in pairs, the same way as declared on --connect_facets.')
   
    parser.add_argument('--fig_plot'      , '-fp' , default = [],
                        type = str  , nargs = '*' , help    = 'Save figures with properties distributions. Standard is T, omega and energy.')
    parser.add_argument('--colormap'      , '-cm' , default = ['jet'],
                        type = str  , nargs = 1   , help    = 'Set matplotlib colormap to be used on all plots. Standard is jet.')
    parser.add_argument('--theme'         , '-th' , default = ['white'], choices = ['white', 'light', 'dark'],
                        type = str  , nargs = 1   , help    = 'Set theme color for all plots.')

    parser.add_argument('--conv_crit'     , '-cc' , default = [0, 1],
                        type = float, nargs = 2   , help    = 'Value of convergence criteria and number of checks to keep it under criteria to consider convergence.')

    parser.add_argument('--mat_folder'      , '-mf', default  = ['']   ,  type = str, nargs = '*', help     = 'Set folder with material data.'  ) # lattice properties
    parser.add_argument('--poscar_file'     , '-pf', required = True   , type = str, nargs = '*', help     = 'Set the POSCAR file to be read.' ) # lattice properties
    parser.add_argument('--hdf_file'        , '-hf', required = True   , type = str, nargs = '*', help     = 'Set the hdf5 file to be read.'   ) # phonon properties of the material

    parser.add_argument('--results_folder'  , '-rf', default  = []     , type = str,nargs = '*', help     = 'Set the results folder.'    ) # 
    

    ############## DEBUG OPTIONS ########################

    parser.add_argument('--part_dist'      , '-pd', default = ['random_subvol'],
                        type = str  , nargs = 1   , help    = [argparse.SUPPRESS, 'How to distribute particles. It can be used any combination of random/center _ domain/subvol, or input an external file ' +
                                                                                  'of particle data. The file should have the same structure as particle_data.txt given in the results.'][int(debug_flag)])
    
    parser.add_argument('--reference_temp' , '-rt', default = ['local'],
                                      nargs = 1   , help    = [argparse.SUPPRESS, 'Set reference temperature to be considered in the system, in Kelvin. Also accepts "local", so deltas are calculated in relation to local temperature.'][int(debug_flag)]) 
    
    parser.add_argument('--output'        , '-op' , default = 'file',
                        type = str  , nargs = 1   , help    = [argparse.SUPPRESS, 'Where to print the output. "file" to save it in outuput.txt. "screen" to print on terminal.'][int(debug_flag)])
    
    return parser

def read_args(debug_flag):

    # if a file is specified
    if ('-ff' in sys.argv) or ('--from_file' in sys.argv):
        
        # set filename
        if '-ff' in sys.argv:
            filename = sys.argv[sys.argv.index('-ff') +1]
        elif '--from_file' in sys.argv:
            filename = sys.argv[sys.argv.index('--from_file') +1]

        parser = initialise_parser(debug_flag)
        
        f = open(filename, 'r')
        f.seek(0)
        
        # read arguments from file
        args = parser.parse_args( f.read().split() )
        
        f.close()

        args.from_file = filename

    # else, read from command line
    else:
        parser = initialise_parser(debug_flag)
        args = parser.parse_args()

    return args

def generate_results_folder(args):

    # get results location relative path
    if len(args.results_folder) == 0:
        args.results_folder = os.getcwd()
        return args
    else:
        loc = os.path.normpath(os.path.relpath(args.results_folder[0]))

        if not os.path.isabs(loc): # if it is not absolute, get from current working directory
            loc = os.path.join(os.getcwd(), loc)

        # create folder
        i = get_folder_index(loc)
        os.makedirs(f'{loc}_{i}', exist_ok=False)
        
        args.results_folder = f'{loc}_{i}'
    return args

def get_folder_index(loc):
    
    basename = os.path.basename(loc)
    dirname = os.path.dirname(loc)

    if not os.path.exists(dirname):
        return 0

    dirs = os.listdir(dirname)
    same = [int(d.split('_')[-1]) for d in dirs if basename in d]

    if len(same) == 0:
        return 0
    else:
        return max(same)+1
    





