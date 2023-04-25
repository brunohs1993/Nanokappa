import os, re, subprocess, sys
from datetime import timedelta
from multiprocessing import Pool
import pandas as pd
from time import sleep
import copy

def get_exec_params(file):
    with open(file, 'r') as f:
        a = f.readlines()               # read options file

    a = [i.splitlines()  for i in a] # remove break line characters
    a = [[j for j in i[0].split(' ') if j != ''] for i in a]   # split keys and values

    options = {}
    for i in range(len(a)):
        try:
            options[a[i][0]] = int(a[i][1])
        except:
            options[a[i][0]] = a[i][1]
    
    return options

def import_params(file):
    params = pd.read_csv(file)                       # read simulation parameters from csv data
    params.replace(float('nan'), '', inplace = True) # replace nan values with empty strings

    return params

def generate_results_folder(folder):
    '''Generate the result folder in place, with the
       name given as argument.'''

    folders = os.listdir() # get folders in working directory

    valid_folders = [i for i in folders if i.startswith(folder+'_')] # get folders that start with the same name

    if len(valid_folders) == 0: # if there is no folder with desired name, create the first (zeroth) one
        folder = folder+'_0'
    else:                       # if there is, create the next one
        numbers = [int(i[ len(folder)+1: ]) for i in valid_folders ]
        
        folder = folder + '_{:d}'.format(max(numbers)+1)

    os.mkdir(folder)    # create folder
    
    return folder

def generate_chunks(exec_params, sim_params, processes):
    '''Generate a list of indexes of cases for each group of simulations
       to be executed together.'''

    max_mem = int(exec_params['mem'].replace('G', ''))
    chunks = []
    while len(sim_params.index) > processes or sim_params['memory'].sum() > max_mem:
        i_max = sim_params['memory'].idxmax()
        chunk = [i_max]
        total_mem = sim_params['memory'].loc[i_max]
        
        sim_params.drop(i_max, axis = 0, inplace = True)
        
        while total_mem < max_mem and len(chunk) < processes:
            
            i_min = sim_params['memory'].idxmin()
            if total_mem + sim_params['memory'].loc[i_min] < max_mem:
                chunk.append(i_min)
                total_mem += sim_params['memory'].loc[i_min]
                sim_params.drop(i_min, axis = 0, inplace = True)
            else:
                break

            if len(chunk) < processes:
                
                i_max = sim_params['memory'].idxmax()
                
                if total_mem + sim_params['memory'].loc[i_max] < max_mem:
                    chunk.append(i_max)
                    total_mem += sim_params['memory'].loc[i_max]
                    sim_params.drop(i_max, axis = 0, inplace = True)
            
            
        chunks.append(chunk)
    
    chunks.append(list(sim_params.index))
    
    return chunks
        
def time_for_cluster(t, margin = '0-01:00:00', return_timedelta = False):
    '''Add an extra margin to maximum simulation time to
       avoid the cluster finishing the job before the simulation ends.
       The standard is 1 hour.'''

    # if t is not string, there is more than one simulation and the maximum has to be taken
    if type(t) != str: 
        cluster_time = 0
        all_t = copy.copy(t)
        del(t)
        for i, t in enumerate(all_t):
            t_list = re.split('-|:', t)

            t_list = [int(i) for i in t_list]

            new_time = (((t_list[0]*24)+t_list[1])*60 + t_list[2])*60 + t_list[3]
            if new_time > cluster_time:
                cluster_time = new_time
                t_max = t
        
        t = t_max
    
    print('--', t_max)
        
    t = re.split('-|:', t)
    m = re.split('-|:', margin)
    
    t = [int(i) for i in t]
    m = [int(i) for i in m]

    c = [t[i]+m[i] for i in range(4)]

    if c[3] >= 60:
        c[2] += c[3] // 60 # extra minutes
        c[3]  = c[3] %  60
    if c[2] >= 60:
        c[1] += c[2] // 60 # extra minutes
        c[2]  = c[2] %  60
    if c[1] >= 24:
        c[0] += c[1] // 24 # extra minutes
        c[1]  = c[1] %  24
    
    if return_timedelta:
        return timedelta(days   = c[0],
                        hours   = c[1],
                        minutes = c[2],
                        seconds = c[3])
    
    else:
        return '{:d}-{:s}:{:s}:{:s}'.format(c[0],
                                            str(c[1]).zfill(2),
                                            str(c[2]).zfill(2),
                                            str(c[3]).zfill(2))
    
def exec_sim(sim_params_csv, mode, exec_params):
    if mode == 'cluster':
        
        sim_params = import_params(sim_params_csv)

        exec_params['time'] = time_for_cluster(sim_params['max_sim_time']) # add margin for cluster time
        
        # generates slurm script
        generate_script(sim_params_csv,
                        exec_params)
        
        # override sim parameters for cluster execution
        sim_params['results_location'] = 'local'
        sim_params['mat_folder']  = ['mat_{:d}'.format(i) for i in range(len(exec_params['material_folder']))]
        if sim_params['geometry'] is not None:
            sim_params['geometry'] = os.path.basename(sim_params['geometry'])

        # generate_parameter_file(sim_params) # generate parameters file

        subprocess.run('sbatch script', shell = True) # run script

        os.chdir('..') # go back to main result folder

    elif mode == 'local':

        sim_params = import_params(sim_params_csv)

        cmd = 'conda run -n nanokappa python {}/nanokappa.py'.format(exec_params['nanokappa_folder'])
        for k in sim_params.keys():
            cmd += ' --{} {}'.format(k, sim_params[k])

        if sys.platform in ['linux', 'linux2']:
            sp = subprocess.Popen('gnome-terminal --wait -- ' + cmd, shell = True)
            sp.wait()
        elif sys.platform == 'darwin':
            sp = subprocess.Popen("osascript -e 'tell app \"Terminal\" to do script \"{}\"' ".format(cmd), shell = True)
            sp.wait()
        elif sys.platform == 'win32':
            wait_time = time_for_cluster(sim_params['max_sim_time'], return_timedelta = True, margin = '0-00:01:00')
            sp = subprocess.Popen('wt '+cmd, shell = True)
            sleep(wait_time.total_seconds())

    else:
        raise Exception('Wrong mode. Please choose from "cluster" or "local".')

def generate_script(cluster_options,
                    sim_params,
                    chunk,
                    chunk_index):

    '''THIS HAS BEEN TESTED ON UNIVERSITE DE LORRAINE'S CLUSTER EXPLOR ONLY.
    
    - cluster_options is a dict with all options to be added at the beginning of the script;
    - params_csv is the name of the csv file to copy and execute'''
    
    s = '#!/bin/bash\n'
    for option in cluster_options.keys():
        if option not in ['nanokappa_folder', 'conda_env']:
            if option == 'ntasks':
                s += '#SBATCH --{:s} {:s}\n'.format(option, str(len(chunk)))
            else:
                s += '#SBATCH --{:s} {:s}\n'.format(option, str(cluster_options[option]))
    
    s+= '#SBATCH --time {:s}\n'.format(time_for_cluster(sim_params.loc[chunk, 'max_sim_time']))
    
    # initialise conda, create workdir and copy nanokappa
    s+= 'module purge\n' + \
        'module use -a /opt/modulefiles/shared/mcs_mod/\n' + \
        'module load mcs_mod/softwares/anaconda3/2022.05\n' + \
        'source $HOME_ANACONDA/anaconda.rc\n' + \
        'WORKDIR=$SCRATCHDIR/job.$SLURM_JOB_ID.$USER\n' + \
        'mkdir -p $WORKDIR\n' + \
        'cp -rf {:s} $WORKDIR\n'.format(cluster_options['nanokappa_folder'])
        
    for i in chunk:
        s += 'cp -rf parameters_{:d}.txt $WORKDIR\n'.format(i)

    # copy materials files
    mat_db = sim_params[['mat_folder', 'hdf_file', 'poscar_file']].loc[chunk].drop_duplicates()
    for _, case in mat_db.iterrows():
        mat_folder = case['mat_folder'].split(' ')
        mat_folder = [i for i in mat_folder if i != '']

        hdf = case['hdf_file'].split(' ')
        hdf = [i for i in hdf if i != '']

        poscar = case['poscar_file'].split(' ')
        poscar = [i for i in poscar if i != '']

        mat_cmd_list = []
        
        for i, folder in enumerate(mat_folder):
            cmd_str = 'mkdir -p $WORKDIR/{:s}\n'.format(os.path.basename(folder)) + \
                      'cp -rf {:s} $WORKDIR/{:s}/\n'.format(folder +'/'+ hdf[i], os.path.basename(folder)) + \
                      'cp -rf {:s} $WORKDIR/{:s}/\n'.format(folder +'/'+ poscar[i], os.path.basename(folder))
            if cmd_str not in mat_cmd_list:
                mat_cmd_list.append(cmd_str)
        
    for cmd in mat_cmd_list:
        s += cmd

    # copy geometry file if needed
    geo_db = sim_params[['geometry']].loc[chunk].drop_duplicates()
    for i, case in geo_db.iterrows():
        if case['geometry'] not in ['cuboid', 'cylinder', 'sphere', 'cone', 'capsule']:
            s+= 'cp -rf {:s} $WORKDIR\n'.format(case['geometry'])
    
    # copy parameters and run simulation
    s += 'cd $WORKDIR\n' + \
         'mkdir results\n' +\
         'conda activate {:s}\n'.format(cluster_options['conda_env'])
    # s += 'pid=()\n' +\
    #      'chunk=('
    # for i in chunk:
    #     s += ' {:d}'.format(i)
    # s += ')\n' +\
    #      'for i in ${chunk[@]}; do\n' +\
    #      'python Nanokappa/nanokappa.py -ff parameters_$i.txt > results/output_$i.txt &\n' +\
    #      'ipid=$!\n' +\
    #      'pid=(${pid[@]} $ipid)\n' +\
    #      'done\n' +\
    #      'wait ${pid[@]}\n'

    for i in chunk[:-1]:
        s += 'python {:s}/nanokappa.py -ff parameters_{:d}.txt > results/output_{:d}.txt & '.format(os.path.basename(cluster_options['nanokappa_folder']), i, i)
    s += 'python {:s}/nanokappa.py -ff parameters_{:d}.txt > results/output_{:d}.txt\n'.format(os.path.basename(cluster_options['nanokappa_folder']), chunk[-1], chunk[-1])
    
    s += 'wait\n'
    s += 'cp -rf results*/* $SLURM_SUBMIT_DIR/.\n'

    # save script
    with open('script_{:d}'.format(chunk_index), 'w') as f:
        f.write(s)
        f.close()

def generate_parameter_file(sim_params, return_string = False, case_index = 0):
    '''sim_params is a dict with the complete simulation parameter names as keys and
     its value(s). Accetps string, int, float, list or tuple as value types.
     
     Example:
     {'particles':1e7,
      'bound_pos':['relative', 0, 0.5, 0.5, 1, 0.5, 0.5],
      'bound_cond':'T T R',
      ...}
      
      '''
    if return_string:
        trail = ' '
    else:
        trail = '\n'
    
    s = ''
    for k in sim_params.keys():
        if type(sim_params[k]) in [list, tuple]:
            values = ''
            for v in sim_params[k]:
                values = values+'{} '.format(v)
        else:
            values = '{}'.format(sim_params[k])
        s += '--{:s} {:s}'.format(k, values)+trail
    
    if return_string:
        return s
    
    with open('parameters_{:d}.txt'.format(case_index), 'w') as f:
        f.write(s)
        f.close()

def adjust_paths(sim_params):
    
    sim_params['results_location'] = 'results'

    for i in sim_params.index:
        mat_folder = sim_params.loc[i, 'mat_folder'].split(' ')
        mat_folder = [os.path.basename(i) for i in mat_folder if i != '']
        mat_str = ''
        for s in mat_folder:
            mat_str += '{:s} '.format(s)
        sim_params.loc[i, 'mat_folder'] = mat_str

        if sim_params.loc[i, 'geometry'] is not None:
            sim_params.loc[i, 'geometry'] = os.path.basename(sim_params.loc[i, 'geometry'])
    
    return sim_params