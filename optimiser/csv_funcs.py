import os, re, subprocess, sys
from datetime import timedelta
from multiprocessing import Pool
import pandas as pd
from time import sleep


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

def time_for_cluster(t, margin = '0-00:10:00', return_timedelta = False):
    '''Add an extra margin to maximum simulation time to
       avoid the cluster finishing the job before the simulation ends.
       The standard is '''

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
        return '{:d}-{:2d}:{:2d}:{:2d}'.format(c[0], c[1], c[2], c[3])

def exec_sim(sim_params, mode, exec_params):
    if mode == 'cluster':

        folder = generate_results_folder(sim_params['results_folder'])  # creates results folder
        os.chdir(folder)                                                # go to folder

        exec_params['time'] = time_for_cluster(sim_params['max_sim_time'])
        
        # generates slurm script
        generate_script(job_name         = exec_params['job_name'],
                        nanokappa_path   = exec_params['nanokappa_path'],
                        nanokappa_folder = exec_params['nanokappa_folder'],
                        parameter_path   = exec_params['parameter_path'],
                        parameter_file   = exec_params['parameter_file'],
                        conda_env        = exec_params['conda_env'],
                        time             = exec_params['time'],
                        partition        = exec_params['partition'],
                        n_of_nodes       = exec_params['n_of_nodes'],
                        n_of_tasks       = exec_params['n_of_tasks'],
                        mail_type        = exec_params['mail_type'],
                        mail_user        = exec_params['mail_user'])
        
        # override sim parameters for cluster execution
        sim_params['results_location'] = ['local']
        sim_params['results_folder']   = ['results']
        sim_params['material_folder']  = ['']

        generate_parameter_file(sim_params) # generate parameters file

        subprocess.run('sbatch script') # run script

        os.chdir('..') # go back to main result folder

    elif mode == 'local':

        wait_time = time_for_cluster(sim_params['max_sim_time'], return_timedelta = True, margin = '0-00:01:00')
        
        cmd = 'conda run -n nanokappa python {}/{}/nanokappa.py'.format(exec_params['nanokappa_path'], exec_params['nanokappa_folder'])
        for k in sim_params.keys():
            cmd += ' --{} {}'.format(k, sim_params[k])

        if sys.platform in ['linux', 'linux2']:
            sp = subprocess.Popen('gnome-terminal --wait -- ' + cmd, shell = True)
            sp.wait()
        elif sys.platform == 'darwin':
            sp = subprocess.Popen("osascript -e 'tell app \"Terminal\" to do script \"{}\"' ".format(cmd), shell = True)
            sp.wait()
        elif sys.platform == 'win32':
            sp = subprocess.Popen('wt '+cmd, shell = True)
            sleep(wait_time.total_seconds())

    else:
        raise Exception('Wrong mode. Please choose from "cluster" or "local".')

def generate_script(job_name,
                    nanokappa_path,
                    nanokappa_folder,
                    material_path,
                    material_files,
                    geometry_path,
                    geometry_file,
                    parameter_path,
                    parameter_file,
                    conda_env,
                    time = '1-00:00:00',
                    partition = 'std',
                    n_of_nodes = 1,
                    n_of_tasks = 1,
                    mail_type = 'ALL',
                    mail_user = None):

    '''THIS HAS BEEN TESTED ON UNIVERSITÉ DE LORRAINE'S CLUSTER EXPLOR ONLY.'''

    s = '#!/bin/bash\n' + \
        '#SBATCH -p {:s}\n'.format(partition) + \
        '## Number of nodes\n' + \
        '#SBATCH -N {:d}}\n'.format(n_of_nodes) + \
        '#SBATCH -J {:s}\n'.format(job_name) + \
        '#SBATCH -n {:d}}\n'.format(n_of_tasks) + \
        '#SBATCH -t {:s}\n'.format(time)
    
    if mail_user is not None:
        s += '#SBATCH --mail-type={:s}\n'.format(mail_type) + \
        '#SBATCH --mail-user={:s}\n'.format(mail_user)
    
    s+= 'module purge\n' + \
        'module use -a /opt/modulefiles/shared/mcs_mod/\n' + \
        'module load mcs_mod/softwares/anaconda3/2022.05 # opening the most recent anaconda\n' + \
        'source $HOME_ANACONDA/anaconda.rc\n' + \
        'WORKDIR=$SCRATCHDIR/job.$SLURM_JOB_ID.$USER\n' + \
        'mkdir -p $WORKDIR\n' + \
        'cp -rf {:s} $WORKDIR\n'.format(nanokappa_path+'/'+nanokappa_folder) + \
        'cp -rf {:s} $WORKDIR\n'.format(geometry_path +'/'+geometry_file) + \
        'cp -rf {:s} $WORKDIR\n'.format(material_path +'/'+material_files[0]) + \
        'cp -rf {:s} $WORKDIR\n'.format(material_path +'/'+material_files[1]) + \
        'cp -rf {:s} $WORKDIR\n'.format(parameter_path+'/'+parameter_file) + \
        'cd $WORKDIR\n' + \
        'conda activate {:s}\n'.format(conda_env) + \
        'srun python {:s}/nanokappa.py -ff {:s}\n'.format(nanokappa_folder, parameter_file) + \
        'cp -rf {:s}/* $SLURM_SUBMIT_DIR/.'
    
    with open('script', 'w') as f:
        f.write(s)
        f.close()

def generate_parameter_file(params):
    '''Params is a dict with keys with the complete parameter name and
     its value(s). Accetps string, int, float, list or tuple as value types.
     
     Example:
     {'particles':1e7,
      'bound_pos':['relative', 0, 0.5, 0.5, 1, 0.5, 0.5],
      'bound_cond':'T T R',
      ...}
      
      '''
    
    s = ''
    for k in params.keys():
        if type(params[k]) in [list, tuple]:
            values = ''
            for v in params[k]:
                values = values+'{} '.format(v)
        else:
            values = '{}'.format(params[k])
        s += '--{:s} {:s}\n'.format(k, values)
    
    with open('parameters.txt', 'w') as f:
        f.write(s)
        f.close()