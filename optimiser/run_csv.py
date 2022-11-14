import pandas as pd
import argparse
from multiprocessing import Pool
from functools import partial
from csv_funcs import *

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', required = True, type = str, nargs = 1, help = 'Mode of execution.', choices = ['local', 'cluster'])
parser.add_argument('--file', '-f', required = True, type = str, nargs = 1, help = 'CSV file with sim parameters.')

# get arguments
args = parser.parse_args()

csv_file = args.file[0]
mode     = args.mode[0]

# read simulations params
params_db = import_params(csv_file)

N_sims = len(params_db.index)

cluster_params = dict(mail_user  = 'bruno.hartmann-da-silva@univ-lorraine.fr',
                      job_name   = 'nanokappa',
                      time       = 'calculate',
                      partition  = 'std',
                      n_of_nodes = 1,
                      n_of_tasks = 1,
                      mail_type  = 'ALL')

local_params = dict(processes = 3,
                    nanokappa_path = 'd:/LEMTA/Code',
                    nanokappa_folder = 'Nanokappa')

if mode == 'cluster':
    for i in range(N_sims):
        params = dict(params_db.loc[i, :])
        params['results_folder'] = 'params_{:d}'.format(i)
        exec_sim(params, mode, cluster_params)
    
elif mode == 'local':
    func = partial(exec_sim, mode = mode, exec_params = local_params)
    param_list = [dict(params_db.loc[i, :]) for i in params_db.index]

    if __name__ == '__main__':
        with Pool(processes = local_params['processes']) as pool:
            pool.map(func, param_list) # run cases in 3 cores (check for memory limitations)
