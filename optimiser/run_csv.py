import argparse
from multiprocessing import Pool
from functools import partial
from csv_funcs import *

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode'      , '-m', required = True, type = str, nargs = 1, help = 'Mode of execution.', choices = ['local', 'cluster'])
parser.add_argument('--parameters', '-p', required = True, type = str, nargs = 1, help = 'CSV file with sim parameters.')
parser.add_argument('--options'   , '-o', required = True, type = str, nargs = 1, help = 'Text file with execution options.')

# get arguments
args = parser.parse_args()

csv_file = args.parameters[0]
mode     = args.mode[0]

# read simulations params
params_db = import_params(csv_file)

exec_params = get_exec_params(args.options[0])

N_sims = len(params_db.index)

if mode == 'cluster':
    for i in range(N_sims):
        params = dict(params_db.loc[i, :])
        params['results_folder'] = 'params_{:d}'.format(i)
        exec_sim(params, mode, exec_params)
    
elif mode == 'local':
    func = partial(exec_sim, mode = mode, exec_params = exec_params)
    param_list = [dict(params_db.loc[i, :]) for i in params_db.index]

    if __name__ == '__main__':
        with Pool(processes = exec_params['processes']) as pool:
            pool.map(func, param_list) # run cases