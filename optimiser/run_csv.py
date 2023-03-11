import argparse
from multiprocessing import Pool
from functools import partial
from csv_funcs import *
from math import ceil

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
    chunksize = exec_params['ntasks']
    mem_lim   = exec_params['mem']
    n_jobs = int(ceil( N_sims/chunksize ))

    chunks = generate_chunks(exec_params, copy.copy(params_db), chunksize) # generate a csv for each chunk

    for i, chunk in enumerate(chunks):
        generate_script(exec_params, params_db, chunk, i)
        # subprocess.run('sbatch script_{:d}'.format(i), shell = True)

    params_db = adjust_paths(params_db)

    for i in range(len(params_db.index)):
        generate_parameter_file(params_db.iloc[i].drop('memory'), case_index = i)
    
    for i, chunk in enumerate(chunks):
        subprocess.run('sbatch script_{:d}'.format(i), shell = True)

elif mode == 'local':
    func = partial(exec_sim, mode = mode, exec_params = exec_params)
    param_list = [dict(params_db.loc[i, :]) for i in params_db.index]

    if __name__ == '__main__':
        with Pool(processes = exec_params['processes']) as pool:
            pool.map(func, param_list) # run cases