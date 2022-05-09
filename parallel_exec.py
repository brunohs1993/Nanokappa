# this file is a first try to automate several executions in one shot.

import subprocess
from multiprocessing import Pool

def exec_sim(i):
    param_file = 'parameters_linux.txt'

    dim_list = ['8e3', '9e3', '1e4', '2e4', '3e4', '4e4', '5e4', '6e4', '7e4', '8e4', '9e4', '1e5', '2e5', '3e5', '4e5']
    sl_list = [ '10',  '10',  '15',  '15',  '15',  '15',  '15',  '15',  '15',  '15',  '15',  '20',  '20',  '20',  '20']

    f = open(param_file, 'r')
    text = f.read().split()
    f.close()

    # change commands
    text[text.index('--dimensions')+1] = dim_list[i]

    save_folder = 'periodic_' + dim_list[i] 
    text[text.index('--results_folder')+1] = save_folder

    text[text.index('--subvolumes')+2] = sl_list[i]

    params = ''
    for p in text:
        params = params + p + ' '
    
    command = 'python main_program.py '+ params

    sp = subprocess.Popen('gnome-terminal --wait -- ' + command, shell = True)
    sp.wait()

    if sp.returncode == 0:
        return 'finished case {}'.format(i)
    else:
        exec_sim(i)
    
if __name__ == '__main__':

    nruns = 15 # same length as dim_list
    ind = [i for i in range(nruns)] # getting iterable

    with Pool(processes = 3) as pool:
        pool.map(exec_sim, ind) # run cases in 3 cores (check for memory limitations)