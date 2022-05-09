# this file is a first try to automate several executions in one shot.

import subprocess

param_file = 'parameters_linux.txt'

dim_list = ['8e3', '9e3', '1e4', '2e4', '3e4', '4e4', '5e4', '6e4', '7e4', '8e4', '9e4', '1e5', '2e5', '3e5', '4e5']
sl_list = [ '10',  '10',  '15',  '15',  '15',  '15',  '15',  '15',  '15',  '15',  '15',  '20',  '20',  '20',  '20']

nruns = len(dim_list)

count = 0
while count < nruns:

    f = open(param_file, 'r')
    text = f.read().split()
    f.close()

    text[text.index('--dimensions')+1] = dim_list[count]

    save_folder = 'serial_' + dim_list[count] 
    text[text.index('--results_folder')+1] = save_folder

    text[text.index('--subvolumes')+2] = sl_list[count]

    params = ''
    for p in text:
        params = params + p + ' '
    
    command = 'python main_program.py '+ params
    
    sp = subprocess.call(command, shell = True)

    if sp == 0:
        count += 1
