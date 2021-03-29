from Geometry import *
from Phonon import *
from Population import *

import os

# Imports classes used and sets the work folder

# changing the working directory to be the same as the mcphonon folder

file_path = os.path.realpath(__file__)
file_dir = os.path.dirname(file_path)

os.chdir(file_dir)
    
def generate_results_folder(folder):
    folders       = os.lisrdir()
    
    valid_folders = [i for i in folders if i.startswith(folder)]

    if len(valid_folders) == 0: # if there is no folder with desired name, create the first (zeroth) one
        folder = folder+'_0'
    else:
        numbers = [int(i[ len(folder): ]) for i in valid_folders ]
        
        folder = folder + '_{:d}'.format(max(numbers)+1)

    os.mkdir(folder)

    return folder+'/'