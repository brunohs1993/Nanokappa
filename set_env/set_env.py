import subprocess, sys, time, os
from threading import local

if sys.platform == 'win32':
    folder_sep = '\\'
elif sys.platform in ['linux', 'linux2', 'darwin']:
    folder_sep = '/'

main_path = os.path.realpath(__file__).replace('set_env{}set_env.py'.format(folder_sep), '') # path of Nanokappa main

env_name = input('Type the name of the environment to be created and press ENTER: ')

subprocess.run(f'conda create -n {env_name} python=3.9 --yes', shell = True)
subprocess.run(f'conda activate {env_name}', shell = True)
subprocess.run('conda config --add channels conda-forge', shell = True)

with open('set_env/modules.txt', 'r') as f:
    modules = [m.strip() for m in f.readlines()]

conda_mods = []
pip_mods   = []

for m in modules:
    print('Checking repositories for {}...'.format(m))
    try:
        subprocess.run('conda search {}'.format(m), check = True, shell = True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        conda_mods.append(m)
    except:
        pip_mods.append(m)

conda_mods_str = ''
for m in conda_mods:
    conda_mods_str += ' ' + m

pip_mods_str = ''
for m in pip_mods:
    pip_mods_str += ' ' + m

print('Conda modules:'+conda_mods_str)
print('Pip modules:'+pip_mods_str)

with open('set_env/install_log.txt', 'w') as f:
    if len(conda_mods_str) > 0:
        print('Installing packages from conda...')
        subprocess.run(f'conda install -n {env_name} --yes'+conda_mods_str  , shell = True, stdout = f)
    if len(pip_mods_str) > 0:
        print('Installing packages in pip...')
        subprocess.run(f'conda run -n {env_name} python -m pip install'+pip_mods_str, shell = True, stdout = f)

print('Running test, do not close the new terminal window...')
cmd = f'conda run -n {env_name} python {main_path}nanokappa.py -ff {main_path}parameters_test.txt'
if sys.platform in ['linux', 'linux2']:
    sp = subprocess.Popen('gnome-terminal --wait -- ' + cmd, shell = True)
    sp.wait()
elif sys.platform == 'darwin':
    sp = subprocess.Popen("osascript -e 'tell app \"Terminal\" to do script \"{}\"' ".format(cmd), shell = True)
    sp.wait()
elif sys.platform == 'win32':
    sp = subprocess.Popen('wt '+cmd, shell = True)
    sp.wait()

while sp.poll() is None:
    time.sleep(1)

if sp.returncode == 0:
    print('The test should be finished by now. You can close this window.')
