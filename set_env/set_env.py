import subprocess, sys, time

subprocess.run('conda create -n nanokappa python=3.8 --yes', shell = True)
subprocess.run('conda activate nanokappa', shell = True)
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
        subprocess.run('conda install -n nanokappa --yes'+conda_mods_str  , shell = True, stdout = f)
    if len(pip_mods_str) > 0:
        print('Installing packages in pip...')
        subprocess.run('conda run -n nanokappa python -m pip install'+pip_mods_str, shell = True, stdout = f)

print('Running test...')
cmd = 'echo Do not close this window... & conda run -n nanokappa python nanokappa.py -ff parameters_test.txt'
if sys.platform in ['linux', 'linux2']:
    sp = subprocess.Popen('gnome-terminal --wait -- ' + cmd, shell = True)
    sp.wait()
elif sys.platform == 'win32':
    sp = subprocess.Popen('wt '+cmd, shell = True)
    sp.wait()

while sp.poll() is None:
    time.sleep(1)

if sp.returncode == 0:
    print('The test should be finished by now. You can close this window.')
