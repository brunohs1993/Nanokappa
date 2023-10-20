from datetime import datetime, timedelta
from classes.Geometry import Geometry
from classes.Phonon import Phonon
from classes.Population import Population
from classes.Visualisation import Visualisation
from argument_parser import *
import sys
import re

print('\n'+
    ' nonano          no   onanonan        onanon          on nonanonan              anon    anona\n'+
    'ano   ona       ano  nona   anon     non  onan       nonano     anon            ano   ona\n'+
    'ano    nano     anonano      nona    non    anon     nonan       non    anona   anonano\n'+
    'ano     anon    anonan         nan   non     nona    nonan       non   nanon    nano anon\n'+
    'ano       onan  anona           anonanon      onan   nonano     anon           nan   nona\n'+
    'an         nanonano               nonanon      nanonano  nonanonan            onan    onano')

print('\n'+'Running simulation, please wait. Check the results folder for the current status.')

args = read_args()

# seting result files location
args = generate_results_folder(args)

# saving arguments on file
args_filename = os.path.join(args.results_folder, 'arguments.txt')

if args.output == 'file':
    output_file = open(os.path.join(args.results_folder, 'output.txt'), 'a')
    sys.stdout = output_file

f = open(args_filename, 'w')

for key in vars(args).keys():
    s = f'--{key} '
    if type(vars(args)[key]) == str:
        s += f'{vars(args)[key]}\n'
    else:
        for i in vars(args)[key]:
            s += f'{i} '
        s += '\n'
    f.write(s)

f.close()

# getting maximum sim time
max_time = re.split('-|:', args.max_sim_time[0])
max_time = [int(i) for i in max_time]
max_time = timedelta(days    = max_time[0],
                    hours   = max_time[1],
                    minutes = max_time[2],
                    seconds = max_time[3])

# getting start time
start_time = datetime.now()

print('---------- o ----------- o ------------- o ------------')
print("Year: {:<4d}, Month: {:>02d}, Day: {:>02d}".format(start_time.year, start_time.month, start_time.day))
print("Start at: {:>02d} h {:>02d} min {:>02d} s".format(start_time.hour, start_time.minute, start_time.second))	
print(f"Simulation name: {args.results_folder}")
print('---------- o ----------- o ------------- o ------------')

# initialising geometry

geo = Geometry(args)

# opening file
if len(args.mat_folder) == 1:
    phonons = Phonon(args, 0)
else:
    phonons = [Phonon(args, i) for i in range(len(args.mat_folder))]

# THIS IMPLEMENTATION OF PHONONS AS A LIST NEEDS TO BE INCLUDED IN THE POPULATION CLASS.
# FOR NOW ONLY ONE MATERIAL WORKS

pop = Population(args, geo, phonons)

flag = True
while flag:
    pop.run_timestep(geo, phonons)

    flag = (pop.current_timestep < args.iterations[0]) and not pop.finish_sim
    
    if max_time.total_seconds() > 0:
        flag = flag and datetime.now()-start_time < max_time

print('Saving end of run particle data...')
pop.write_final_state(geo)

pop.f.close()

pop.save_plot_real_time()

pop.view.postprocess()

end_time = datetime.now()

total_time = end_time - start_time

print('---------- o ----------- o ------------- o ------------')
print("Start at: {:>02d}/{:>02d}/{:>4d}, {:>02d} h {:>02d} min {:>02d} s".format(start_time.day, start_time.month, start_time.year, start_time.hour, start_time.minute, start_time.second))	
print("Finish at: {:>02d}/{:>02d}/{:>4d}, {:>02d} h {:>02d} min {:>02d} s".format(end_time.day, end_time.month, end_time.year, end_time.hour, end_time.minute, end_time.second))

hours = total_time.seconds//3600
minutes = (total_time.seconds//60)%60

seconds = total_time.seconds - 3600*hours - 60*minutes

print("Total time: {:>02d} days {:>02d} h {:>02d} min {:>02d} s\n".format(total_time.days, hours, minutes, seconds))
print('---------- o ----------- o ------------- o ------------')

if args.output == 'file':
    output_file.close()