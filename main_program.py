from datetime import datetime
from mcphonon import *
import sys

args = read_args()

# seting result files location
args = generate_results_folder(args)

# saving arguments on file
args_filename = args.results_folder + 'arguments.txt'

# output_file = open(args.results_folder + 'output.txt', 'a')
# sys.stdout = output_file

f = open(args_filename, 'w')

for key in vars(args).keys():
    f.write( '{} = {} \n'.format(key, vars(args)[key]) )

f.close()

# getting start time
start_time = datetime.now()

print('---------- o ----------- o ------------- o ------------')
print("Year: {:<4d}, Month: {:<2d}, Day: {:<2d}".format(start_time.year, start_time.month, start_time.day))
print("Start at: {:<2d} h {:<2d} min {:<2d} s".format(start_time.hour, start_time.minute, start_time.second))	
print("Simulation name: " + args.results_folder)
print('---------- o ----------- o ------------- o ------------')

# initialising geometry
geo = Geometry(args)

# opening file
if len(args.mat_names) == 1:
    phonons = Phonon(args, 0)
    if 0 in args.pickled_mat:
        phonons = phonons.open_pickled_material()
    else:
        phonons.load_properties()
else:
    phonons = [Phonon(args, i) for i in range(len(args.mat_names))]
    for i in range(len(phonons)):
        if i in args.pickled_mat:
            phonons[i] = phonons[i].open_pickled_material()
        else:
            phonons[i].load_properties()

# THIS IMPLEMENTATION OF PHONONS AS A LIST NEEDS TO BE INCLUDED IN THE POPULATION CLASS.
# FOR NOW ONLY ONE MATERIAL WORKS

pop = Population(args, geo, phonons)

pop.plot_figures(geo, property_plot = ['T', 'n', 'omega', 'e'])

visuals = Visualisation(args, geo, phonons)

visuals.preprocess()

print('Simulating...')

while (pop.current_timestep < args.iterations[0]) and (pop.residue.max() >= pop.conv_crit):
    
    pop.run_timestep(geo, phonons)

print('Saving end of run particle data...')
pop.write_final_state(geo, phonons)

pop.f.close()

pop.save_plot_real_time()

visuals.postprocess()

end_time = datetime.now()

total_time = end_time - start_time

print('---------- o ----------- o ------------- o ------------')
print("Start at: {:<2d} h {:<2d} min {:<2d} s".format(start_time.hour, start_time.minute, start_time.second))	
print("Finish at: {:<2d} h {:<2d} min {:<2d} s".format(end_time.hour, end_time.minute, end_time.second))

hours = total_time.seconds//3600
minutes = (total_time.seconds//60)%60

seconds = total_time.seconds - 3600*hours - 60*minutes

f = open(args_filename, 'a+')
f.write('---------- o ----------- o ------------- o ------------\n' +
        "Start at: {:<2d} h {:<2d} min {:<2d} s\n".format(start_time.hour, start_time.minute, start_time.second)+
        "Finish at: {:<2d} h {:<2d} min {:<2d} s\n".format(end_time.hour, end_time.minute, end_time.second)+
        "Total time: {:<2d} h {:<2d} min {:<2d} s\n".format(hours, minutes, seconds)+
        '---------- o ----------- o ------------- o ------------')
f.close()

print("Total time: {:<2d} h {:<2d} min {:<2d} s".format(hours, minutes, seconds))
print('---------- o ----------- o ------------- o ------------')

output_file.close()