    ooo        ooooo   .oooooo.        ooooooooo.   oooo                                                    
    `88.       .888'  d8P'  `Y8b       `888   `Y88. `888                                                    
     888b     d'888  888                888   .d88'  888 .oo.    .ooooo.  ooo. .oo.    .ooooo.  ooo. .oo.   
     8 Y88. .P  888  888                888ooo88P'   888P"Y88b  d88' `88b `888P"Y88b  d88' `88b `888P"Y88b  
     8  `888'   888  888                888          888   888  888   888  888   888  888   888  888   888  
     8    Y     888  `88b    ooo        888          888   888  888   888  888   888  888   888  888   888  
    o8o        o888o  `Y8bood8P'       o888o        o888o o888o `Y8bod8P' o888o o888o `Y8bod8P' o888o o888o 

# What is MC Phonon?

MC Phonon is a Python code for phonon transport simulation using Monte Carlo method. It allows to estimate the transport of heat in a given material by importing  its properties derived from ab-initio calculations. It is possible to use standard geometries or import external ones.

# Setting the environment

It is recommended to run MC Phonon through Conda. This is set by following the steps:

1. Clone this repository or download its files to the desired `<mcphonon-folder>`;
2. Install Anaconda on your computer (https://www.anaconda.com/);
3. Create an environment (here called `mcphonon`, but it is an user's choice) and activate it:
   
       conda create -n mcphonon
       conda activate mcphonon

4. Add conda-forge to the available channels:
   
       conda config --add channels conda-forge

5. Install the needed modules:
   
        conda install --file <mcphonon-folder>/modules.txt

# Running a simulation

## Simulation parameters

### Table of parameters

| Parameter                | Command        | Reduced command | Values                                        | Types         | Default         |
| ------------------------ | -------------- | --------------- | --------------------------------------------- | ------------- | --------------- |
| hdf5 file                | --hdf_file     | -hf             | File name w/ extension                        | String        |                 | 
| POSCAR file              | --poscar_file  | -pf             | File name                                     | String        |                 |
| Geometry                 | --geometry     | -g              | std geo name or file name                     | String        | `cuboid`        |
| N° of particles          | --particles    | -p              | approximate n° of particles                   | Number        | `1e6`           |
| Temperatures             | --temperatures | -t              | Boundary conditions for temperature in Kelvin | Number Number | `310 290`       |
| Temperature distribution | --temp_dist    | -td             | initial temperature profile                   | String        | `constant_cold` |

<!-- XXXXXXXX KEEP GOING xxxxxxxxxxxxxxxx -->

### Material properties

The properties describing the material are derived from hdf5 and POSCAR files. These files needed to be informed in full (with extensions) from the `materials` folder. For example the command:

    $ python main_program.py -hf Si/<hdf5-file>.hdf5 -pf Si/<poscar-file>

runs a simulation with materials properties of silicon, saved in the folder `/materials/Si/`, retrieved from the hdf5 and POSCAR files informed. All other parameters are standard.

The files with material properties are the only mandatory parameters to be informed. All others have standard values associated for quick tests.

### Slices

The slices are subdivisions of the domain so it is possible to calculate local quantities. The domain is subdivided in `N` equal parts (slices) along a given axis, and local quantities are calculated from the phonons located inside that slice. To declare the number of slices and the direction along which to slice, 

