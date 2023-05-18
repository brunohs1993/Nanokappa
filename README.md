![](/readme_fig/logo_white.png#gh-light-mode-only)
![](/readme_fig/logo_black.png#gh-dark-mode-only)

# What is Nano-k?

Nano-k is a Python code for phonon transport simulation. It allows to estimate the transport of heat in a given material by importing its properties derived from ab-initio calculations. It is possible to use standard geometries or import external ones.

# Setting the environment

It is recommended to run nano-k using a Conda environment. Firstly, install [Anaconda](https://www.anaconda.com/) on your computer.

Next, clone this repository or download its files to the desired folder.

The necessary libraries can be installed either automatically or manually.

## **Automatic setting:**

To set the `nanokappa` environment automatically, open a terminal window and run:

        $ python <path-to-nanokappa>/set_env/set_env.py
        
> **Obs.:** make sure Python version $\geq$ 3.5. On Linux or MacOS you may need to call `python3` instead of `python`.

This should ensure that all necessary packages are installed either from Conda repositories or via pip. A test simulation is run at the end to check if everything goes ok. It should take some minutes.

## **Manual setting**

If you prefer, you can also set the environment manually:

1.  Create an environment (here called `nanokappa`, but it is an user's choice) and activate it:
   
        $ conda create -n nanokappa python=3.8
        $ conda activate nanokappa

The `(nanokappa)` word will appear on the command line, signaling the conda environment is active.

2. Add conda-forge to the available channels:
   
       (nanokappa) $ conda config --add channels conda-forge

3. Install the needed modules:
   
        (nanokappa) $ conda install -n nanokappa --file <path-to-nanokappa>/Nanokappa/set_env/modules.txt

4. Run a test by executing `python nanokappa.py -ff <path-to-nanokappa>/Nanokappa/parameters_test.txt`. The resulting files should be located at `Nanokappa/test_results/test_X/`. These result files can be safely deleted after the test is finished.

> **Obs.**: Depending on the operating system, some modules may not be available on Conda repositories. In this case, check the modules that caused errors, and manually install from Conda the available modules by running:
>
>        (nanokappa) $ conda install -n nanokappa module1 module2 [module3 ...]
>
> And then try to install the remaining via pip by running
>        
>        (nanokappa) $ conda run -n nanokappa python -m pip install module1 module2 [module3 ...]
>        
> This proedure is done automatically by the `set_env.py` when running the automatic installation.

<!-- **Obs.**: To install on the cluster:

        conda create -n nanokappa -c conda-forge python=3.8
        conda activate nanokappa
        conda install -c conda-forge h5py trimesh phonopy pyembree
        mkdir nanokappa
        cd nanokappa
        git clone https://github.com/brunohs1993/Nanokappa
        conda install -c conda-forge ipython -->

# Running a simulation

## Simulation parameters

In order to define a study case and run a simulation, several parameters need to be set. Geometry, material, boundary conditions and calculation parameters need to be completely defined. These parameters are passed to the program as a pair <keyword, values> directly on command line.

Here is a list of all parameters that can be set:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Parameters file           | `--from_file`        | `-ff`   | File name with extension of a file containing all input parameters. Used to avoid inputing lots of parameters directly on terminal. Full path advised. | String | | 
| Material folder           | `--mat_folder`       | `-mf`   | Path of the folders containing the material files. Full path advisable. | String | |
| hdf5 file                 | `--hdf_file`         | `-hf`   | File names in each `-mf` with extensions. | String | | 
| POSCAR file               | `--poscar_file`      | `-pf`   | File names in each `-mf` with extensions. | String | |
| Results folder            | `--results_folder`   | `-rf`   | The name of the folder to be created containing all result files. If none is informed, no folder is created. | String | `''` |
| Results location          | `--results_location` | `-rl`   | The path where the result folder will be created. It accepts `local` if the results should be saved in the current directory, `main` if they should be saved in the same directory as `nanokappa.py`, or a custom path. | String | `local` |
| Geometry                  | `--geometry`         | `-g`    | Standard geometry name or file name. Geometry coordinates in angstroms. | String | `cuboid` |
| Dimensions                | `--dimensions`       | `-d`    | Dimensions for the standard base geometries (box, cylinder, etc.). | Floats | `20e3 1e3 1e3` |
| Scale                     | `--scale`            | `-s`    | Scale factors for the base geometry (x, y, z) | Float x3 | `1 1 1` |
| Geometry rotation         | `--geo_rotation`     | `-gr`   | Euler angles to rotate the base geometry and which axis to apply (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | Float, String | |
| Material rotation         | `--mat_rotation`     | `-mr`   | Euler angles to change crystal orientation (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | Float, String |  |
| Boundary conditions       | `--bound_cond`       | `-bc`   | Type of boundary condition for each facet declared in `--bound_facets`. If one condition more is given, it is considered to be the same for all non informed facets. Accepts `T` for temperature, `P` for periodic, `R` for roughness/reflection. | String | |
| Positions with imposed BC | `--bound_pos`        | `-bp`   | Set the POSITIONS from which to find the closest facet to apply the specific boundary conditions. First value is a keyword `relative` - considers all points in the mesh between 0 and 1 - or `absolute` - direct positions. | String Float | | 
| Boundary condition values | `--bound_values`     | `-bv`   | Values for each imposed boundary conditions. Temperatures in Kelvin, roughness in angstroms. | Float | `303 297` |
| Temperature distribution  | `--temp_dist`        | `-td`   | Shape of the initial temperature profile. Accepts `cold`, `hot`, `mean`, `linear`, `random`, `custom`. | String | `cold` |
| Subvolume temperature     | `--subvol_temp`      | `-st`   | Initial temperature of each subvolume when `custom` is informed in `-td`, in Kelvin. | Float | |
| Reference temperature     | `--reference_temp`   | `-rt`   | The temperature at which the occupation number for every mode will be considered zero, in Kelvin. Alternatively, the user can set it as "local" to use the local temperature of each particle.| Float/String | `local` |
| Temperature interpolation | `--temp_interp`      | `-ti`   | How to interpolate the temperature between the subvolumes' reference points. Accepts `nearest`, `linear` (when slice subvolumes are used) or `radial` (for grid or voronoi subvolumes). | String | `nearest` | 
| N° of particles           | `--particles`        | `-p`    | Number of particles given as `keyword number`. Can be given as the total number (keyworld `total`), the number per-mode-per-subvolume (keyworld `pmps`) and the number per cubic angstom (keyworld `pv`). | String Integer | `pmps 1` |
| Timestep                  | `--timestep`         | `-ts`   | Timestep of each iteration in picoseconds | Float | `1` |
| Iterations                | `--iterations`       | `-i`    | Number of iterations to be performed | Integer | `10000` |
| Maximum simulation time   | `--max_sim_time`     | `-mt`   | Maximum time the simulation will run. Declared as `D-HH:MM:SS`. If the simulation arrives to this time, the calculation is finished and post processing is executed. If `0-00:00:00` is informed, no time limit is imposed. | String |`0-00:00:00`|
| Subvolumes                | `--subvolumes`       | `-sv`   | Type of subvolumes, number of subvolumes and slicing axis when the case (x = 0, y = 1, z = 2). Accepts `slice`, `grid` and `voronoi` as subvolume types. | String Integer (Integer Integer) | `slice 10 0` |
| Path points               | `--path_points`      | `-pp`   | Set the approximate points where the path to calculate $\kappa_{path}$ will go through. Declared the same way as `--bound_pos`.| String Float | `relative 0 0.5 0.5 1 0.5 0.5` |
| Number of datapoints      | `--n_mean`           | `-nm`   | Number of datapoints considered to calculated mean and standard deviation values. Each datapoint is 10 timesteps.| Int | `100` |
| Real time plot            | `--rt_plot`          | `-rp`   | Property to plot particles to generate animations.  | String | |
| Figure plot               | `--fig_plot`         | `-fp`   | Property to plot particles at end of run (frequency, occupation, etc.) | Strings | `e` |
| Colormap                  | `--colormap`         | `-cm`   | Matplotlib [colormap](https://matplotlib.org/stable/gallery/color/colormap_reference.html) to be used in geometry plots (not convergence). | String | `jet` |
| Convergence criteria      | `--conv_crit`        | `-cc`   | Criteria for convergence and stop simulation. | Float | 1e-6 |

### Debugging parameters

These should only be used to detect possible errors in the simulation:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Particle distribution     | `--part_dist`        | `-pd`   | How to distribute particles at the beginning of the simulation. Composed of two keywords: `random/center_subvol/domain`. | String | `random_subvol` | -->
| Reservoir generation      | `--reservoir_gen`    | `-rg`   | How to generate particles on the reservoirs. With `constant`, particles are generated in a constant rate based in a timestep counter; `fixed_rate` means the particles are generated randomly, but in an approximately fixed rate; `one_to_one` means that the number of particles generated is the same of particles leaving in order to keep the number of particles stable. | String | `fixed_rate` |
| Empty subvols             | `--empty_subvols`    | `-es`   | Index of subvolumes that are to be initialised as empty (no particles). | Integer | |
| Energy normalisation      | `--energy_normal`    | `-en`   | The way to normalise energy to energy density. Choose between `fixed` (according to the expected number of particles in the subvolume) and `mean` (arithmetic mean of the particles inside). | String | `mean` |

<p>&nbsp</p>

### The `--from_file` parameter

The user can input all parameters sequentially directly on terminal, which is easy when there are a few parameters that differ from the standard values. When there is a highly customised simulation to be ran, it is better to use an external input file and use the `--from_file` or `-ff` argument.

All inputs can be given in form of a text file. The following shows an example of the content of a txt file that we are calling `parameters.txt`:

    --mat_folder       D:\Materials\Si\
    --hdf_file         kappa-m313131.hdf5
    --poscar_file      POSCAR
    --geometry         box
    --dimensions       10e3 1e3 1e3
    --scale            1 1 1
    --geo_rotation     90 y
    --subvolumes       slice 10 0
    --bound_pos        relative 0 0.5 0.5 1 0.5 0.5 0.5 0 0.5 0.5 1 0.5
    --bound_cond       T T R R P
    --bound_values     302 298 10 10
    --connect_pos      absolute 5e3 5e2 0 5e3 5e2 1e3
    --reference_temp   local
    --temp_dist        cold
    --particles        total 1e5
    --timestep         1
    --iterations       2000
    --results_folder   inplane_film
    --results_location D:\Results
    --conv_crit        0
    --fig_plot         subvolumes
    --rt_plot          
    --max_sim_time     0-01:00:00

This file defines a simulation with the material data contained in the Si folder, being defined by the hdf and poscar files. The geometry is a box with dimensions 10e3 ang, 1e3 ang,  and 1e3 ang in x, y and z directions respectively. Temperatures (302 K and 298 K) are imposed on both extremities in the x direction, and a roughness (10 ang) is set to the opposite facets in y direction. Opposite facets in the z direction are conected as periodic. A local reference temperature is used. The result files will be stored in `D:\Results\inplane_film\`. Besides the usual convergence plots, the subvolumes plot will be generated at the beginning using the default jet colormap. The simulation will last either 2000 iterations or 1 hour, whatever happens first.

Any non-necessary arguments can be left empty. In this example, `--subvol_temp` is left empty since the temperature profile is completely defined by the `cold` keyword used as argument for `--temp_dist`. Since there is no inputs for `--rt_plot`, no real time plots will be produced. It is important to note, however, that passing an empty argument in the file will override the standard values and pass an empty list to the parser. If the user wishes to use standard input value for a given argument, the argument should be omitted in the parameters.txt file altogether.

The program could then be executed on terminal by calling:

    $ python <path_to_nanokappa>/nanokappa.py -ff <path-to-file>/parameters.txt

### Material properties

The properties describing the material are derived from hdf5 and POSCAR files. These files needed to be informed with extensions. They should both be in the folder informed in full to the `--mat_folder` argument. The files with material properties are the only mandatory parameters to be informed. All others have standard values associated for quick tests, as seen on the table above. The material can also be rotated by passing a set of angles and rotation order to `--mat_rotation`.

### Geometry definition

The geometry can be defined from an standard geometry or an external file. The standard geometries are the following:

| Geometry        | Key for `--geometry`     | Parameters for `--dimensions`         |
| --------------- | ------------------------ | ------------------------------------- |
| Box             | `box`, `cuboid`          | Lx, Ly, Lz                    |
| Cylinder        | `cylinder`, `rod`, `bar` | H, R, N_sides    |
| Variable cross-section corrugated wire | `corrugated` | L, l, R, r, N_sides, N_sections |
| Constant cross-section corrugated wire | `zigzag` | H, R, N_sides, N_sections, h|
| "Castle"        | `castle`                 | L, l, R, r, N_sides, N_sections, S |
| Radially corrugated wire | `star`          | H, R, r, N_points. |
| Free shape wire | `freewire`          | R0, L0, R1, L1, R2, L2 ... L(N), R(N+1), N_sides |


<p>&nbsp</p>

These standard geometries can be modified by entering the parameters `--scale` and `--geo_rotation`. For example, a box with L_x = 100, L_y = 100 and L_z = 200 (all lengths in angstroms), there are two ways to enter parameters.

1. By directly inputing the dimensions...

        --geometry   box
        --dimensions 100 100 200

2. ...or by inputing any dimensions and scaling it by x, y and z factors, such as:
   
        --geometry   box
        --dimensions 100 100 100
        --scale      1 1 2

The `--dimensions` arguments are required only for standard geometries. Imported geometries ignore this command. The inputs `--scale` and `--geo_rotation`, however, work with imported geometries the same way as with the standard ones.

Whatever is the geometry, it is always rezeroed so that all vertices are in the all positive quadrant (every x, y and z coordinate is greater than or equal to zero).

### Boundary conditions

The boundary conditions (BC) consist of heat transfer restrictions (such as imposed temperatures) and boundary surfaces properties (roughness or periodic). The user sets the desired facets on which to apply the BC by passing their positions `--bound_pos`, the BC type to `--bound_cond` and their respective values to `--bound_values`. For instance, in the previously given example we have:

    --bound_pos    relative 0 0.5 0.5 1 0.5 0.5 0.5 0 0.5 0.5 1 0.5
    --bound_cond   T T R R P
    --bound_values 302 298 10 10
    --connect_pos  absolute 5e3 5e2 0 5e3 5e2 1e3

So, in order:

- Facet located at the relative position [0 0.5 0.5] has an imposed temperature of 302 K;
- Facet located at the relative position [1 0.5 0.5] has an imposed temperature of 298 K;
- Facets located at relative positions [0.5 0 0.5] and [0.5 1 0.5] both have a roughness of 10 angstroms;
- The remaining facets are not mentioned in `--bound_pos`, hence they pick the last informed boundary condition, which is periodic (`P`). To complete, their connection needs to be informed to `--connect_pos`.

The `periodic` BC can only be applied to facets that have vertices in the same relative position, and their normal must be parallel and pointing in opposite directions. This is of vital importance, since crystal orientation is relevant to the phonon properties: an interface between crystals of the same material but in different orientations (such as grain boundaries) _cannot_ be considered periodic. In this configuration, whenever a particle crosses a boundary (that is not with a fixed temperature) it is transported to the same position as it entered on the opposite boundary, as the solid was composed by several cuboid domains side by side (hence, periodic).

### Subvolumes

The subvolumes are subdivisions of the domain so it is possible to calculate local quantities, such as energy density, temperature and heatflux. The domain is subdivided in `N` non-intersecting pieces and local quantities are calculated only from the phonons located inside that region. There are three types of subvolumes that can be declared:

- `slice` slices the domain i.e. divides the geometry using equidistant planes along a given axis. For example, to slice the domain 5 times along x axis:

    --subvolumes slice 5 0

- `grid` slices the domain in all three axis and removes subvolumes too small in comparison to the rest. For example, to generate a grid of 5 x 4 x 3 subvolumes (x, y and z, respectively):

    --subvolumes grid 5 4 3

- `voronoi` generates subvolumes by iteratively adjusting their positions so that it is more or less equilibrated. It is useful for complex geometries. For example, to generate 10 subvolumes without any position restriction:

    --subvolumes voronoi 10

The definition of the Voronoi subvolumes is given by the following algorithm:

- Generate initial subvolume reference points $x_r$;
- Sample the geometry with a number $N_s$ of generated points;
- Define their correspondent subvolume i.e. the one with closest $x_r$;
- Update $x_r$ to the center of mass of the subvolume (average samples position);
- Repeat while increasing the number of samples until there is no change on $x_r$.

The Figure shows an example in two dimensions. The red dots are the current $x_r$ and the black dots are the updated $x_r$ for the next iteration. It can be seen that even with a number of subvolumes that is not a perfect square, the subvolumes are organised rather evenly at the end of the process.

![](/readme_fig/voronoi.png)

This algorithm has shown to be flexible, but can cause some problems depending of the complexity of the geometry and of the initial $x_r$. Geometries with indents or holes, for example, can lead to particles on each side of the gap to be considered in the same subvolume, which can lead to an unreal energy transfer through the space. This sometimes can be avoided by just rerunning the simulation, but usually better results can be achieved by increasing the number of subvolumes, so that each side of the empty space is classified as a different subvolume. It is important to have user discretion while applying this type of subvolume.

### Result files

The user can specify where they want to save the results of the simulation using the `--results_location` and `--results_folder` parameters. For example:

    --results_location D:/Documents/Results
    --results_folder   test

Here a folder called `test_0` will be created in D:/Documents/Results and all results will be stored inside. If there is already a `test_<N>` folder in that location, it will create a `test_<N+1>` folder. The `--results_location` parameter also accepts `local` and `main` as arguments. In the former, the `--results_folder` is created from the directory Nanokappa was called from. In the latter, the folder is created in the Nanokappa root folder.

The results files are comprised of:
- List of the arguments used in the simulastion (`arguments.txt`);
- Plot of the geometry with the facets color coded by boundary condition (`BC_plot.png`);
- Plots of particles at the beginning of the simulation, as demanded in `--fig_plot`;
- Convergence plots of relevant quantities (temperature, energy density, heat flux, number of particles, thermal conductivity);
- Convergence data (`convergence.txt` and `residue.txt`);
- Particle and subvolume data (`particle_data.txt` and `subvolumes.txt`);
- Modes related by specular reflection (`specular_correspondences.txt`);
- The animation of the simulation if demanded in `--rt_plot` (`simulation.gif`).

<p>&nbsp</p>

# Executing multiple simulations in parallel

If the user wishes to simulate several cases in order to do, for example, a parametric analysis, it is possible to do it with the help of a CSV file. This file should have in the first line the name of each parameter separated by commas **without spaces** i.e. `particles,poscar_file,hdf_file,...` and so on. The subsequent lines contain the desired parameters in the same order as declared in the first line. This file will be read an treated as a Pandas dataframe, and each line will be passed as arguments to call `nanokappa.py` as a subprocess. The parallel processing can be done locally or on a cluster that uses Slurm for job scheduling.

## Running locally

To run multiple simulations locally, write on terminal the following command:

    $ python <Nanokappa-folder>/optimiser/run_csv.py --mode local --parameters <csv_file.csv> --options <options_file.txt>

The `--mode` (or `-m`) argument says that the run will be local; the `--parameters` (or `-p`) argument informs the CSV file with the simulation parameters; and the `--options`  (or `-o`) argument passes the options to be given to the Python multiprocessing funcions. The options file for local run should contain the number of processes and the Nanokappa folder that contains the desired version of `nanokappa.py`. This should be written in the file in the following way:

    processes        3
    nanokappa_folder d:/LEMTA/Code/Nanokappa

In the local run, the results are stored according to the names and paths given as arguments, in the same way it happens when running a single simulation.

**Obs.:** on Windows, the new terminal subprocesses are spawned as independent from the parent terminal, and the subprocess returns its exit code as concluded as soon as the new terminal windows is opened. This causes all simulations to run simultaneously. To avoid that, a waiting time is imposed so that the number of concurrent simulations is kept safely as intended. The waiting time is the time informed in `--max_sim_time` plus an 10 minutes margin for postprocessing. This does not happen on Linux and was not yet tested on MacOS.

## Running on a cluster

The command to run it on the cluster is given by:

    $ python <Nanokappa-folder>/optimiser/run_csv.py --mode cluster --parameters <csv_file.csv> --options <options_file.txt>

The options file for this needs more parameters. These will be informed to Slurm at the beginning of the bash script submitted to the job:

    mail_user        <user_email@email.com>
    mail_type        ALL
    job_name         nanokappa
    partition        <partition_name>
    n_of_nodes       1
    n_of_tasks       1
    nanokappa_folder <path_to_nanokappa>/Nanokappa
    conda_env        nanokappa

This informs the user email and the events about which the user should be notified; the name of the job; the partition to be used; the number of nodes and tasks; the folder where `nanokappa.py` is located; and the name of the conda environment where the necessary modules are installed.

In the cluster run, the user should run the command from the folder where they want the results to be saved. So for example, the user can create a folder called `Results` and run the command previously shown from inside this folder:

    $ mkdir Results
    $ cd Results
    $ python <Nanokappa-folder>/optimiser/run_csv.py -m cluster -p <csv_file.csv> -o <options_file.txt>

The code will create, for each line in the CSV file, a result folder called `params_<line>_<try>`, where `<line>` is the line number and `<try>` is the number of the try for those parameters (for example, if the line 3 of the csv is run twice from the `Results` folder, there would be the folders `Results/params_3_0` and `Results/params_3_1`). Every time one of these folders are created, the code generates the parameter file and the batch script that will be submited to Slurm. In this script, the code generates a work directory in the partition and copies all relevant files (material, geometry and Nanokappa source code itself). It then runs Nanokappa in the work directory and copies back the result files to the `params` folder in the submitting directory. This makes thus each run independent and leaves the queue under Slurm's responsibility.

# Optimisation



# Future enhancements

Ideas for next developments are welcome and will be developed as time allows it. Some of our ideas for next implementations include:

- Heat flux as boundary condition;
- Different reflection models;
- Interface between materials as boundary condition;
- Subvolumes of different materials and calculated phonon transmition between them;
- GUIs for setting parameters and accompanying results as the simulation runs;
- Time optimisation;
- Superposed electron transport;
- Different Monte Carlo approaches;
- Different calculation methods;
- And more.

# Aknowledgements

This code was written by Bruno Hartmann da Silva as part of his PhD, under supervision of Dr. Prof. Laurent Chaput and Dr. Prof. David Lacroix at the Laboratoire Énergies et Mécanique Théorique et Appliquée (LEMTA), Université de Lorraine, Vandoeuvre-lès-Nancy, France. The PhD was funded by the Agence National de la Recherche (ANR).
