![](/readme_fig/logo_white.png#gh-light-mode-only)
![](/readme_fig/logo_black.png#gh-dark-mode-only)

# What is Nano-&#954;?

Nano-&#954; is a Python code for phonon transport simulation, combining Monte Carlo method with ab-initio material data. The user can simulate preset, standard geometries or import external ones in STL format.

# Setting the environment

It is recommended to run Nano-&#954; using a Conda environment. Firstly, install [Anaconda](https://www.anaconda.com/) on your computer.

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

In order to define a study case and run a simulation, several parameters need to be set. Geometry, material data, boundary conditions and calculation parameters need to be completely defined. These parameters are passed to the program as a pair <keyword, values> directly on command line.

### The `--from_file` parameter

The user can input all parameters sequentially directly on terminal, which is easy when there are just a few parameters. However it is often not the case, and it is usually better to use an external input file and use the `--from_file` or `-ff` argument. You can save the parameters on a txt file written just like you would do on command line, but separating each parameter in a new line. Then you could run:

    python nanokappa.py -ff <parameter_file>.txt

### Where to save the results

The location of the results is set with two arguments:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Results folder            | `--results_folder`   | `-rf`   | The name of the folder to be created containing all result files. If none is informed, no folder is created. | String | `''` |
| Results location          | `--results_location` | `-rl`   | The path where the result folder will be created. It accepts `local` if the results should be saved in the current directory, `main` if they should be saved in the same directory as `nanokappa.py`, or a custom path. | String | `local` |

### Geometrical parameters

These parameters are the ones that relate to the geometry that will be simulated. The geometry can be imported as an STL file by setting `--geometry <file_name>` or chosen among the following options:

| Geometry        | Key for `--geometry`     | Parameters for `--dimensions`         |
| --------------- | ------------------------ | ------------------------------------- |
| Box             | `box`, `cuboid`          | Lx, Ly, Lz                    |
| Cylinder        | `cylinder`, `rod`, `bar` | H, R, N_sides    |
| Variable cross-section corrugated wire | `corrugated` | L, l, R, r, N_sides, N_sections |
| Constant cross-section corrugated wire | `zigzag` | L, R, dx, dy, N_sides, N_sections|
| "Castle"        | `castle`                 | L, l, R, r, N_sides, N_sections, S |
| Radially corrugated wire | `star`          | H, R, r, N_points. |
| Free shape wire | `freewire`          | R0, L0, R1, L1, R2, L2 ... L(N), R(N+1), N_sides |

<p>&nbsp</p>

These standard geometries can be modified by entering the parameters `--scale` and `--geo_rotation`. For example, in order to declare a box with $L_x$ = 100, $L_y$ = 100 and $L_z$ = 200 (all lengths in angstroms), there are several ways to enter parameters:

1. The most straightforward, by directly inputing the dimensions...

        --geometry   box
        --dimensions 100 100 200

2. ...or by inputing any dimensions and scaling it by x, y and z factors, such as...
   
        --geometry   box
        --dimensions 100 100 100
        --scale      1 1 2

3. ... or even by setting the dimensions in another order and rotating it:

        --geometry     box
        --dimensions   200 100 100
        --geo_rotation 90 y

The `--dimensions` arguments are required only for standard geometries. Imported geometries ignore this command. The inputs `--scale` and `--geo_rotation`, however, work with imported geometries the same way as with the standard ones. It is important to note that the scaling of the geometry comes _before_ than its rotation.

Whatever is the geometry, it is always rezeroed so that all vertices are in the all positive quadrant (every x, y and z coordinate is greater than or equal to zero).

In summary:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Geometry                  | `--geometry`         | `-g`    | Standard geometry name or file name. Geometry coordinates in angstroms. | String | `cuboid` |
| Dimensions                | `--dimensions`       | `-d`    | Dimensions for the standard base geometries (box, cylinder, etc.). | Floats | `20e3 1e3 1e3` |
| Scale                     | `--scale`            | `-s`    | Scale factors for the base geometry (x, y, z) | Float x3 | `1 1 1` |
| Geometry rotation         | `--geo_rotation`     | `-gr`   | Euler angles to rotate the base geometry and which axis to apply (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | Float, String | |

### Material parameters

The parameters that treat the material data are:

| Parameter          | Keyword          | Reduced | Description | Types | Default |
| ------------------ | ---------------- | ----- | ----------- | ----- | ------- |
| Material folder    | `--mat_folder`   | `-mf` | Path of the folder containing the material files. Full path advised. | String | |
| hdf5 file          | `--hdf_file`     | `-hf` | File name with extension. | String | | 
| POSCAR file        | `--poscar_file`  | `-pf` | File name with extension. | String | |
| Material rotation  | `--mat_rotation` | `-mr` | Euler angles to change crystal orientation (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | Float, String |  |
| Isotope scattering | `--isotope_scat` | `-is` | Additional scattering due to impurities/defects. | Int | |

The properties describing the material are derived from hdf5 and POSCAR files derived as result from [Phono3py](https://phonopy.github.io/phono3py/index.html). They should both be in the folder informed in full to the `--mat_folder` argument. The material can also be rotated by passing a set of angles and rotation order to `--mat_rotation`.

The `--isotope_scat` parameter signals whether there is additional phonon scattering to be considered due to impurities or deffects, e.g. in alloys data. For this the hdf5 file should include the `gamma_isotope` field. The activation is signaled by passing the index of the material that should consider it. Since Nano-&#954; accepts only one material for the simulation, the user should pass `-is 0` if there is additional scattering, and nothing if there is not.

> **Obs.:** Only one material is accepted at the time because Nano-&#954; does not support phonon transmission on interfaces between different materials for now. This is planned to be added in the future.

To use the material offered as example in the test_material folder, it should be declared as:

    --mat_folder  <path_to_nanokappa>/test_material/
    --hdf_file    kappa-m313131.hdf5
    --poscar_file POSCAR

### Boundary conditions

The boundary conditions (BC) consist of heat transfer restrictions (such as imposed temperatures) and boundary surfaces properties (roughness or periodic). The user sets the BC with the following parameters:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Positions with imposed BC | `--bound_pos`        | `-bp`   | Set the coordinates from which to find the closest facet to apply the specific boundary conditions. First value is a keyword `relative` (normalises the bounding box between 0 and 1) or `absolute` (direct positions). The points are passed as $x_1~y_1~z_1~x_2~y_2~z_2...$.| String Float | | 
| Boundary conditions       | `--bound_cond`       | `-bc`   | Type of boundary condition for each facet detected from the coordinates declared in `--bound_pos`. Accepts `T` for temperature, `P` for periodic, `R` for roughness/reflection. If one extra condition is given, it is considered to be the same for all non informed facets.  | String | |
| Boundary condition values | `--bound_values`     | `-bv`   | Values for each imposed boundary condition. Temperatures in Kelvin, roughness in angstroms. | Float | |
| Connected facets          | `--connect_pos`      | `-cp`   | Declared the same way as in `-bp`, it declares which facets are connected. They are treated in pairs: the first is connected to the second, the third to the fourth, etc. | String Float | |

As an example, let's say that you would like to simulate a box measuring 10e3 &#8491; X 1e3 &#8491; X 1e3 &#8491;:

    --geometry box
    --dimensions 1e4 1e3 1e3

We could simulate heat transfer in a thin film in inplane direction by setting:
 - a temperature difference between both $x$ extremities;
 - rough facets in upper and lower facets;
 - periodicity in the two that are left.

Declared as parameters, it could look like:

    --bound_pos    relative 0 0.5 0.5 1 0.5 0.5 0.5 0.5 0 0.5 0.5 1
    --bound_cond   T T R R P
    --bound_values 302 298 10 10
    --connect_pos  absolute 5e3 0 5e2 5e3 1e3 5e2

So, in order:

- Facet located at the relative position [0 0.5 0.5] has an imposed temperature of 302 K;
- Facet located at the relative position [1 0.5 0.5] has an imposed temperature of 298 K;
- Facets located at relative positions [0.5 0.5 0] and [0.5 0.5 1] both have a roughness of 10 &#8491;;
- The remaining facets are not mentioned in `--bound_pos`, hence they pick the last informed boundary condition, which is periodic (`P`). To complete, their positions need to be informed to `--connect_pos`.

The periodic (`P`) BC can only be applied to facets that have the same geometrical boundaries in relation to their centroid, and their normal must be parallel and pointing in opposite directions. This is of vital importance, since crystal orientation is relevant to the phonon properties: an interface between crystals of the same material but in different orientations (such as grain boundaries) _cannot_ be considered periodic. In this configuration, whenever a particle crosses a boundary (that is not with a fixed temperature) it is transported to the same position as it entered on the opposite boundary, as the solid was composed by several cuboid domains side by side (hence, periodic).

The imposition of temperatures as BC (`T`) treats the respective facet as black body, always emmiting phonons according to the Bose-Einstein distribution:

$$ n^0(\omega, T) = \frac{1}{\exp(\hbar \omega/k_b T)-1}$$

The rough facet BC (`R`) uses a generalisation of the model described by [Ziman](https://academic.oup.com/book/32666) to calculate the probability of a particle being subjected to a specular or to a diffuse reflection, depending on its vibrational mode. In specular reflections, the velocity is perfectly mirrorred, frequency is kept and intensity (occupation) is conserved. In diffuse refflections, the wall behaves like a a black body absorbing completely the energy of the particle and reemitting another random mode with occupation calculated according to the Bose-Einstein distribution at local temperature.

### Subvolumes

The subvolumes (SV) are subdivisions of the domain so it is possible to calculate local quantities, such as energy density, temperature and heat flux. Think cells, mesh or grid, but not exactly. There is only one parameter that deals with that:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Subvolumes                | `--subvolumes`       | `-sv`   | Type of subvolumes, number of subvolumes and slicing axis when the case (x = 0, y = 1, z = 2). Accepts `slice`, `grid` and `voronoi` as subvolume types. | String Integer (Integer Integer) | `slice 10 0` |

The SVs are defined by the use of reference points, so that a particle passing by is considered to be contained in the SV with the nearest reference point. The defined SV are thus non-intersecting, and local quantities are calculated considering the particles that are contained in it. The definition of the reference points can be given to the `--subvolumes` parameters in three different ways:

- `slice` slices the domain i.e. divides the geometry using equidistant planes along a given axis. For example, to slice the domain 5 times along x axis:

    --subvolumes slice 5 0

- `grid` slices the domain in all three axis. For example, to generate a grid of 5 x 4 x 3 subvolumes (x, y and z, respectively):

    --subvolumes grid 5 4 3

- `voronoi` generates subvolumes by iteratively adjusting their positions so that it is more or less equilibrated. It is useful for complex geometries. For example, to generate 10 subvolumes without any position restriction:

    --subvolumes voronoi 10

The figure below shows how the system sets the voronoi subvolumes, in two dimensions for clarity. Samples are taken throughout the domain and initial reference points are randomly generated, in red. The samples are divided among SVs, and the centroid of each region is calculated (black dots). These centroids are then used as reference for the next iterations, and the process is repeated. The SV are defined when the reference points stop moving.

![](/readme_fig/voronoi.png)

The `voronoi` SV type allows for great flexibility but some unexpected effects can appear (such as two regions separated by a corner or a hole being connected by the same SV). Usually increasing the number of SV solves this issue.

### Initial conditions and particle temperature

The initial state of the domain is set by a temperature distribution that can be chosen by the user. The parameters responsible for that are:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Temperature distribution  | `--temp_dist`        | `-td`   | Shape of the initial temperature profile. Accepts `cold`, `hot`, `mean`, `linear`, `random`, `custom`. | String | `cold` |
| Subvolume temperature     | `--subvol_temp`      | `-st`   | Initial temperature of each subvolume when `custom` is informed in `-td`, in Kelvin. | Float or String | |
| Particle distribution     | `--part_dist`        | `-pd`   | Txt file with particle data to be imported. | String | | 

Particles are initialised in each SV with their Bose-Einstein occupation at the local temperature. The temperature is based on the imposed temperatures of the reservoirs. If there is no imposed temperature as BC, the only available option is `custom`, and the desired temperature for each SV should be informed to `--subvol_temp`.

A custom temperature distribution can be declared by giving numerical values for each subvolume or by passing a txt file with subvolume data from a previous simulation.

Another way that the initial condiction could be set is by importing the particle data from a previous simulation, in which case the initial temperature is automatically calculated from the energy of the particles.

> **Obs.:** If you import a particle data file, make sure that the set number of particles in the domain is the same, otherwise there will be a mismatch between the rate of particles leaving and entering the domain, which could lead to divergence.

### Particle temperature

In order to calculate phonon-phonon scattering, the temperature to which the particle is submitted needs to be defined. The particle can either assume the temperature of its subvolume or have it calculated by some type of interpolation:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Temperature interpolation | `--temp_interp`      | `-ti`   | How to interpolate the temperature between the subvolumes' reference points. Accepts `nearest` or `linear` (only for `slice` subvolumes are used). | String | `nearest` |

> **Obs.:** Currently only the `nearest` interpolation is accepted for non-sliced geometries. There is some work to do for 2D and 3D interpolations that should be done very soon.

### Duration and precision of the simulation

Some of the parameters directly affect the precision and the time of the simulation, or are conditions that limit the maximum runnning time:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| N° of particles           | `--particles`        | `-p`    | Number of particles given as `keyword number`. Can be given as the total number (keyworld `total`), the number per-mode-per-subvolume (keyworld `pmps`) and the number per cubic angstom (keyworld `pv`). | String Integer | `pmps 1` |
| Timestep                  | `--timestep`         | `-ts`   | Timestep of each iteration in picoseconds | Float | `1` |
| Iterations                | `--iterations`       | `-i`    | Number of iterations to be performed | Integer | `10000` |
| Number of datapoints      | `--n_mean`           | `-nm`   | Number of datapoints considered to calculated mean and standard deviation values. Each datapoint is 10 timesteps.| Int | `100` |
| Convergence criteria      | `--conv_crit`        | `-cc`   | Criteria for convergence and stop simulation. | Float | 0 1|
| Maximum simulation time   | `--max_sim_time`     | `-mt`   | Maximum time the simulation will run. Declared as `D-HH:MM:SS`. If the simulation arrives to this time, the calculation is finished and post processing is executed. If `0-00:00:00` is informed, no time limit is imposed. | String |`0-00:00:00`|


The number of particles and the timestep size determine how good your simulation results will be. Of course, the more particles and the smaller the timestep, the better. However, the two of them impact directly on the time needed for your calculation, so we have to make some compromises there. What can be changed depends a lot on the material you are simulating. Velocity, relaxation time, number of modes in the data, even geometry and boundary conditions, all that can influence how noisy and unstable is your calculation. Run tests for some 1000 iterations to check whether your parameters are good enough!!

Every 10 iterations, the global data is saved in the `convergence.txt` file. This includes time information, temperature, heat flux, thermal conductivity, number of particles and energy balance. Every 100 iterations, the code calculates the mean and standard deviations of these quantities over time. The desired number of datapoints (each datapoint saved 10 timesteps apart) to be considered for this mean and standard deviation is passed to `--n_mean`.

These mean values are also used to calculate de convergence criterion passed to `--conv_crit`. The standard is to not have any convergence criterion. If the user wishes, it can be set by passing the criteria to be used and the number of checks (that happen every 100 iterations) in which the maximum error should stay below the criterion. The error is calculated as:

$$ \epsilon = \max \Bigg(\bigg| \frac{\mu^{k}}{\mu^{k-1}}-1 \bigg| \Bigg) $$

where $\mu$ refers to each considered quantity (local temperature, local heat flux, thermal conductvity, etc.).

If you want to be sure your simulation won't be interrupted unexpectedly by your cluster or you do not want to leave it running all night because you underestimated the simulation time, you can set a time limit with `--max_sim_time`. When the code detects that the limit time was reached, it finishes the simulation and safely saves all result files at its current state.

### Figures and cosmetic parameters

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Real time plot            | `--rt_plot`          | `-rp`   | Property to plot particles to generate animations.  | String | |
| Figure plot               | `--fig_plot`         | `-fp`   | Property to plot particles at end of run (frequency, occupation, etc.) | Strings | `e` |
| Colormap                  | `--colormap`         | `-cm`   | Matplotlib [colormap](https://matplotlib.org/stable/gallery/color/colormap_reference.html) to be used in geometry plots (not convergence). | String | `jet` |
| Theme

#################### EDITING ############################

### Debugging parameters

These should only be used to detect possible errors in the simulation:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Particle distribution     | `--part_dist`        | `-pd`   | How to distribute particles at the beginning of the simulation. Composed of two keywords: `random/center_subvol/domain`. | String | `random_subvol` | -->
| Reservoir generation      | `--reservoir_gen`    | `-rg`   | How to generate particles on the reservoirs. With `constant`, particles are generated in a constant rate based in a timestep counter; `fixed_rate` means the particles are generated randomly, but in an approximately fixed rate; `one_to_one` means that the number of particles generated is the same of particles leaving in order to keep the number of particles stable. | String | `fixed_rate` |
| Empty subvols             | `--empty_subvols`    | `-es`   | Index of subvolumes that are to be initialised as empty (no particles). | Integer | |
| Energy normalisation      | `--energy_normal`    | `-en`   | The way to normalise energy to energy density. Choose between `fixed` (according to the expected number of particles in the subvolume) and `mean` (arithmetic mean of the particles inside). | String | `mean` |
| Reference temperature     | `--reference_temp`   | `-rt`   | The temperature at which the occupation number for every mode will be considered zero, in Kelvin. Alternatively, the user can set it as "local" to use the local temperature of each particle.| Float/String | `local` |
| Path points               | `--path_points`      | `-pp`   | Set the approximate points where the path to calculate $\kappa_{path}$ will go through. Declared the same way as `--bound_pos`.| String Float | `relative 0 0.5 0.5 1 0.5 0.5` |

<p>&nbsp</p>



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
