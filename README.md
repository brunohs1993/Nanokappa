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
   
       conda create -n mcphonon python=3.8
       conda activate mcphonon

4. Add conda-forge to the available channels:
   
       conda config --add channels conda-forge

5. Install the needed modules:
   
        conda install --file <mcphonon-folder>/set_env/modules.txt

**Obs.**: Sometimes this process may take excessive time or raise errors. If you find any problems with the Python version during the environment setup, it may work by creating the environment initially with whatever Python version is installed (typing only `conda create -n mcphonon`), activating it and installing all modules listed in `modules.txt`, then downgrading the environment to Python 3.6 by typing `conda -n mcphonon install python=3.6`. Anaconda will take care of the dependencies while downgrading.

**Obs.**: To install on the cluster:

        conda create -n mcphonon -c conda-forge python=3.8
        conda activate mcphonon
        conda install -c conda-forge h5py trimesh phonopy pyembree
        mkdir mcphonon
        cd mcphonon
        git clone https://github.com/brunohs1993/MultiscaleThermalCond
        conda install -c conda-forge ipython

# Running a simulation

## Simulation parameters

| Parameter                | Command            | Reduced | Description                                                            | Types      | Default         |
| ------------------------ | ------------------ | --------| ---------------------------------------------------------------------- | ---------- | --------------- |
| Parameters file    | `--from_file`      | `-ff`   | Full file name with path and extension of a file containing all input parameters. Used to avoid inputing lots of parameters directly on terminal.        | String     |                 | 
| hdf5 file                | `--hdf_file`       | `-hf`   | Full file name with path and extension                                 | String     |                 | 
| POSCAR file              | `--poscar_file`    | `-pf`   | Full file name with path and extension                                 | String     |                 |
| Results folder           | `--results_folder` | `-rf`   | The name of the folder to be created containing all result files. If none is informed, no folder is created.                      | String     | `''`            |
| Results location           | `--results_location` | `-rl`   | The path where the result folder will be created. It accepts `local` if the results should be saved in the current directory, `main` if they should be saved in the same directory as `main_program.py`, or a custom path.               | String     | `local`            |
| Geometry                 | `--geometry`       | `-g`    | Standard geometry name or file name. Geometry coordinates in angstroms | String     | `cuboid`        |
| Dimensions               | `--dimensions`     | `-d`    | Dimensions for standard base geometries as asked by [trimesh.creation](https://trimsh.org/trimesh.creation.html) primitives | Floats | `20e3 1e3 1e3` |
| Scale                    | `--scale`          | `-s`    | Scale factors for the base geometry (x, y, z)                          | Float   x3 | `1 1 1`         |
| Rotation angles          | `--rotation`       | `-r`    | Euler angles to rotate the base geometry (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | Float x3 | `0 0 0`         |
| Rotation order           | `--rot_order`      | `-ro`   | Order of the Euler angles informed in `--rotation` (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | String | `xyz`         |
| Boundary conditions      | `--bound_cond`     | `-bc`   | Type of boundary condition for each facet declared in `--bound_facets`. If one condition more is given, it is considered to be the same for all non informed facets. Accepts `T` for temperature, `P` for periodic, `R` for roughness/reflection, `F` for heat flux. | String     | `T T P`      |
| Facets with imposed BC   | `--bound_facets`   | `-bf`   | The facets to imposed the boundary conditions on. | Integer | `0 3` | 
| Boundary condition values | `--bound_values`   | `-bv`    | Values for each imposed boundary conditions. Temperatures in Kelvin, heat fluxes in W/m$^2$, Roughness in angstroms.                 | Float   | `303 297`       |
| Connected faces | `--connect_facets` | `-cf` | Indexes of the connected facets to apply the periodic boundary condition. They are grouped in pairs (first with second, third with fourth, and so on). They must: 1. Have vertices with the same coordinates in relation to their centroids; 2. Have the same area; 3. Have their normals parallel to each other and in opposite directions. | Integer | `1 5 2 4` |
| Temperature distribution | `--temp_dist`      | `-td`   | Shape of the initial temperature profile. Accepts `constant_cold`, `constant_hot`, `mean`, `random`, `custom`.    | String     | `constant_cold` |
|Subvolume temperature | `--subvol_temp` | `-st` | Initial temperature of each subvolume when `custom` is informed in `--temp_dist`, in Kelvin. | Float | |
| Reference temperature | `--reference_temp` | `-rt` | The temperature at which the occupation number for every mode will be considered zero, in Kelvin. | Float | `0` |
| N° of particles          | `--particles`      | `-p`    | Number of particles given as `keyword number`. Can be given as the total number (keyworld `total`), the number per-mode-per-subvolume (keyworld `pmps`) and the number per cubic angstom (keyworld `pv`).                               | String, Integer    | `pmps 1`             |
| Particle distribution    | `--part_dist`  | `-pd`    | How to distribute particles at the beginning of the simulation. Composed of two keywords: `random/center_subvol/domain`. | String | `random_subvol`
| Timestep                 | `--timestep`       | `-ts`   | Timestep of each iteration in picoseconds                                  | Float      | `1`         |
| Iterations               | `--iterations`     | `-i`    | Number of iterations to be performed                                   | Integer    | `10000`         |
| Subvolumes               | `--subvols`         | `-sv`   | Type and number of subvolumes to measure local energy density. Accepts `slice`, `box` and `sphere`. When type is `slice`, it is needed to provide the axis of slicing (`0` for x, `1` for y, `2` for z).                | String Integer (Integer) | `slice 10 0`          |
| Empty subvols | `--empty_subvols` | `-es` | Index of subvolumes that are to be initialised as empty (no particles). | Integer |   |
| Energy normalisation | `--energy_normal` | `-en` | The way to normalise energy to energy density. Choose between `fixed` (according to the expected number of particles in the subvolume) and `mean` (arithmetic mean of the particles inside). | String | `fixed`
| Real time plot           | `--rt_plot`        | `-rp`   | Property to plot particles in real time (frequency, occupation, etc.)  | String     | `random`        |
| Figure plot              | `--fig_plot`       | `-fp`   | Property to plot particles at end of run (frequency, occupation, etc.) | Strings    | `T omega e`     |
| Colormap                 | `--colormap`       | `-cm`   | Colormap to use in every plot                                          | String     | `viridis`       |


<p>&nbsp</p>

### `--from_file` parameter

The user can input all parameters sequentially directly on terminal, which is easy when there are a few parameters that differ from the standard values. When there is a highly customised simulation to be ran, it is better to use an external input file and use the `--from_file` or `-ff` argument.

All inputs (besides `-ff`, of course) can be given in form of a text file. The following shows an example that we are calling `parameters.txt`:

    --hdf_file         D:\Materials\Si\kappa-m313131.hdf5
    --poscar_file      D:\Materials\Si\POSCAR
    --geometry         D:\Geometries\film_oblique_45.obj
    --subvolumes       slice 10 0
    --bound_facets     0 5
    --bound_cond       T T P
    --bound_values     303 297
    --connect_facets   1 4 2 3
    --reference_temp   300
    --energy_normal    fixed
    --temp_dist        constant_cold
    --particles        pmps 1
    --part_dist        random_subvol
    --timestep         1
    --iterations       1000
    --results_folder   sim_results
    --results_location D:\Results

This file defines a simulation of Silicon, with the material being defined by the hdf and poscar files being informed. The geometry is imported from an external file (`film_oblique_45.obj`), and consequently the indexes of the facets are changed. Temperatures (303 K and 297 K) are imposed on facets 0 and 5, and a periodic boundary condition connects facets 1 to 4 and 2 to 3. A reference temperature of 300 K is used.
The program could then be executed on terminal by calling:

    $ python main_program.py -ff parameters.txt

### Material properties

The properties describing the material are derived from hdf5 and POSCAR files. These files needed to be informed in full (with path and extensions). For example the command:

    $ python main_program.py -hf <hdf5-file>.hdf5 -pf <poscar-file>

runs a simulation with materials properties of silicon, saved in the folder `/materials/Si/`, retrieved from the hdf5 and POSCAR files informed. All other parameters are standard.

The files with material properties are the only mandatory parameters to be informed. All others have standard values associated for quick tests, as seen on the table above.

### Geometry definition

The geometry can be defined from an standard geometry or an external file. The standard geometries are defined by functions in [trimesh.creation](https://trimsh.org/trimesh.creation.html):

| Geometry | Key for `--geometry` | [trimesh.creation](https://trimsh.org/trimesh.creation.html) function | Parameters for `--dimensions`         | Obs               |
| -------- | -------------------- | --------------------------------------------------------------------- | ------------------------------------- | ----------------- |
| Cuboid   | `cuboid`             | `trimesh.creation.box`                                                | x, y and z lengths                    |                   |
| Sphere   | `sphere`             | `trimesh.creation.icosphere`                                          | Radius and subdivisions               | To be implemented |
| Cylinder | `cylinder`           | `trimesh.creation.cylinder`                                           | Radius and height                     | To be implemented |
| Cone     | `cone`               | `trimesh.creation.cone`                                               | Radius and height                     | To be implemented |
| Annulus  | `annulus`            | `trimseh.creation.annulus`                                            | Inner radius, outer radius and height | To be implemented |

<p>&nbsp</p>

These standard geometries can be modified by entering the parameters `--scale`, `--rotation` and `--rot_order`. For example, a box with L_x = 100, L_y = 100 and L_z = 200 (all lengths in angstroms), there are two ways to enter parameters.

1. By directly inputing the dimensions...

        $ python main_program.py -hf <hdf5-file>.hdf5 -pf <poscar-file> -g cuboid -d 100 100 200

2. ...or by inputing any dimensions and scaling it by x, y and z factors, such as:

        $ python main_program.py -hf <hdf5-file>.hdf5 -pf <poscar-file> -g cuboid -d 100 100 100 -s 1 1 2

The `--dimensions` arguments are required only for standard geometries. Imported geometries ignore this command. The inputs `--scale`, `--rotation` and `--rot_order`, however, work with imported geometries the same way as with the standard ones.

Whatever is the geometry, it is always rebased so that all coordinates are positive.

### Boundary conditions

The boundary conditions consist of heat transfer restrictions (such as imposed temperatures or heat fluxes) and boundary surfaces properties (specular, diffuse, periodic). The user sets



The `periodic` boundary condition can only be applied to the `cuboid` geometry. In this configuration, whenever a particle crosses a boundary (that is not with a fixed temperature or heat flux) it is transported to the same position as it entered on the opposite boundary, as the solid was composed by several cuboid domains side by side (hence, periodic).

### Slices

The slices are subdivisions of the domain so it is possible to calculate local quantities. The domain is subdivided in `N` equal parts (slices) along a given axis, and local quantities are calculated from the phonons located inside that slice. To declare the number of slices and the direction along which to slice, 

### Result files

The user can specify the name of a folder to save all simulation results into by inputing it to the `--results_folder` parameter. The default is no folder (an empty string `''`), so all files are saved in the folder where the user is located.

<!-- XXXXXXXX KEEP GOING xxxxxxxxxxxxxxxx -->