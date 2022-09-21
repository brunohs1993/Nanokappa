![](/readme_fig/logo_white.png#gh-light-mode-only)
![](/readme_fig/logo_black.png#gh-dark-mode-only)

# What is nano-k?

Nano-k is a Python code for phonon transport simulation. It allows to estimate the transport of heat in a given material by importing its properties derived from ab-initio calculations. It is possible to use standard geometries or import external ones.

# Setting the environment

It is recommended to run nano-k using a Conda environment. It can be set by following the steps:

1. Clone this repository or download its files to the desired folder;
2. Install Anaconda on your computer (https://www.anaconda.com/);
3. To set the `nanokappa` environment automatically, open `Nanokappa` folder on terminal and run (make sure Python version $\geq$ 3.5):

        python set_env/set_env.py

This should ensure that all necessary packages are installed either from conda repositories or via pip. A test simulation is run at the end to check if everything goes ok. It should take some minutes.

4. You can also set the environment manually. Create an environment (here called `nanokappa`, but it is an user's choice) and activate it:
   
       conda create -n nanokappa python=3.8
       conda activate nanokappa

5. Add conda-forge to the available channels:
   
       conda config --add channels conda-forge

6. Install the needed modules:
   
        conda install -n nanokappa --file <nanokappa-folder>/set_env/modules.txt

**Obs.**: Depending on the operating system, some modules may not be available on conda repositories. In this case, manually install from conda the available modules and try to install the remaining via pip by running `conda run -n nanokappa python -m pip install module1 module2 [module3 ...]`. This is done automatically by the `set_env.py` file mentioned on step 3.

7. Run a test by executing `python nanokappa.py -ff parameters_test.txt`. The resulting files should be located at `Nanokappa/test_results/test_X/`. These result files can be safely deleted after the test is finished.

<!-- **Obs.**: To install on the cluster:

        conda create -n nanokappa -c conda-forge python=3.8
        conda activate nanokappa
        conda install -c conda-forge h5py trimesh phonopy pyembree
        mkdir nanokappa
        cd nanokappa
        git clone https://github.com/brunohs1993/MultiscaleThermalCond
        conda install -c conda-forge ipython -->

# Running a simulation

## Simulation parameters

In order to define a study case and run a simulation, several parameters need to be set. Geometry, material, boundary conditions and calculation parameters need to be completely defined. These parameters are passed to the program as a pair <keyword, values> directly on command line.

Here is a list of all parameters that can be set:

| Parameter                | Keyword              | Reduced | Description                                                            | Types      | Default         |
| ------------------------ | -------------------- | --------| ---------------------------------------------------------------------- | ---------- | --------------- |
| Parameters file          | `--from_file`        | `-ff`   | File name with extension of a file containing all input parameters. Used to avoid inputing lots of parameters directly on terminal. Full path advised.       | String     |                 | 
| Material folder          | `--mat_folder`       | `-mf`   | Path of the folders containing the material files. Full path advisable. | String     | |
| hdf5 file                | `--hdf_file`         | `-hf`   | File names in each `-mf` with extensions.                                 | String     |                 | 
| POSCAR file              | `--poscar_file`      | `-pf`   | File names in each `-mf` with extensions.                                 | String     |                 |
| Material names           | `--mat_names`        | `-mn`   | Material name identifier.  | String | |
| Pickled materials        | `--pickled_mat`      | `-pm`   | Material indexes as declared on `-mf` to check whether it is already processed and pickled or not. If it is not, it will be pickled for use next time. Useful to avoid redoing phonon calculations for the same reference temperature. | Integer | |
| Results folder           | `--results_folder`   | `-rf`   | The name of the folder to be created containing all result files. If none is informed, no folder is created.                      | String     | `''`            |
| Results location         | `--results_location` | `-rl`   | The path where the result folder will be created. It accepts `local` if the results should be saved in the current directory, `main` if they should be saved in the same directory as `main_program.py`, or a custom path.               | String     | `local`            |
| Geometry                 | `--geometry`         | `-g`    | Standard geometry name or file name. Geometry coordinates in angstroms | String     | `cuboid`        |
| Dimensions               | `--dimensions`       | `-d`    | Dimensions for standard base geometries as asked by [trimesh.creation](https://trimsh.org/trimesh.creation.html) primitives | Floats | `20e3 1e3 1e3` |
| Scale                    | `--scale`            | `-s`    | Scale factors for the base geometry (x, y, z)                          | Float   x3 | `1 1 1`         |
| Geometry rotation        | `--geo_rotation`     | `-gr`   | Euler angles to rotate the base geometry (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | Float x3 | `0 0 0`         |
| Material rotation        | `--mat_rotation`     | `-mr`   | Euler angles to change crystal orientation (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | Float x3 | `0 0 0`         |
| Boundary conditions      | `--bound_cond`       | `-bc`   | Type of boundary condition for each facet declared in `--bound_facets`. If one condition more is given, it is considered to be the same for all non informed facets. Accepts `T` for temperature, `P` for periodic, `R` for roughness/reflection, `F` for heat flux. | String     | `T T P`      |
| Facets with imposed BC   | `--bound_facets`     | `-bf`   | The facets to imposed the boundary conditions on. | Integer | `0 3` | 
| Boundary condition values| `--bound_values`     | `-bv`   | Values for each imposed boundary conditions. Temperatures in Kelvin, heat fluxes in W/m$^2$, Roughness in angstroms.                 | Float   | `303 297`       |
| Connected faces          | `--connect_facets`   | `-cf`   | Indexes of the connected facets to apply the periodic boundary condition. They are grouped in pairs (first with second, third with fourth, and so on). They must: 1. Have vertices with the same coordinates in relation to their centroids; 2. Have the same area; 3. Have their normals parallel to each other and in opposite directions. | Integer | `1 5 2 4` |
| Collision offset         | `--offset`           | `-os`   | Offset of the collision detection from the wall to avoid errors in the reflection procedure. Usually unnecessary to change, but useful to have as an option. | Float | `2e-8` |
| Temperature distribution | `--temp_dist`        | `-td`   | Shape of the initial temperature profile. Accepts `cold`, `hot`, `mean`, `random`, `custom`.    | String     | `cold` |
| Subvolume temperature    | `--subvol_temp`      | `-st`   | Initial temperature of each subvolume when `custom` is informed in `-td`, in Kelvin. | Float | |
| Reference temperature    | `--reference_temp`   | `-rt`   | The temperature at which the occupation number for every mode will be considered zero, in Kelvin. | Float | `0` |
| N° of particles          | `--particles`        | `-p`    | Number of particles given as `keyword number`. Can be given as the total number (keyworld `total`), the number per-mode-per-subvolume (keyworld `pmps`) and the number per cubic angstom (keyworld `pv`).                               | String, Integer    | `pmps 1`             |
| Particle distribution    | `--part_dist`        | `-pd`   | How to distribute particles at the beginning of the simulation. Composed of two keywords: `random/center_subvol/domain`. | String | `random_subvol` |
| Reservoir generation     | `--reservoir_gen`    | `-rg`   | How to generate particles on the reservoirs. `fixed_rate` means the particles are generated in an approximately fixed rate independently of the leaving particles. `one_to_one` means that the number of particles generated is the same of particles leaving in order to keep the number of particles stable. | String | `fixed_rate` | 
| Timestep                 | `--timestep`         | `-ts`   | Timestep of each iteration in picoseconds                                  | Float      | `1`         |
| Iterations               | `--iterations`       | `-i`    | Number of iterations to be performed                                   | Integer    | `10000`         |
| Subvolumes               | `--subvolumes`       | `-sv`   | Type of subvolumes, number of subvolumes and slicing axis when the case (x = 0, y = 1, z = 2). Accepts `slice`, `grid` and `voronoi` as subvolume types.                | String Integer (Integer Integer) | `slice 10 0`          |
| Empty subvols            | `--empty_subvols`    | `-es`   | Index of subvolumes that are to be initialised as empty (no particles). | Integer |   |
| Subvol material          | `--subvol_material`  | `-sm`   | Material index of each subvolume, according to the order given at `-pf` and `-hf`. | Integer |  |
| Energy normalisation     | `--energy_normal`    | `-en`   | The way to normalise energy to energy density. Choose between `fixed` (according to the expected number of particles in the subvolume) and `mean` (arithmetic mean of the particles inside). | String | `fixed`
| Real time plot           | `--rt_plot`          | `-rp`   | Property to plot particles in real time (frequency, occupation, etc.)  | String     | `random`        |
| Figure plot              | `--fig_plot`         | `-fp`   | Property to plot particles at end of run (frequency, occupation, etc.) | Strings    | `T omega e`     |
| Colormap                 | `--colormap`         | `-cm`   | Colormap to use in every plot                                          | String     | `viridis`       |
| Convergence criteria     | `--conv_crit`        | `-cc`   | Criteria for convergence and stop simulation. | Float | 1e-6 |
| Use lookup table         | `--lookup`           | `-lu`   | Whether temperature should be calculated from particles or from the updated solution for the BTE (solution table).| Bool or Int| `False` or `0` | 


<p>&nbsp</p>

### The `--from_file` parameter

The user can input all parameters sequentially directly on terminal, which is easy when there are a few parameters that differ from the standard values. When there is a highly customised simulation to be ran, it is better to use an external input file and use the `--from_file` or `-ff` argument.

All inputs (besides `-ff`, of course) can be given in form of a text file. The following shows an example of the content of a txt file that we are calling `parameters.txt`:

    --mat_folder       D:\Materials\Si\
    --hdf_file         kappa-m313131.hdf5
    --poscar_file      POSCAR
    --pickled_mat      0
    --mat_names        silicon
    --geometry         cuboid
    --dimensions       10e3 1e3 1e3
    --scale            1 1 1
    --geo_rotation     0 0 0 xyz
    --subvolumes       slice 10 0
    --bound_facets     0 3 4 2
    --bound_cond       T T R R P
    --bound_values     302 298 10 10
    --connect_facets   1 5
    --offset           1e-3
    --reference_temp   300
    --energy_normal    fixed
    --temp_dist        constant_cold
    --empty_subvols    
    --subvol_temp      
    --particles        total 1e7
    --part_dist        random_subvol
    --timestep         1
    --iterations       2000
    --results_folder   box_wire
    --results_location D:\Results
    --conv_crit        0
    --lookup           0 0
    --colormap         jet
    --fig_plot         subvolumes
    --rt_plot          

This file defines a simulation of Silicon, with the material being defined by the material folder, hdf and poscar files being informed. The geometry is a box (cuboid) with dimensions 10e3 ang x 1e3 ang x 1e3 ang. Temperatures (302 K and 298 K) are imposed on facets 0 and 3, and a roughness is set to facets 2 and 4 as being 10 ang. Facets 1 and 5 are conected as periodic. A reference temperature of 300 K is used. The result files will be stored in `D:\Results\box_wire\`. Besides the usual convergence plots, the subvolumes plot will be generated at the beginning using the jet colormap.

Any non-necessary arguments can be left empty. In this example, `--subvol_temp` is left empty since the temperature profile is completely defined by the `constant_cold` keyword used as argument for `--temp_dist`. Since there is no inputs for `--rt_plot`, no real time plots will be produced. It is important to note, however, that passing an empty argument in the file will override the standard values and pass an empty list to the parser. If the user wishes to use standard input value for a given argument, the argument should be omitted in the txt file altogether.

The program could then be executed on terminal by calling:

    $ python nanokappa.py -ff <path-to-file>\parameters.txt

### Material properties

The properties describing the material are derived from hdf5 and POSCAR files. These files needed to be informed with extensions. They should both be in the folder informed in full to the `--mat_folder` argument. The files with material properties are the only mandatory parameters to be informed. All others have standard values associated for quick tests, as seen on the table above. The material can also be rotated by passing a set of angles and rotation order to `--mat_rotation`.

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

These standard geometries can be modified by entering the parameters `--scale` and `--geo_rotation`. For example, a box with L_x = 100, L_y = 100 and L_z = 200 (all lengths in angstroms), there are two ways to enter parameters.

1. By directly inputing the dimensions...

        --geometry   cuboid
        --dimensions 100 100 200

2. ...or by inputing any dimensions and scaling it by x, y and z factors, such as:
   
        --geometry   cuboid
        --dimensions 100 100 100
        --scale      1 1 2

The `--dimensions` arguments are required only for standard geometries. Imported geometries ignore this command. The inputs `--scale` and `--geo_rotation`, however, work with imported geometries the same way as with the standard ones.

Whatever is the geometry, it is always rezeroed so that all vertices are in the all positive quadrant (every x, y and z coordinate is greater than or equal to zero).

### Boundary conditions

The boundary conditions (BC) consist of heat transfer restrictions (such as imposed temperatures) and boundary surfaces properties (roughness or periodic). The user sets the desired facets on which to apply the BC by passing their indices to `--bound_facets`, the BC type to `--bound_cond` and their respective values to `--bound_values`. For instance, in the previously given example we have:

    --bound_facets     0 3 4 2
    --bound_cond       T T R R P
    --bound_values     302 298 10 10
    --connect_facets   1 5

So, in order:

- Facet 0 has an imposed temperature of 302 K;
- Facet 3 has an imposed temperature of 298 K;
- Facets 4 and 2 both have a roughness of 10 angstroms;
- Facets 1 and 5 are not mentioned, hence they pick the last informed boundary condition, which is periodic. To complete, their connection needs to be informed to `--connect_facets`.

The `periodic` BC can only be applied to facets that have vertices in the same relative position, and their normal must be parallel and pointing in opposite directions. This is of vital importance, since crystal orientation is relevant to the phonon properties: an interface between crystals of the same material but in different orientations (such as grain boundaries) _cannot_ be periodic. In this configuration, whenever a particle crosses a boundary (that is not with a fixed temperature or heat flux) it is transported to the same position as it entered on the opposite boundary, as the solid was composed by several cuboid domains side by side (hence, periodic).

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

![](readme_fig\voronoi.png)

This algorithm has shown to be flexible, but can cause some problems depending of the complexity of the geometry and of the initial $x_r$. Geometries with indents or holes, for example, can lead to particles on each side of the gap to be considered in the same subvolume, which can lead to an unreal energy transfer through the space. This sometimes can be avoided by just rerunning the simulation, but usually better results can be achieved by increasing the number of subvolumes, so that each side of the empty space is classified as a different subvolume. It is important to have user discretion while applying this type of subvolume.

### Result files

The user can specify the name of a folder to save all simulation results into by inputing it to the `--results_folder` parameter. The default is no folder (an empty string `''`), so all files are saved in the folder where the user is located.


### Future enhancements

Ideas for next developments are welcome and will be developed as time allows it. Some of our ideas for next implementations include:

- Convergence detection;
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

<!-- XXXXXXXX KEEP GOING xxxxxxxxxxxxxxxx -->
