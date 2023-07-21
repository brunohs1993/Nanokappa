![](/readme_fig/logo_white.png#gh-light-mode-only)
![](/readme_fig/logo_black.png#gh-dark-mode-only)

# What is this file?

This document explains each of the simulation parameters that the user can declare. We will go in detail about what they mean and how to pass them to Nano-&#954;, and we will execute the simulation at the end and check the results.

# Simulation parameters

In order to define a study case and run a simulation, several parameters need to be set. Geometry, material data, boundary conditions and calculation parameters need to be completely defined. These parameters are passed to the program as a pair <keyword, values> directly on command line.

## The `--from_file` parameter

The user can input all parameters sequentially directly on terminal, which is easy when there are just a few parameters. However it is often not the case, and it is usually better to use an external input file and use the `--from_file` or `-ff` argument. You can save the parameters on a txt file written just like you would do on command line, but separating each parameter in a new line. Then you could run:

    python nanokappa.py -ff <parameter_file>.txt

## Where to save the results

The location of the results is set with one argument:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Results folder            | `--results_folder`   | `-rf`   | The path and the name of the folder to be created containing all result files. If none is informed, no folder is created and files are saved in the working directory. | String | `''` |

Relative and absolute paths are accepted, with relative being searched from the current working directory. If the folder does not exist, all folders in the given path are created.

## Geometrical parameters

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

The `--dimensions` arguments are required only for standard geometries. Imported geometries ignore this command. The inputs `--scale` and `--geo_rotation`, however, work with imported geometries the same way as with the standard ones. It is important to note that the scaling of the geometry comes _before_ its rotation.

Whatever is the geometry, it is always rezeroed so that all vertices are in the all positive quadrant (every x, y and z coordinate is greater than or equal to zero).

In summary:

| Parameter                 | Keyword              | Reduced | Description | Types | Default |
| ------------------------- | -------------------- | ------- | ----------- | ----- | ------- |
| Geometry                  | `--geometry`         | `-g`    | Standard geometry name or file name. Geometry coordinates in angstroms. | String | `cuboid` |
| Dimensions                | `--dimensions`       | `-d`    | Dimensions for the standard base geometries (box, cylinder, etc.). | Floats | `20e3 1e3 1e3` |
| Scale                     | `--scale`            | `-s`    | Scale factors for the base geometry (x, y, z) | Float x3 | `1 1 1` |
| Geometry rotation         | `--geo_rotation`     | `-gr`   | Euler angles to rotate the base geometry and which axis to apply (see [scipy.rotation.from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)) | Float, String | |

## Material parameters

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

## Boundary conditions

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

## Subvolumes

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

## Initial conditions and particle temperature

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

## Duration and precision of the simulation

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

## Figures and cosmetic parameters

As the simulation runs, the code generates plots so that we can graphically assess the results. The parameters that control them are:

| Parameter      | Keyword      | Reduced | Description | Types | Default |
| -------------- | ------------ | ------- | ----------- | ----- | ------- |
| Real time plot | `--rt_plot`  | `-rp`   | Property to plot particles to generate animations.  | String | |
| Figure plot    | `--fig_plot` | `-fp`   | Property to plot particles at end of run (frequency, occupation, etc.) | Strings | |
| Colormap       | `--colormap` | `-cm`   | Matplotlib [colormap](https://matplotlib.org/stable/gallery/color/colormap_reference.html) to be used in geometry plots (not convergence). | String | `jet` |
| Theme          | `--theme`    | `-th`   | Theme used for the plots. Choose among 'white', 'light', 'dark' or 'black'. | String | 'white' |

It would be nice to generate plots with a theme similar to that you're using here right now, so it will show `white` theme if you are on a light coloured page, or `dark` theme if you are in a dark coloured page. Both will use the `jet` colormap, and we want to plot the particles according to their energy. So we are going to add to our parameters:

    --fig_plot energy
    --colormap jet
    --theme    white OR dark

# Setting the parameter file

Let's say we want to simulate the heat transfer in a thin film, in the in-plane direction, like we talked about in the boundary conditions' section. The film is 100 nm thick, with a surface roughness of 1 nm. The $\Delta T$ is applied between 1 &#956;m distance. We declare the material file, the box dimensions and the boundary conditions. We're going to divide the domain in 20 slices, starting with a cold temperature. The temperature of each particle will be interpolated linearly between subvolumes. For a quick demonstration, we are going to use only 1e5 particles in timesteps of 1ps, for 5000 iterations. We also set the plots as we just discussed.

    --mat_folder     <path_to_nanokappa>/test_material/
    --hdf_file       kappa-m313131.hdf5
    --poscar_file    POSCAR
    --geometry       box
    --dimensions     1e4 1e4 1e3
    --bound_pos      relative 0 0.5 0.5 1 0.5 0.5 0.5 0.5 0 0.5 0.5 1
    --bound_cond     T T R R P
    --bound_values   302 298 10 10
    --connect_pos    absolute 5e3 0 5e2 5e3 1e3 5e2
    --subvolumes     slice 20 0
    --temp_dist      cold
    --temp_interp    linear
    --timestep       1
    --particles      total 1e5
    --fig_plot       energy
    --colormap       jet
    --theme          <white OR dark>
    --results_folder <folder_name>

# Checking the results

TODO


