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
> This procedure is done automatically by the `set_env.py` when running the automatic installation.

<!-- **Obs.**: To install on the cluster:

        conda create -n nanokappa -c conda-forge python=3.8
        conda activate nanokappa
        conda install -c conda-forge h5py trimesh phonopy pyembree
        mkdir nanokappa
        cd nanokappa
        git clone https://github.com/brunohs1993/Nanokappa
        conda install -c conda-forge ipython -->

# Running a calculation

Please, refer to our [how-to guide](/tutorials/howto.md) for a more detailed description of each of the simulation parameters and how to declare them.

The easiest case to simulate is a heat transfer in a thin film in the crossplane direction. For that we can use the material data offered as sample. The parameters could be listed in a `parameters.txt` file as:

    --mat_folder     <path_to_nanokappa>/test_material/
    --hdf_file       kappa-m313131.hdf5
    --poscar_file    POSCAR
    --geometry       box
    --dimensions     20e3 20e3 20e3
    --bound_pos      relative 0 0.5 0.5 1 0.5 0.5
    --bound_cond     T T P
    --bound_values   302 298
    --connect_pos    relative 0.5 0 0.5 0.5 1 0.5 0.5 0.5 0 0.5 0.5 1
    --subvolumes     slice 20 0
    --particles      total 1e6
    --timestep       1
    --iterations     10000
    --results_folder test

and run on command line with:

    python <path_to_nanokappa>/nanokappa.py -ff <path>/parameters.txt

## What Nano-&#954; shows as result of a simulation?

After (and during) a simulation, you will find in the results folder:

- An `arguments.txt` file that is generated containing all arguments used in the simulation. You can use it directly to rerun the same case if needed.
- A `convergence.txt` file with convergence data every 10 iterations.
- Files describing the final state of the simulation: `particle_data.txt`, `subvolumes.txt` and `subvol_connections.txt`;
- Convergence plots: temperature, particles, heat flux, energy balance, thermal conductivity;
- Geometry plots: BCs, SVs, SV connection conductivity, particle scatter plots;
- Plot of thermal conductivity contribution by frequency;
- Output file, when `--output file` is used;

## Parallel computing

Currently Nano-&#954; does not support parallel computing of a single case. It does support however parallel computations of multiple cases. To do it you just need to list the parameters in a csv file. More detail is given in the [parallel computation guide](tutorials/parallel.md).

# How to cite?

The scientific paper describing Nano-&#954; is currently being reviewed. The citation to it in bibtex format will be here when it becomes published.

# What is planned for the future?

There is A LOT of features that we want to add, some more urgently, others less:

- On the computational part, we intend to improve calculation speed, write a comprehensible documentation and standardise the code.
- On the scientific side, we want to add more boundary conditions options and improve mathematical models, and include the possibility of simulating more types of materials.

We want the user to always have more flexibility, prettier visuals, faster simulations and better results.

# "I like it! What can I do?"

If you are a researcher, feel free to use Nano-&#954;, as long as you cite us. If you think you could contribute to its development, please send us a message and we could have a coffee and talk about how.

# Aknowledgements

The first version of this code was written by Bruno Hartmann da Silva as part of his PhD, under supervision of Prof. Laurent Chaput and Prof. David Lacroix at the Laboratory of Energy and Theorethical and Applied Mechanics (LEMTA), Lorraine University, Vandoeuvre-lès-Nancy, France. The PhD was funded by the French National Research Agency (ANR).
