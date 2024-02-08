![](/readme_fig/logo_white.png#gh-light-mode-only)
![](/readme_fig/logo_black.png#gh-dark-mode-only)

# What is Nano-&#954;?

Nano-&#954; is a Python code for phonon transport simulation, combining Monte Carlo method with ab-initio material data. The user can simulate preset, standard geometries or import external ones in STL format.

# Installation

## Clone the Nano-&#954; github repository to the desired folder:

```bash
$ git clone https://github.com/brunohs1993/Nanokappa.git
```

## Setting the environment

It is recommended to run Nano-&#954; using a virtual environment, either with Conda or Pip.

<details>
<summary> <font size="4"><b>Using Conda</b></font> </summary>

Firstly, install [Anaconda](https://www.anaconda.com/) on your computer.

Create an environment and activate it:

```bash   
$ conda create -n nanokappa
$ conda activate nanokappa
```

The `(nanokappa)` word will appear on the command line, signaling the environment is active.

Add conda-forge to the available channels:

```bash
(nanokappa) $ conda config --add channels conda-forge
```

Install the requirements:

```bash
(nanokappa) $ conda install -n nanokappa --file <path_to_nanokappa>/Nanokappa/set_env/requirements.txt
```

</details>

<br/>

<details>
    <summary> <font size="4"><b>Using Pip</b></font> </summary>

You need `virtualenv` installed. Under Debian you can use:

```bash
$ sudo apt install python3-virtualenv
```

Create and load the environment:

```bash
$ python3 -m virtualenv ~/envs/nanokappa
$ source ~/envs/nanokappa/bin/activate
```

The `(nanokappa)` word will appear on the command line, signaling the environment is active.

Install Nano-&#954;'s requirements:

```bash
(nanokappa) $ cd nanokappa
(nanokappa) $ pip install -r <path_to_nanokappa>/Nanokappa/set_env/requirements.txt
```

</details>

<br/>

## Run a test

Run the following command to run a test calculation to ensure everything is running smoothly:

```bash
(nanokappa) $ cd <path_to_nanokappa>/Nanokappa/
(nanokappa) $ python3 nanokappa.py -ff parameters_test.txt
```

The calculation outputs will be located in `<path_to_nanokappa>/Nanokappa/test_results/test_0/`.

To deactivate the virtual environment, type `deactivate` for pip or `conda deactivate` for Conda and press Enter.

# Running a calculation

Please, refer to our [how-to guide](/tutorials/howto.md) for a more detailed description of each of the simulation parameters and how to declare them.

The easiest case to simulate is a heat transfer in a thin film in the crossplane direction. For that we can use the material data offered as sample. The parameters could be listed in a `parameters.txt` file as:

    --mat_folder     <path_to_nanokappa>/test_material/Si/
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

```bash
$ python <path_to_nanokappa>/nanokappa.py -ff <path>/parameters.txt
```

## What Nano-&#954; shows as result of a simulation?

After (and during) a simulation, you will find in the results folder:

- An `arguments.txt` file that is generated containing all arguments used in the simulation. You can use it directly to rerun the same case if needed.
- A `convergence.txt` file with convergence data every 10 iterations.
- Files describing the final state of the simulation: `particle_data.txt`, `subvolumes.txt` and `subvol_connections.txt`;
- Convergence plots: temperature, particles, heat flux, energy balance, thermal conductivity;
- Geometry plots: BCs, SVs, SV connection conductivity, particle scatter plots;
- Plot of thermal conductivity contribution by frequency;
<!-- - Output file, when `--output file` is used; -->

# How to cite?

We ask for everyone that uses Nano-&#954; to cite our work. Nano-&#954; was published in the following paper with citations in BibTeX:

        @article{2024-Nanokappa,
        title = {Monte Carlo simulation of phonon transport from ab-initio data with Nano-κ},
        journal = {Computer Physics Communications},
        volume = {294},
        pages = {108954},
        year = {2024},
        issn = {0010-4655},
        doi = {https://doi.org/10.1016/j.cpc.2023.108954},
        url = {https://www.sciencedirect.com/science/article/pii/S0010465523002990},
        author = {B.H. Silva and D. Lacroix and M. Isaiev and L. Chaput},
        keywords = {Monte Carlo, Phonon transport, Ab-initio, Nanoscale heat transfer, Thermal conductivity}
        }


# What is planned for the future?

There is A LOT of features that we want to add, some more urgently, others less:

- On the computational part, we intend to improve calculation speed, write a comprehensible documentation and standardise the code.
- On the scientific side, we want to add more boundary conditions options and improve mathematical models, and include the possibility of simulating more types of materials.

We want the user to always have more flexibility, prettier visuals, faster simulations and better results.

# "I like it! What can I do?"

- If you are a researcher, feel free to use Nano-&#954;, as long as you cite us;
  
- If you find any bugs, please report on Nano-&#954;'s repository or send us an email so we can fix it as quick as possible;
  
- If you think you could contribute to Nano-&#954;'s development, please send us a message and we could have a coffee and talk about how.

# Aknowledgements

The first version of this code was written by Bruno Hartmann da Silva as part of his PhD, under supervision of Prof. Laurent Chaput and Prof. David Lacroix at the Laboratory of Energy and Theorethical and Applied Mechanics (LEMTA), Lorraine University, Vandoeuvre-lès-Nancy, France. The PhD was funded by the French National Research Agency (ANR).
