# 2d TLMPM implemented with Taichi

This repo presents a didactic implementation of the Total Langranian Material Point Method (TLMPM) as described in the papers:

- "A Total-Lagrangian Material Point Method for solid mechanics problems
  involving large deformation", (DOI: 10.1016/j.cma.2019.112783), and
- "Modelling contacts with a total Lagrangian material point method" (DOI: 10.13140/RG.2.2.10187.62245). **Note:** in this repo, we'll use the shorthand name "TLMPM Contacts" to refer to this paper

(These papers can be found in the `papers` folder of this repo.)

The objective is to present a simple, step-by-step series of improvements to the TLMPM implementation beginning from a very literal translation of Algorithm 2 from "TLMPM Contacts" based on [the famous `mpm128.py` implementation](https://www.youtube.com/watch?v=9M18rc9-VWU) of MLS-MPM provided by Yuanming Hu and the Taichi team, and ending with a much more performant version.

Ideally, in the first step you should be able to see how the algorithm from "TLMPM Contacts" maps to the code, and even though the final version we end up with will look quite different, since we've gone step-by-step making small changes along the way, you'll hopefully be able to understand and reconstruct why the final code looks the way it does.

## How to read this repo

To follow along with the implementation improvements, it's recommended that you clone/download this repo to your local machine and then use you favorite text editor to compare the different TLMPM implementations in the folder `TLMPM_perf_optimization`. The files are all named starting with a version number, e.g. `v{n}_*.py`; in each case, the changes between `v{n}_*.py` and `v{n+1}_*.py` is meant to be one discrete set of improvements that should be easy to follow and understand.

(Note: to compare versions in VSCode, click on `v{n}_*.py` and then control+click `v{n+1}_*.py`, then right click and select "Compare selected".)

These files follow the strategy I took when implementing TLMPM: start with the working code in `mpm128.py`, and adapt it to Algorithm 2 in "TLMPM Contacts", then make incremental improvements from there. This is a little awkward at first, because these algorithms are different enough that shoehorning TLMPM into the `mpm128.py` framework resulted in some code that is not what you would want if you we're approaching TLMPM from scratch, but I as a relatively new user of Taichi, it was helpful to have this scaffolding.

Below we describe the changes made between each version, starting with the file `v0_mpm128_jelly_spinning_beam.py`.

### `v0_mpm128_jelly_spinning_beam.py`

In This first file, we make a numer of changes to `mpm128.py` to give create a baseline MLS-MPM scenario that will allow us to track the progress of our TLMPM implementation:

- because TLMPM is best suited for solids that don't undergo plastic deformation, the water and snow materials are not really appropriate so we use only the "jelly" material;
- because the handling of conatact is a bit more involved in TLMPM than MLS-MPM, we want a scenario in which the object simulated won't have to bounce off of the bounds of teh simulation area, so we initialize the particles as a spinning beam;
- because we want to be able to be able to use exactly the same number of particles in our MLS-MPM and TLMPM simulations, we initialize the particles on a Cartesian lattice rather than randomly.

### `v1_from_paper.py`

This initial version departs from the particle-oriented MPM implementation from `v0_mpm128_jelly_spinning_beam.py`, in which the p2g and g2p phases "iterates" over particles; in p2g particles scatter information to the grid (using atomic adds), and in g2p they gather from the grid.However, there are a lot of code changes between `v0_mpm128_jelly_spinning_beam.py` and `v1_from_paper.py` (so a direct diff between them may not be super illuminating at this point).

For this file, I'd recommend reviewing Algorithm 2 in "TLMPM Contacts"; the file is heavily annotated with comments indicating exactly which line of code maps to which line in the algorithm from the paper.

### `v2_grid_of_particles_in_config_space.py`

Comparing this version to the previous one, it will initially look like a lot of changes have been made. However, in actuality there is really only one change: since the positions of particles are fixed in configuration space in TLMPM, and moreover since they are fixed on a cartesian lattice with a fixed relationship to the MPM background grid (4 particles per grid cell), it makes sense to store the particles in a 2d field and access them by their index in the lattice. The actual position in configuration space can be easily calculated with the index in the particle field and the spacing of the background grid.

This also substantially simplifies some of the details regarding which grid nodes each particle maps to, allowing us to easily use a background grid that is sized specifically for the initial particle distribution in configuration space. This dramatically reduces the amount of memory needed for the grid and the amount of computation required for kernels that operate on the entire grid.

Note: starting with this file we always use the notation _(i,j)_ to refer to indices of the MPM background grid, and _(f,g)_ to refer to indices of particles in configuration space.

### `v3_particle_grid_to_cell_grid_accessors.py`

This version makes one very small change. The `mpm128.py`, there is the concept of the grid `base` that corresponds to each particle; this is the bottom-left-most grid node of the 3x3 range of nodes that need to be iterated over for each particle during p2g and g2p. For TLMPM, this is not really needed because, since particles have a fixed position in configuration space and can be store on a fixed grid, the `base` need not be calculated from the configuration space position, it can calculated using the indices od the particle in the field of particles.

### `v1_from_paper.py`

### `v1_from_paper.py`

### `v1_from_paper.py`

## To run the demos

You'll need to `pip install taichi`; this code has been tested on `0.8.11` with CUDA. You may need install CUDA if you want to use GPU, I have not tried the non-CUDA backends.
