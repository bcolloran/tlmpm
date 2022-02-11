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

Basic knowledge of MPM will be assumed, and it might also help to be familiar with `mpm128.py`.

Below we describe the changes made between each version, starting with the file `v1_from_paper.py`

### `v1_from_paper.py`

This initial version departs from the particle-oriented MPM implementation from `mpm128.py`, in which the p2g and g2p phases "iterates" over particles, and has them scatter

## To run the demos

You'll need to `pip install taichi`; this code has been tested on `0.8.11` with CUDA. You may need install CUDA if you want to use GPU, I have not tried the non-CUDA backends.
