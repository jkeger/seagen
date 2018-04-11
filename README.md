# SEAGen

A work-in-progress python implementation of the stretched-equal-area (SEA)
algorithm for generating spherically symmetric arrangements of particles for
SPH initial conditions (or any particle distributions) with accurate densities,
as presented in Kegerreis et al. (2018), *in prep*.

Jacob Kegerreis and Josh Borrow

jacob.kegerreis@durham.ac.uk

## Requirements:

+ `python3.6.0` or above
+ `tqdm`, `matplotlib`, `scipy`, `numpy`, `typing`, `numba`
    (available through `pip3 install -r requirements.txt`)

Currently this package is not on PyPI but we plan to release it there once
completed.

## Current Status

+ All basic functionality now implemeneted. i.e. generating single shells and
  nested shells to build a sphere, following a density profile that can include
  multiple layers.
+ Significant minor features still needed, especially regarding user input and
  options control.
+ Lots of code tidying and documentation updating to do!

## Contributing

Pull requests are more than welcome.

