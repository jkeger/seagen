# SEAGen

A python implementation of the stretched-equal-area (SEA) algorithm for
generating spherically symmetric arrangements of particles with accurate
particle densities, e.g. for SPH initial conditions that precisely match an
arbitrary density profile, as presented in Kegerreis et al. (2018), *in prep*.

Jacob Kegerreis and Josh Borrow

jacob.kegerreis@durham.ac.uk

Visit `https://github.com/jkeger/seagen` to download the code including example
scripts.

## Requirements:

+ `python3.6.0` or above
+ `tqdm`, `matplotlib`, `scipy`, `numpy`, `typing`, `numba`
    (available through `pip3 install -r requirements.txt`)

## Current Status

+ All basic functionality now implemeneted. i.e. generating single shells and
  nested shells to build a sphere, following a density profile that can include
  multiple layers.
+ Significant minor features still needed, especially regarding user input and
  options control.
+ Lots of code tidying and documentation updating to do!

## Contributing

Pull requests are more than welcome.

