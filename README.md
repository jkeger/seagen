SEAGen
======

A python implementation of the stretched-equal-area (SEA) algorithm for
generating spherically symmetric arrangements of particles for SPH initial
conditions with accurate densities (Kegerreis et al., 2018).

Jacob Kegerreis and Josh Borrow

jacob.kegerreis@durham.ac.uk
joshua.borrow@durham.ac.uk

--------------
Requirements:

+ `python3.6.0` or above
+ `tqdm`, `matplotlib`, `scipy`, `numpy`, `typing`, `numba`
    (available through `pip3 install -r requirements.txt`)

Currently this package is _not_ on PyPI but we plan to release it there once
completed.

Current Status
--------------

+ Fresh start: Keeping initial implementation of the single-shell generation
  from jborrow with just a few checks and tweaks to do, but, for simplicity,
  restart from scratch with the full-nested-shells generation for easier
  compatability with arbitrary density profiles and the inclusion of other
  profile properties.
+ Similar re-making needed of test codes and examples.
+ Documentation will obviously need revising too.


Contributing
------------

Pull requests are more than welcome.

