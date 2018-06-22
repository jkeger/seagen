SEAGen
======

A python implementation of the stretched equal area (SEA) algorithm for
generating spherically symmetric arrangements of particles with accurate
particle densities, e.g. for SPH initial conditions that precisely match an
arbitrary density profile, as presented in Kegerreis et al. (2018), *in prep*.

Copyright (C) 2018 Jacob Kegerreis (jacob.kegerreis@durham.ac.uk)

Jacob Kegerreis and Josh Borrow

Visit https://github.com/jkeger/seagen to download the code including examples
and for support.

This program has been tested for a wide range of cases but not exhaustively. If
you find any bugs, potential improvements, or features worth adding, then please
do contact the author at the email address above!

Pull requests are more than welcome.


Contents
--------
+ `seagen.py` The main program classes and functions.
+ `examples.py` Examples to demonstrate how to use the SEAGen module.
+ `setup.py`, `__init__.py`, `MANIFEST.in` Python package files.
+ `LICENSE.txt` GNU general public license v3.


Notation etc.
-------------
+ PEP8 is followed in most cases apart from some indentation alignment.
+ Arrays are explicitly labelled with a prefix `A1_`, or `An_` for an
    `n`-dimensional array.
+ Particle is abbreviated to `picle`.
+ `' '` is used for keyword arguments, `" "` for other strings.




