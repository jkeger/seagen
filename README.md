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
+ `setup.py`, `setup.cfg`, `__init__.py`, `MANIFEST.in` Python package files.
+ `LICENSE.txt` GNU general public license v3.


Basic Usage
-----------
+ See the doc strings in `seagen.py` for all the details.
+ Create a single shell of particles and print their positions:
    ```
    import seagen
    N = 100
    r = 1

    particles = seagen.GenShell(N, r)

    print(particles.x, particles.y, particles.z)
    ```
+ Create a full sphere of particles on an arbitrary density profile and print
    their positions, masses, densities, and (optional) material IDs:
    ```
    import seagen
    N = 100000
    radii = [ ... ]
    densities = [ ... ]
    materials = [ ... ]

    particles = seagen.GenSphere(N, radii, densities, materials)

    print(particles.x, particles.y, particles.z, particles.m, particles.rho,
          particles.mat)
    ```
+ See `examples.py` for other working examples.


Installation
------------
+ `PyPI`: Automatically install the package with `pip install seagen`, see
    https://pypi.org/project/seagen/
+ Direct download: The single `seagen.py` file can be imported and used without
    any extra installation, so you can just download this repository and place
    the file in a local directory or wherever your python will look for modules.


Requirements
------------
+ Python 2 or 3 (tested with 2.7.13 and 3.6.0).


Notation etc.
-------------
+ PEP8 is followed in most cases apart from some indentation alignment.
+ Arrays are explicitly labelled with a prefix `A1_`, or `An_` for an
    `n`-dimensional array.
+ `' '` is used for keyword arguments, `" "` for other strings.
+ Particle is abbreviated to `picle`.




