SEAGen
======

A python implementation of the stretched equal area (SEA) algorithm for
generating spherically symmetric arrangements of particles with accurate
particle densities, e.g. for SPH initial conditions that precisely match an
arbitrary density profile, as presented in Kegerreis et al. (2019), MNRAS 487:4, 
5029-5040, https://doi.org/10.1093/mnras/stz1606.

See also https://github.com/srbonilla/WoMa for making the initial profiles, 
placing particles with SEAGen, and modifications for spinning bodies.

Jacob Kegerreis (2020) jacob.kegerreis@durham.ac.uk  
Josh Borrow

Visit https://github.com/jkeger/seagen to download the code including examples
and for support.

This program has been tested for a wide range of cases but not exhaustively. If
you find any bugs, potential improvements, or features worth adding, then please
let us know!


Contents
--------
+ `seagen.py` The main program classes and functions.
+ `examples.py` Examples to demonstrate how to use the SEAGen module.
+ `setup.py`, `setup.cfg`, `__init__.py`, `MANIFEST.in` Python package files.
+ `LICENSE.txt` GNU general public license v3+.


Basic Usage
-----------
+ See the doc strings in `seagen.py` for all the details.
+ Create a single shell of particles and print their positions:
    ```python
    import seagen
    N = 100
    r = 1

    particles = seagen.GenShell(N, r)

    print(particles.x, particles.y, particles.z)
    ```
+ Create a full sphere of particles on a simple density profile and print their
    positions and masses:
    ```python
    import seagen
    import numpy as np
    N = 100000
    radii = np.arange(0.01, 10, 0.01)
    densities = np.ones(len(radii))     # e.g. constant density

    particles = seagen.GenSphere(N, radii, densities)

    print(particles.x, particles.y, particles.z, particles.m)
    ```
+ See `examples.py` for other working examples, e.g. an arbitrary density
    profile with multiple layers and extra temperature information.


Installation
------------
+ `PyPI`: Automatically install the package with `pip install seagen`, see
    https://pypi.org/project/seagen/
+ Direct download: The single `seagen.py` file can be imported and used without
    any extra installation, so you can just download this repository and place
    the file in a local directory or wherever your python will look for modules.


Requirements
------------
+ Python 3 (tested with 3.6.0).


Notation etc.
-------------
+ PEP8 is followed in most cases apart from some indentation alignment.
+ Arrays are explicitly labelled with a prefix `A1_`, or `An_` for an
    `n`-dimensional array.
+ Particle is abbreviated to `picle`.
