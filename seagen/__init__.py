"""
Streched Equal Area Generator

A python implementation of the stretched equal area (SEA) algorithm
for generating spherically symmetric arrangements of particles with
accurate particle densities, e.g. for SPH initial conditions that
precisely match an arbitrary density profile, as presented in
Kegerreis et al. (2018), in prep.

Copyright (C) 2018 Jacob Kegerreis (jacob.kegerreis@durham.ac.uk)

GNU General Public License http://www.gnu.org/licenses/

Jacob Kegerreis and Josh Borrow

Visit https://github.com/jkeger/seagen to download the code including
example scripts and for support.
"""

from seagen.objects import GenShell, GenSphereIC

import seagen.helper as helper
import seagen.sph as sph
import seagen.secant as secant

