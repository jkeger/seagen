SEAGen
======

A python implementation of Jacob Kegerreis' Stretched-Equal-Area generation
algorithm for SPH spheres.

Requirements:

+ `python3.6.0` or above
+ `tqdm`, `matplotlib`, `scipy`, `numpy` (available through `pip3 install -r requrements.txt`)

Currently this package is _not_ on PyPI but we plan to release it there once
completed.

Current Status
--------------

+ Initial implementation of the generation algorithms is coming along nicely;
  these are available through the `GenIC` object in `seagen/objects.py`.
+ Documentation is somewhat sparse, however this should improve shortly.
+ Need to implement some basic SPH algorithms for generating density and
  smoothing length from the particle aragements.
+ Frontends are still required; I hope to produce one for GADGET-2 (SWIFT)
  files and one that matches a text-file based input system.


Contributing
------------

Pull requests are more than welcome.

