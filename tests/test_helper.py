"""
Tests the helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from seagen.helper import polar_to_cartesian


def test_polar_to_cartesian():
    """
    Tests the polar to cartesian conversion by ensuring
    that r is conserved.
    """
    phi = np.random.rand(1000) * np.pi
    theta = np.random.rand(1000) * np.pi * 2

    r = np.array([1])

    x, y, z = polar_to_cartesian(r, theta, phi)

    # Here we check if r is conserved!
    new_r = np.sqrt(x*x + y*y + z*z)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    plot = ax.scatter(x, y, z)

    plt.savefig("test_polar_to_cartesian.png")


    assert np.isclose(r, new_r, 1e-9).all()

