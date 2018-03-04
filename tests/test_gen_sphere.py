"""
Tests the sphere generation code. This is notriously difficult.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from seagen import GenSphere, GenIC
from seagen.helper import polar_to_cartesian


def test_gen_sphere():
    sphere = GenSphere(1000, 10, 0.1)

    x, y, z = polar_to_cartesian(sphere.r, sphere.phi, sphere.theta)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    plot = ax.scatter(x, y, z)

    plt.savefig("test_gen_sphere.png")


def test_gen_sphere_stretch():
    sphere = GenSphere(1000, 10, 0.1)
    sphere.apply_stretch_factor()

    x, y, z = polar_to_cartesian(sphere.r, sphere.phi, sphere.theta)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    plot = ax.scatter(x, y, z)

    plt.savefig("test_gen_sphere_stretched.png")


def test_gen_ic():

    def density(r):
        return 1.

    ics = GenIC(density, 1, (0.1, 10.))

    x, y, z, = polar_to_cartesian(ics.r, ics.phi, ics.theta)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    d = 100
    plot = ax.scatter(x[::d], y[::d], z[::d], c=ics.r[::d])

    plt.savefig("test_gen_ic.png")

if __name__ == "__main__":
    test_gen_ic()
