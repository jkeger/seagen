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

    assert np.isclose(10.05, sphere.r, 1e-9).all()


def test_gen_sphere_stretch():
    sphere = GenSphere(1000, 10, 0.01)
    sphere.apply_stretch_factor()

    x, y, z = polar_to_cartesian(sphere.r, sphere.phi, sphere.theta)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    plot = ax.scatter(x, y, z)

    plt.savefig("test_gen_sphere_stretched.png")

    assert np.isclose(10.05, sphere.r, 1e-9).all()

def test_gen_tetra():

    def density(r):
        return 1.

    ics = GenIC(density, 0.1, (0.1, 10.))

    # Remove a quarter of the particles.
    mask = ics.r < 1.0
    ics.r = ics.r[mask]
    ics.phi = ics.phi[mask]
    ics.theta = ics.theta[mask]

    x, y, z, = polar_to_cartesian(ics.r, ics.phi, ics.theta)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.view_init(45, 45)

    d = 1
    plot = ax.scatter(x[::d], y[::d], z[::d], c=ics.r[::d], s=1, alpha=1)

    plt.savefig("test_gen_tetra.png")


def test_gen_ic():

    def density(r):
        return 1.

    ics = GenIC(density, 0.1, (0.1, 10.))

    # Remove a quarter of the particles.
    mask = np.logical_and(
        ics.phi > (np.pi * 1.0 / 2.0),
        ics.phi < (np.pi * 4.0 / 2.0),
    )
    ics.r = ics.r[mask]
    ics.phi = ics.phi[mask]
    ics.theta = ics.theta[mask]

    # Need to sort for rendering order.
    sort_order = np.flipud(ics.r.argsort())
    ics.r = ics.r[sort_order]
    ics.phi = ics.phi[sort_order]
    ics.theta = ics.theta[sort_order]


    x, y, z, = polar_to_cartesian(ics.r, ics.phi, ics.theta)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)

    ax.view_init(45, 45)

    d = 5
    plot = ax.scatter(x[::d], y[::d], z[::d], c=ics.r[::d], s=1, alpha=1)

    plt.savefig("test_gen_ic.png")

if __name__ == "__main__":
    test_gen_tetra()
