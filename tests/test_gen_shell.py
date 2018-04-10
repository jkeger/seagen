"""
Tests the generation of a single spherical shell of particles.
"""
#import sys
#sys.path.append("/media/jacob/Data/Dropbox/gihr/seagen")

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from seagen import GenShell
from seagen.helper import polar_to_cartesian

deg_to_rad  = np.pi/180

def test_gen_shell(N=100):
    """
    Test the generation of a single spherical shell of N particles.
    """
    print("Testing single shell generation with N = %d particles..." % N)
    particles   = GenShell(N, 1, do_stretch=True, do_rotate=False)

    # Figure
    ax_lim  = 1                     # axis limits (in +/- x, y, z)
    elev    = 25                    # 3D viewpoint
    azim    = 0

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(1, 1, 1, aspect='equal', projection="3d")

    # Dense longitude lines since plotting an actual sphere surface looks awful
    for phi in np.arange(0, 360, 1):
        r       = np.ones(100) * 0.99
        theta   = np.linspace(0, np.pi, 100)
        phi     = np.ones(100) * phi * deg_to_rad
        x, y, z = polar_to_cartesian(r, theta, phi)
        plt.plot(x, y, z, c='0.8')

    # Plot particles with alternating collar colours
    z_collars   = np.unique(particles.z)
    for i_col in range(len(z_collars)):
        sel_col = np.where(abs(particles.z - z_collars[i_col]) < 0.001)[0]
        if i_col%2 == 0:
            colour  = 'dodgerblue'
        else:
            colour  = 'blueviolet'

        ax.scatter(
            particles.x[sel_col], particles.y[sel_col], particles.z[sel_col],
            c=colour, marker='o', s=100
            )

        # Latitude lines
        r       = np.ones(100)
        theta   = np.ones(100) * np.arccos(z_collars[i_col])
        phi     = np.linspace(0, 2*np.pi, 101)[:-1]
        x_lat, y_lat, z_lat = polar_to_cartesian(r, theta, phi)

        plt.plot(x_lat, y_lat, z_lat, c=colour, alpha=0.2)

    # Equator
    r       = np.ones(100)
    theta   = np.ones(100) * 90 * deg_to_rad
    phi     = np.linspace(0, 2*np.pi, 101)[:-1]
    x, y, z = polar_to_cartesian(r, theta, phi)
    plt.plot(x, y, z, c='k', ls='--', alpha=0.3)

    # z axis
    plt.plot([0, 0], [0, 0], [-1.2, -1.0], c='k', ls='-', alpha=0.8)
    plt.plot([0, 0], [0, 0], [-1.0, 1.0], c='k', ls='-', alpha=0.3)
    plt.plot([0, 0], [0, 0], [1.0, 1.2], c='k', ls='-', alpha=0.8)

    # Axes etc.
    ax.view_init(elev=elev, azim=azim)
    ax._axis3don = False

    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)

    plt.tight_layout()

    filename = "test_gen_shell.png"
    plt.savefig(filename)
    print("Saved figure to %s" % filename)

    plt.show()

if __name__ == "__main__":
    test_gen_shell()

