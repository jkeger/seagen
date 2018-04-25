"""
This file is part of SEAGen.
Copyright (C) 2018 Jacob Kegerreis (jacob.kegerreis@durham.ac.uk)
GNU General Public License http://www.gnu.org/licenses/

Jacob Kegerreis and Josh Borrow

Tests the generation of a single spherical shell of particles.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from seagen import GenShell
from seagen.helper import polar_to_cartesian

deg_to_rad  = np.pi/180


def test_gen_shell(N=100, show=False):
    """
    Test the generation of a single spherical shell of N particles.

    Note: Matplotlib's 3D is not always great and sometimes some particles can
        go invisible depending on the view angle!

    @param N | (opt.) int | The number of particles to arrange in a shell.

    @param show | (opt.) bool | Set True to display the test figure as well
        as saving it. Default False to only save the figure.
    """
    print("========================================================")
    print(" Testing single shell generation with N = %d particles" % N)
    print("========================================================")
    particles   = GenShell(N, 1, do_stretch=True, do_rotate=False)

    # Figure
    ax_lim  = 1                     # axis limits (in +/- x, y, z)
    elev    = 25                    # 3D viewpoint
    azim    = 0

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(1, 1, 1, aspect='equal', projection="3d")

    # Dense longitude lines since plotting an actual sphere surface looks awful
    for phi in np.arange(0, 360, 1):
        A1_r        = np.ones(100)
        A1_theta    = np.linspace(0, np.pi, 100)
        A1_phi      = np.ones(100) * phi * deg_to_rad

        A1_x, A1_y, A1_z    = polar_to_cartesian(A1_r, A1_theta, A1_phi)

        plt.plot(A1_x, A1_y, A1_z, c='0.8')

    # Plot particles with alternating collar colours
    A1_z_collar = np.unique(particles.A1_z)
    for i_col in range(len(A1_z_collar)):
        sel_col = np.where(abs(particles.A1_z - A1_z_collar[i_col]) < 0.001)[0]
        if i_col%2 == 0:
            colour  = 'dodgerblue'
        else:
            colour  = 'blueviolet'

        ax.scatter(
            particles.A1_x[sel_col], particles.A1_y[sel_col],
            particles.A1_z[sel_col], c=colour, marker='o', s=100
            )

        # Latitude lines
        A1_r        = np.ones(100)
        A1_theta    = np.ones(100) * np.arccos(A1_z_collar[i_col])
        A1_phi      = np.linspace(0, 2*np.pi, 101)[:-1]

        A1_x, A1_y, A1_z    = polar_to_cartesian(A1_r, A1_theta, A1_phi)

        plt.plot(A1_x, A1_y, A1_z, c=colour, alpha=0.2)

    # Equator
    A1_r        = np.ones(100)
    A1_theta    = np.ones(100) * 90 * deg_to_rad
    A1_phi      = np.linspace(0, 2*np.pi, 101)[:-1]

    A1_x, A1_y, A1_z    = polar_to_cartesian(A1_r, A1_theta, A1_phi)
    plt.plot(A1_x, A1_y, A1_z, c='k', ls='--', alpha=0.3)

    # A1_z axis
    plt.plot([0, 0], [0, 0], [-1.2, -1.0], c='k', ls='-', alpha=0.8)
    plt.plot([0, 0], [0, 0], [-1.0, 1.0], c='k', ls='-', alpha=0.3)
    plt.plot([0, 0], [0, 0], [1.0, 1.2], c='k', ls='-', alpha=0.8)

    # Axes etc.
    ax.view_init(elev=elev, azim=azim)
    ax._axis3don    = False

    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)

    plt.tight_layout()

    filename    = "test_gen_shell.png"
    plt.savefig(filename)
    print("Saved figure to %s" % filename)

    if show:
        plt.show()


if __name__ == "__main__":
    test_gen_shell()










