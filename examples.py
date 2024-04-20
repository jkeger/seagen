""" SEAGen Examples

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    Simple example functions to demonstrate how to use the SEAGen module.
    Figures are plotted to show the generated particles, and relevant
    information is printed when appropriate.

    See README.md and https://github.com/jkeger/seagen for more information.

    GNU General Public License v3+, see LICENSE.txt.

    To run all the examples and display the figures, call:
        $  python  examples.py
"""
# ========
# Contents:
# ========
#   I   Example Functions
#   II  Main

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from seagen import GenShell, GenSphere, polar_to_cartesian

# ========
# Constants
# ========
deg_to_rad = np.pi / 180

# //////////////////////////////////////////////////////////////////////////// #
#                               III. Example Functions                         #
# //////////////////////////////////////////////////////////////////////////// #


def test_gen_shell(N):
    """Generate a single spherical shell of particles.

    Save a 3D figure of the particles on the shell.

    Args:
        N (int)
            The number of particles to arrange.

    Note: Matplotlib's 3D is not always great and sometimes some particles
    can go invisible depending on the view angle!
    """
    print(
        "\n======================================================"
        "\n SEAGen single shell generation with N = %d particles "
        "\n======================================================" % N
    )

    # Generate the particles
    particles = GenShell(N, 1, do_stretch=True, do_rotate=False)

    # Figure
    ax_lim = 1  # axis limits (in +/- x, y, z)
    elev = 35  # 3D viewpoint
    azim = 0

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Dense longitude lines since plotting an actual sphere surface looks weird
    for phi in np.arange(0, 360, 1):
        A1_r = np.ones(100)
        A1_theta = np.linspace(0, np.pi, 100)
        A1_phi = np.ones(100) * phi * deg_to_rad

        A1_x, A1_y, A1_z = polar_to_cartesian(A1_r, A1_theta, A1_phi)

        plt.plot(A1_x, A1_y, A1_z, c="0.8")

    # Plot particles with alternating collar colours
    A1_z_collar = np.unique(particles.A1_z)
    for i_col in range(len(A1_z_collar)):
        sel_col = np.where(abs(particles.A1_z - A1_z_collar[i_col]) < 0.001)[0]
        if i_col % 2 == 0:
            colour = "dodgerblue"
        else:
            colour = "blueviolet"

        ax.scatter(
            particles.A1_x[sel_col],
            particles.A1_y[sel_col],
            particles.A1_z[sel_col],
            c=colour,
            marker="o",
            s=100,
        )

        # Latitude lines
        A1_r = np.ones(100)
        A1_theta = np.ones(100) * np.arccos(A1_z_collar[i_col])
        A1_phi = np.linspace(0, 2 * np.pi, 101)[:-1]

        A1_x, A1_y, A1_z = polar_to_cartesian(A1_r, A1_theta, A1_phi)

        plt.plot(A1_x, A1_y, A1_z, c=colour, alpha=0.2)

    # Equator
    A1_r = np.ones(100)
    A1_theta = np.ones(100) * 90 * deg_to_rad
    A1_phi = np.linspace(0, 2 * np.pi, 101)[:-1]

    A1_x, A1_y, A1_z = polar_to_cartesian(A1_r, A1_theta, A1_phi)
    plt.plot(A1_x, A1_y, A1_z, c="k", ls="--", alpha=0.3)

    # z axis
    plt.plot([0, 0], [0, 0], [-1.2, -1.0], c="k", ls="-", alpha=0.8)
    plt.plot([0, 0], [0, 0], [-1.0, 1.0], c="k", ls="-", alpha=0.3)
    plt.plot([0, 0], [0, 0], [1.0, 1.2], c="k", ls="-", alpha=0.8)

    # Axes etc.
    ax.view_init(elev=elev, azim=azim)
    ax._axis3don = False

    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)

    plt.tight_layout()

    plt.savefig("test_gen_shell.png")


def test_gen_sphere_simple():
    """Generate spherical particle positions from a simple density profile.

    Save a figure of the particles on the radial density profile.
    """
    print(
        "\n=========================================================="
        "\n SEAGen sphere particles generation with a simple profile "
        "\n=========================================================="
    )

    N_picle = 1e4

    # Profiles
    N_prof = int(1e6)
    A1_r_prof = np.arange(1, N_prof + 1) * 1 / N_prof
    A1_rho_prof = 3 - 2 * A1_r_prof**2

    # Generate particles
    particles = GenSphere(N_picle, A1_r_prof, A1_rho_prof, verbosity=2, seed=12345)

    # Figure
    plt.figure(figsize=(7, 7))

    plt.plot(A1_r_prof, A1_rho_prof)

    plt.scatter(particles.A1_r, particles.A1_rho)

    plt.xlabel("Radius")
    plt.ylabel("Density")

    plt.xlim(0, None)
    plt.ylim(0, None)

    plt.title("SEAGen Sphere Particles (Simple Profile)")

    plt.tight_layout()

    plt.savefig("test_gen_sphere_simple.png")


def test_gen_sphere_layers():
    """Generate spherical particle positions from a density profile with
    multiple layers, density discontinuities, and a temperature profile.

    Save a figure of the particles on the radial density and temperature
    profiles.
    """
    print(
        "\n==============================================================="
        "\n SEAGen sphere particles generation with a multi-layer profile "
        "\n==============================================================="
    )

    N_picle = 1e4

    # Profiles
    N_prof = int(1e6)
    A1_r_prof = np.arange(1, N_prof + 1) * 1 / N_prof
    # A density profile with three layers of different materials
    A1_rho_prof = 3 - 2 * A1_r_prof**2
    A1_rho_prof *= np.array(
        [1] * int(N_prof / 4) + [0.7] * int(N_prof / 2) + [0.3] * int(N_prof / 4)
    )
    A1_mat_prof = np.array(
        [0] * int(N_prof / 4) + [1] * int(N_prof / 2) + [2] * int(N_prof / 4)
    )
    A1_T_prof = 500 - 200 * A1_r_prof**2

    # Generate particles
    particles = GenSphere(
        N_picle,
        A1_r_prof,
        A1_rho_prof,
        A1_mat_prof=A1_mat_prof,
        A1_T_prof=A1_T_prof,
        verbosity=2,
        seed=12345,
    )

    # Figure
    plt.figure(figsize=(7, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(A1_r_prof, A1_rho_prof, c="b")
    ax1.scatter(particles.A1_r, particles.A1_rho, c="b")

    ax2.plot(A1_r_prof, A1_T_prof, c="r")
    ax2.scatter(particles.A1_r, particles.A1_T, c="r")

    ax1.set_xlabel("Radius")
    ax1.set_ylabel("Density")
    ax2.set_ylabel("Temperature")
    ax1.yaxis.label.set_color("b")
    ax2.yaxis.label.set_color("r")

    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)

    plt.title("SEAGen Sphere Particles (Multi-Layer Profile)")

    plt.tight_layout()

    plt.savefig("test_gen_sphere_layers.png")


# //////////////////////////////////////////////////////////////////////////// #
#                               II. Main                                       #
# //////////////////////////////////////////////////////////////////////////// #

if __name__ == "__main__":
    # Run the examples
    # test_gen_shell(42)
    # test_gen_sphere_simple()
    test_gen_sphere_layers()
