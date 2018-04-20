"""
This file is part of SEAGen.
Copyright (C) 2018 Jacob Kegerreis (jacob.kegerreis@durham.ac.uk)
GNU General Public License http://www.gnu.org/licenses/

Jacob Kegerreis and Josh Borrow

Tests the generation of spherical initial conditions following a density
profile.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from seagen import GenSphereIC
from seagen.helper import polar_to_cartesian


def test_gen_sphere_ic_simple():
    """
    Tests the generation of spherical initial conditions following a simple
    density profile.
    """
    print("=======================================================")
    print(" Testing sphere initial conditions generation (simple) ")
    print("=======================================================")

    N_picle = 1e4

    # Profiles
    N_prof      = int(1e6)
    A1_r_prof   = np.arange(1, N_prof + 1) * 1/N_prof
    A1_rho_prof = 3 - 2*A1_r_prof**2
    A1_u_prof   = np.zeros(N_prof)
    A1_mat_prof = np.zeros(N_prof)

    # Generate particles
    particles   = GenSphereIC(
        N_picle, A1_r_prof, A1_rho_prof, A1_u_prof, A1_mat_prof, verb=2
        )

    # Figure
    plt.figure(figsize=(7, 7))

    plt.plot(A1_r_prof, A1_rho_prof)

    plt.scatter(particles.A1_r, particles.A1_rho)

    plt.xlabel("Radius")
    plt.ylabel("Density")

    plt.xlim(0, None)
    plt.ylim(0, None)

    plt.title("Test Sphere Initial Conditions (Simple)")

    plt.tight_layout()

    filename = "test_gen_sphere_ic_simple.png"
    plt.savefig(filename)
    print("Saved figure to %s" % filename)

    plt.show()


def test_gen_sphere_ic_layers():
    """
    Tests the generation of spherical initial conditions following a density
    profile with multiple layers and density discontinuities.
    """
    print("================================================================")
    print(" Testing sphere initial conditions generation (multiple layers) ")
    print("================================================================")

    N_picle = 1e4

    # Profiles
    N_prof      = int(1e6)
    A1_r_prof   = np.arange(1, N_prof + 1) * 1/N_prof
    A1_rho_prof = 3 - 2*A1_r_prof**2
    # Separate density profile into three layers of different materials
    A1_rho_prof *= np.array(
        [1]*int(N_prof/4) + [0.7]*int(N_prof/2) + [0.3]*int(N_prof/4)
        )
    A1_u_prof   = np.zeros(N_prof)
    A1_mat_prof = np.array(
        [0]*int(N_prof/4) + [1]*int(N_prof/2) + [2]*int(N_prof/4)
        )

    # Generate particles
    particles   = GenSphereIC(
        N_picle, A1_r_prof, A1_rho_prof, A1_u_prof, A1_mat_prof, verb=2
        )

    # Figure
    plt.figure(figsize=(7, 7))

    plt.plot(A1_r_prof, A1_rho_prof)

    plt.scatter(particles.A1_r, particles.A1_rho)

    plt.xlabel("Radius")
    plt.ylabel("Density")

    plt.xlim(0, None)
    plt.ylim(0, None)

    plt.title("Test Sphere Initial Conditions (Multi-Layer)")

    plt.tight_layout()

    filename = "test_gen_sphere_ic_layers.png"
    plt.savefig(filename)
    print("Saved figure to %s" % filename)

    plt.show()


if __name__ == "__main__":
    test_gen_sphere_ic_simple()
    test_gen_sphere_ic_layers()














