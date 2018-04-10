"""
Tests the generation of spherical initial conditions following a density
profile.
"""
import sys
sys.path.append("/media/jacob/Data/Dropbox/gihr/seagen")

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from seagen import GenSphereIC
from seagen.helper import polar_to_cartesian

deg_to_rad  = np.pi/180

def test_gen_sphere_ic():
    """
    Tests the generation of spherical initial conditions following a density
    profile.
    """
    print("Testing sphere initial conditions generation...")

    N_picle = 1e4

    # Profiles
    N_prof      = int(1e6)
    A1_r_prof   = np.arange(1, N_prof + 1) * 1/N_prof
    A1_rho_prof = 3 - A1_r_prof**2
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

    plt.show()


if __name__ == "__main__":
    test_gen_sphere_ic()














