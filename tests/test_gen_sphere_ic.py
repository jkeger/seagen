"""
Tests the generation of spherical initial conditions following a density
profile.
"""
#import sys
#sys.path.append("/media/jacob/Data/Dropbox/gihr/seagen")

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
    print("Testing sphere intiial conditions generation...")

    N_picle = 1e4

    # Profiles
    N_prof      = int(1e6)
    A1_r_prof   = np.arange(1, N_prof + 1) * 1/N_prof
    A1_rho_prof = 3 - A1_r_prof**2
    A1_u_prof   = np.zeros(N_prof)
    A1_mat_prof = np.zeros(N_prof)

    # Generate particles
    particles   = GenSphereIC(
        N_picle, A1_r_prof, A1_rho_prof, A1_u_prof, A1_mat_prof
        )
#    particles.r, particles.theta, particles.phi, particles.m, particles.h,
#    particles.u, particles.mat.

#    A1_x, A1_y, A1_z    = polar_to_cartesian(
#        particles.A1_r, particles.A1_phi, particles.A1_theta
#        )


if __name__ == "__main__":
    test_gen_sphere_ic()

