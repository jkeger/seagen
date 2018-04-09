"""
Objects for SEAGen

Created by: Jacob Kegerreis and Josh Borrow

This file includes:

    + GenShell, an object for generating individual spheres of particles
      using the SEA method
"""

import numpy as np

from typing import Tuple, Callable
from warnings import warn
from scipy.integrate import quad

try:
    from tqdm import tqdm
except ImportError:
    def tqdm():
        pass


class GenShell(object):
    """
    Generate a single spherical shell of particles at a fixed radius, using the
    SEA method.
    """
    def __init__(self, N: int, r: float, do_stretch=True):
        """
        Generates a single spherical shell.

        Access the particle positions with:
            GenShell.r
            GenShell.theta
            GenShell.phi.


        Inputs
        ------

        @param N | integer | the number of cells/particles to create.

        @param r | float | the radius of the shell.

        @param do_stretch | (opt.) bool | set False to not do the SEA method's
            latitude stretching.
        """

        self.N = N
        self.r = r * np.ones(N)

        # Derived properties
        self.A_reg = 4 * np.pi / N

        self.get_collar_areas()
        self.update_collar_thetas()
        self.get_point_positions()
        if do_stretch:
            self.apply_stretch_factor()


    def get_cap_theta(self) -> float:
        """
        Gets the cap colatitude, theta_cap.

        Equation 3.
        """

        return 2 * np.arcsin(np.sqrt(1 / self.N))


    def get_number_of_collars(self) -> float:
        """
        Gets the number of collars, N_col (not including the polar caps).

        Equation 4.
        """

        theta_cap = self.get_cap_theta()

        self.N_col = int(round((np.pi - 2 * theta_cap)/(np.sqrt(self.A_reg))))

        return self.N_col


    def get_collar_thetas(self) -> np.ndarray:
        """
        Gets the top theta of all of the collars, including the bottom cap's
        theta, and stores them in self.collars as well as returning them.
        """

        n_collars = self.get_number_of_collars()
        cap_height = self.get_cap_theta()
        height_of_collar = (np.pi - 2 * cap_height) / n_collars

        # Allocate collars array
        self.collars = np.arange(n_collars+1, dtype=float)
        # Collars have a fixed height initially
        self.collars *= height_of_collar
        # Starting at the bottom of the top polar cap
        self.collars += (cap_height)
        return self.collars


    def get_collar_area(
            self,
            theta_i: float,
            theta_i_minus_one: float
        ) -> float:
        """
        Gets the area of a collar given the collar heights of itself and its
        neighbour.

        Equation 5.
        """

        sin2_theta_i = np.sin(theta_i / 2)**2
        sin2_theta_i_m_o = np.sin(theta_i_minus_one / 2)**2

        return 4 * np.pi * (sin2_theta_i - sin2_theta_i_m_o)


    def get_collar_areas(self) -> np.ndarray:
        """
        Gets the collar areas and stores them in self.collar_areas.
        """

        collar_thetas = self.get_collar_thetas()

        self.collar_areas = np.empty(self.N_col)

        self.collar_areas[:] = self.get_collar_area(
            collar_thetas[1:],
            collar_thetas[:-1]
        )

        return self.collar_areas


    def get_ideal_n_regions_in_collar(self, A_col: float) -> float:
        """
        Gets the ideal number of regions in a collar.

        Equation 7.
        """

        return A_col / self.A_reg


    def get_n_regions_in_collars(self) -> np.ndarray:
        """
        Gets the number of regions in each collar.

        Stores them in self.n_regions_in_collars.

        Equation 8,9.
        """

        # Because of the discrepancy counter, we will just use a regular loop.

        n_regions_in_collars = np.empty(self.N_col, dtype=int)
        collar_areas = self.get_collar_areas()

        loop = enumerate(
            np.nditer(n_regions_in_collars, op_flags=["readwrite"])
        )

        discrepancy = 0

        for i, N_i in loop:
            ideal_n_reg = self.get_ideal_n_regions_in_collar(collar_areas[i])
            N_i[...] = int(round(ideal_n_reg + discrepancy))

            discrepancy = N_i - ideal_n_reg

        self.n_regions_in_collars = n_regions_in_collars

        return self.n_regions_in_collars


    def update_collar_thetas(self) -> np.ndarray:
        """
        After get_n_regions_in_collars, we must update the collar thetas due to
        the now integer numbers of regions in each collar instead of the ideal.

        Also returns self.collars.

        Equation 10.
        """

        # First we must get the cumulative number of regions in each collar,
        # including the top polar cap
        n_regions_cum = np.cumsum(self.get_n_regions_in_collars()) + 1
        n_regions_cum = np.append([1], n_regions_cum)

        self.collars = 2 * np.arcsin(np.sqrt(n_regions_cum * self.A_reg / \
                                             (4 * np.pi)))

        return self.collars


    def choose_phi_0(self,
            N_i: int,
            N_i_minus_one: int,
            d_phi_i: float,
            d_phi_i_minus_one: float,
        ) -> float:
        """
        Choose the starting longitude of each collar, phi_0, using the number
        of regions in this collar and the previous one.

        Paragraph following Equation 12.
        """

        N_i_even = abs((N_i % 2) - 1)
        N_i_minus_one_even = abs((N_i_minus_one % 2) - 1)

        if N_i_even != N_i_minus_one_even:
            # Exclusive or
            return 0.5 * (N_i_even * d_phi_i +
                          N_i_minus_one_even * d_phi_i_minus_one)
        else:
            return 0.5 * min(d_phi_i, d_phi_i_minus_one)


    def get_point_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sets the point positions (theta and phi) using the above-calculated
        data.

        Stores in self.theta, self.phi and also returns them.

        Equation 11,12.
        """

        total_number_of_particles = self.n_regions_in_collars.sum() + 2

        self.theta = np.empty(total_number_of_particles)
        self.phi = np.empty(total_number_of_particles)

        # The cap particles are at the poles, listed at the end of these arrays.
        self.theta[-2] = 0.0
        self.theta[-1] = np.pi
        self.phi[-2] = 0.0
        self.phi[-1] = 0.0

        # All regions in a collar are at the same colatitude, theta.
        theta = np.zeros(self.N_col + 2)
        theta[:-2] = 0.5 * (self.collars[:-1] + self.collars[1:])

        # Particles in each collar are equally spaced in longitude, phi,
        # and offset appropriately from the previous collar.
        d_phi = 2 * np.pi / self.n_regions_in_collars
        phi_0 = np.empty(self.N_col)

        loop = enumerate(
            np.nditer(phi_0, op_flags=["writeonly"])
        )

        for i, phi_0_i in loop:
            # The first collar has no previous collar to rotate away from
            # so doesn't need an offset.
            if i == 0:
                phi_0_i = 0
            else:
                phi_0_i = self.choose_phi_0(
                    self.n_regions_in_collars[i],
                    self.n_regions_in_collars[i-1],
                    d_phi[i],
                    d_phi[i-1]
                )

                # Also add a random initial offset to ensure that successive
                # collars do not create lines of ~adjacent particles.
                # (Second paragraph following Equation 12.)
                m = np.random.randint(0, self.n_regions_in_collars[i-1])
                phi_0_i += (m * d_phi[i-1])

        # Fill the position arrays.
        cumulative_number = 0
        loop = enumerate(
            np.nditer(self.n_regions_in_collars, op_flags=["readonly"])
        )

        for region, n_regions_in_collar in loop:
            next_cumulative_number = n_regions_in_collar + cumulative_number

            # Set theta
            self.theta[cumulative_number:next_cumulative_number] = theta[region]

            # Set phi (Equation 12)
            j = np.arange(n_regions_in_collar, dtype=float)
            these_phi = phi_0[region] + j * d_phi[region]
            self.phi[cumulative_number:next_cumulative_number] = these_phi

            cumulative_number = next_cumulative_number

        self.phi %= 2 * np.pi
        self.theta %= np.pi
        self.theta[-1] = np.pi

        return self.theta, self.phi


    def apply_stretch_factor(self, a=0.2, b=2.0):
        """
        Apply the SEA stretch factor.

        Equation 13.
        """

        pi_over_2 = np.pi / 2
        inv_sqrtN = 1 / np.sqrt(self.N)

        prefactor = (pi_over_2 - self.theta) * a * inv_sqrtN

        exp_factor = - ((pi_over_2 - abs(pi_over_2 - self.theta))
                        / (np.pi * b * inv_sqrtN))

        self.theta += (prefactor * np.exp(exp_factor))

        # Leave the cap points at the poles
        self.theta[-2] = 0.0
        self.theta[-1] = np.pi

        return


def get_shell_mass(r_inner: float, r_outer: float, rho: float):
    """
    Calculate the mass of a uniform-density shell.
    """
    return 4/3*np.pi * rho * (r_outer**3 - r_inner**3)

def get_mass_weighted_mean(A1_mass, A1_value):
    """
    Calculate the mean of the value array weighted by the mass array.
    """
    return np.sum(A1_mass * A1_value) / np.sum(A1_mass)

class GenSphereIC(object):
    """
    Generate particle initial conditions with the SEA method and nested shells,
    following a density profile.
    """
    def __init__(
            self,
            N_picle_des: float,
            A1_r_prof: np.ndarray,
            A1_rho_prof: np.ndarray,
            A1_u_prof: np.ndarray,
            A1_mat_prof: np.ndarray
        ):
        """
        Generates nested spherical shells of particles to match radial profiles.

        Inputs
        ------

        @param N_picle_des | float | desired number of particles.

        @param A1_r_prof | ndarray | an array of the profile radii for this
                                     layer.

        @param A1_rho_prof | ndarray | an array of densities at the profile
                                       radii.

        @param A1_u_prof | ndarray | an array of specific internal energies at
                                     the profile radii.

        @param A1_mat_prof | ndarray | an array of material identifiers at the
                                       profile radii.

        Outputs
        -------

        GenLayer.A1_r, GenLayer.A1_theta, GenLayer.A1_phi, GenLayer.A1_m, GenLayer.A1_h,
        GenLayer.A1_u, GenLayer.A1_mat.
        """
        self.N_picle_des    = N_picle_des
        self.A1_r_prof      = A1_r_prof
        self.A1_rho_prof    = A1_rho_prof
        self.A1_u_prof      = A1_u_prof
        self.A1_mat_prof    = A1_mat_prof

        # Derived
        self.N_prof = len(self.A1_r_prof)
        self.A1_m_prof  = np.empty(self.N_prof)
        # Values for each shell
        A1_N_shell          = []
        A1_m_shell          = []
        A1_m_picle_shell    = []
        A1_r_shell          = []
        A1_rho_shell        = []
        A1_h_shell          = []
        A1_u_shell          = []
        A1_mat_shell        = []
        # All particle data
        self.A1_m       = []
        self.A1_r       = []
        self.A1_h       = []
        self.A1_u       = []
        self.A1_mat     = []
        self.A1_theta   = []
        self.A1_phi     = []

        # # Check profiles start from non-zero radius...

        # # Interpolate profiles if needed...

        # Calculate the mass profile
        self.get_mass_profile()
        # Enclosed mass profile
        self.A1_m_enc_prof  = np.cumsum(self.A1_m_prof)
        self.m_tot          = self.A1_m_enc_prof[-1]
        self.m_picle_des    = self.m_tot / self.N_picle_des

        # Find the radii of all material boundaries (including the outer edge)
        self.find_material_boundaries()

        # First (innermost) layer
        r_bound     = self.A1_r_bound[0]
        idx_bound   = self.A1_idx_bound[0]

        # Vary the particle mass until a particle shell boundary coincides with
        # the profile boundary
        print("\nTweaking the particle mass to fix the boundaries...")
        self.m_picle        = self.m_picle_des
        # Initialise
        idx_shell_bound_prev    = idx_bound
        idx_shell_bound_best_1  = 0
        idx_shell_bound_best_2  = 0
        # This also sets A1_idx_outer and A1_r_outer
        while True:
            # Find the outer boundary radii of all shells
            A1_idx_outer    = []
            A1_r_outer      = []

            # Set the core dr with the radius containing the mass of the central
            # tetrahedron of 4 particles
            N_picle_shell   = 4
            idx_outer       = np.searchsorted(
                self.A1_m_enc_prof, N_picle_shell * self.m_picle
                )
            r_outer         = self.A1_r_prof[idx_outer]
            self.dr_core    = r_outer

            # Mass-weighted mean density
            self.rho_core   = get_mass_weighted_mean(
                self.A1_m_prof[:idx_outer], self.A1_rho_prof[:idx_outer]
                )

            # Record shell boundary
            A1_idx_outer.append(idx_outer)
            A1_r_outer.append(self.dr_core)

            while A1_r_outer[-1] < r_bound:
                # Calculate the shell width from the density relative to the
                # core radius and density
                rho = self.A1_rho_prof[idx_outer]
                dr  = self.dr_core * np.cbrt(self.rho_core / rho)

                # Find the profile radius just beyond this shell (r_outer + dr)
                idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)
                if idx_outer == self.N_prof:    ###
                    idx_outer   -= 1
                r_outer     = self.A1_r_prof[idx_outer]

                A1_idx_outer.append(idx_outer)
                A1_r_outer.append(r_outer)

            # Reduce the particle mass until the shell boundary reaches the
            # profile boundary
            ...
#            idx_shell_bound = A1_idx_outer[-1]
#            # Stop when either the boundary is perfect or has settled at the
#            # same place after two reductions of the mass variation, dm_picle
#            if (idx_shell_bound == idx_bound or
#                (idx_shell_bound == idx_shell_bound_best_1 and
#                 idx_shell_bound == idx_shell_bound_best_2)):
#                break
#            # If we've gone too far then go back and make smaller adjustments
#            elif idx_shell_bound > idx_shell_bound_prev:
#                self.m_picle    *= 1 + dm_picle
#                dm_picle        /= 2
#
#                idx_shell_bound_best_2  = idx_shell_bound_best_1
#                idx_shell_bound_best_1  = idx_shell_bound_prev
#            # Decrease the particle mass and try again
#            else:
#                self.m_picle    *= 1 - dm_picle
#
#            idx_shell_bound_prev  = idx_shell_bound

            break   ###
        print("Done particle mass tweaking! From %g to %g" % (
            self.m_picle_des, self.m_picle
            ))

        print("\nDividing the profiles into shells...")
        print("  with particle properties:")
        header  = ("    Radius    Number   Mass      Density   Energy    "
                   "Material")
        print(header)
        # Set the mass-weighted values for each shell
        idx_inner   = 0
        for i_shell, idx_outer in enumerate(A1_idx_outer):
            A1_m_prof_shell = self.A1_m_prof[idx_inner:idx_outer]

            # Mass
            A1_m_shell.append(np.sum(A1_m_prof_shell))
            # Number of particles
            if i_shell == 0:
                A1_N_shell.append(4)
            else:
                A1_N_shell.append(int(round(A1_m_shell[-1] / self.m_picle)))
            # Actual particle mass
            A1_m_picle_shell.append(self.m_picle / A1_N_shell[-1])

            # Radius (mean of half-way and mass-weighted radii)
            r_half  = (
                self.A1_r_prof[idx_inner] + self.A1_r_prof[idx_outer]
                ) / 2
            r_mwm   = get_mass_weighted_mean(
                A1_m_prof_shell, self.A1_r_prof[idx_inner:idx_outer]
                )
            A1_r_shell.append((r_half + r_mwm) / 2)

            # Other properties
            A1_rho_shell.append(get_mass_weighted_mean(
                A1_m_prof_shell, self.A1_rho_prof[idx_inner:idx_outer]
                ))
            A1_u_shell.append(get_mass_weighted_mean(
                A1_m_prof_shell, self.A1_u_prof[idx_inner:idx_outer]
                ))
            A1_mat_shell.append(get_mass_weighted_mean(
                A1_m_prof_shell, self.A1_mat_prof[idx_inner:idx_outer]
                ))

            idx_inner   = idx_outer

            print("    %.2e  %07d  %.2e  %.2e  %.2e  %d" % (
                A1_r_shell[-1], A1_N_shell[-1], A1_m_picle_shell[-1],
                A1_rho_shell[-1], A1_u_shell[-1], A1_mat_shell[-1]
                ))
        print(header)
        print("Shells done!")

        # Estimate the smoothing lengths from the densities
        num_ngb     = 50
        kernel_edge = 2
        A1_h_shell  = (
            3/(4*np.pi) * num_ngb * np.array(A1_m_shell) / np.cbrt(A1_rho_shell)
            / kernel_edge
            )

#        # Print a table of the shells
#        print("\nFilling the shells with particles...")
#        header  = " r         N       m         h         u         mat"
#        print(header)
#        for N, r, m, h, u, mat in zip(
#            A1_N_shell, A1_r_shell, A1_m_picle_shell, A1_h_shell, A1_u_shell,
#            A1_mat_shell
#            ):
#            print(" %.2e  %06d  %.2e  %.2e  %.2e  %d" % (r, N, m, h, u, mat))
#        print(header)

        # Generate the particles in each shell
        print("\nArranging the particles in each shell...")
        for N, r, m, h, u, mat in zip(
            A1_N_shell, A1_r_shell, A1_m_picle_shell, A1_h_shell, A1_u_shell,
            A1_mat_shell
            ):
            self.generate_shell_particles(N, m, r, h, u, mat)
        print("Particles done!")

        # Randomly rotate each shell
        ...

        # Outer layer(s)
        # Vary the number of particles in the first shell of this layer until a
        # particle shell boundary coincides with the next profile boundary
        ...

    def get_mass_profile(self):
        """
        Calculate the mass profile from the density profile.
        """
        # Find the mass in each profile shell, starting with the central sphere
        self.A1_m_prof      = np.empty(self.N_prof)
        self.A1_m_prof[0]   = get_shell_mass(
            0.0, self.A1_r_prof[0], self.A1_rho_prof[0]
            )
        self.A1_m_prof[1:]  = get_shell_mass(
            self.A1_r_prof[:-1], self.A1_r_prof[1:], self.A1_rho_prof[1:]
            )

        return

    def find_material_boundaries(self):
        """
        Find the radii of any layer boundaries (where the material changes),
        including the outer edge.

        Set A1_idx_bound and A1_r_bound.
        """
        A1_idx_bound    = np.where(np.diff(self.A1_mat_prof) == 1)[0]
        # Include the outer edge
        self.A1_idx_bound   = np.append(A1_idx_bound, self.N_prof - 1)
        self.A1_r_bound     = self.A1_r_prof[self.A1_idx_bound]

        return

    def generate_tetrahedron_particles(
        self, r: float, m: float, h: float, u: float, mat: int
        ):
        """
        Make a tetrahedron of 4 particles with the given properties.
        """
        N   = 4

        # Tetrahedron vertex coordinates
        A1_x        = np.array([1, 1, -1, -1])
        A1_y        = np.array([1, -1, 1, -1])
        A1_z        = np.array([1, -1, -1, 1])
        A1_theta    = np.arccos(A1_z / np.sqrt(A1_x**2 + A1_y**2 + A1_z**2))
        A1_phi      = np.arctan(A1_y / A1_x)

        # Append the data to the arrays of all particles
        self.A1_r.append([r] * N)
        self.A1_m.append([m] * N)
        self.A1_h.append([h] * N)
        self.A1_u.append([u] * N)
        self.A1_mat.append([mat] * N)
        self.A1_theta.append(A1_theta)
        self.A1_phi.append(A1_phi)

        return

    def generate_shell_particles(
        self, N: int, r: float, m: float, h: float, u: float, mat: int
        ):
        """
        Make a single spherical shell of particles with the given properties.
        """
        # Make a tetrahedron for the central 4 particles
        if N == 4:
            self.generate_tetrahedron_particles(r, m, h, u, mat)

            return
        # Make an SEA shell otherwise
        shell = GenShell(N, r)

        # Append the data to the arrays of all particles
        self.A1_r.append([r] * N)
        self.A1_m.append([m] * N)
        self.A1_h.append([h] * N)
        self.A1_u.append([u] * N)
        self.A1_mat.append([mat] * N)
        self.A1_theta.append(shell.theta)
        self.A1_phi.append(shell.phi)

        return



