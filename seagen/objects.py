"""
Objects for SEAGen

Created by: Jacob Kegerreis and Josh Borrow

This file includes:

    + GenShell, an object for generating individual spherical shells of
      particles using the SEA method.

    + GenSphereIC, an object for generating spherical initial conditions of
      particles in nested shells.

Notation:
    Arrays of dimension * are explicitly labelled as A*_name
"""

import numpy as np

from typing import Tuple, Callable
from warnings import warn
from scipy.integrate import quad

from seagen.helper import polar_to_cartesian, get_euler_rotation_matrix, \
    get_shell_mass, get_mass_weighted_mean

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
    def __init__(self, N: int, r: float, do_stretch=True, do_rotate=True):
        """
        Generates a single spherical shell of particles.

        Access the particle positions with:
            GenShell.x
            GenShell.y
            GenShell.z

        (Spherical polar coordinates are used internally but do not have the
        final rotation applied to them.)


        Inputs
        ------

        @param N | integer | the number of cells/particles to create.

        @param r | float | the radius of the shell.

        @param do_stretch | (opt.) bool | set False to not do the SEA method's
            latitude stretching (default: True).

        @param do_stretch | (opt.) bool | set True to randomly rotate the
            sphere of particles after their intial placement (default: True).


        Notation
        ------

        theta: zenith (colatitude)
        phi: azimuth (longitude)
        """
        self.N = N
        self.r = r * np.ones(N)

        # Derived properties
        self.A_reg = 4 * np.pi / N

        # Start in spherical polar coordinates
        self.get_collar_areas()
        self.update_collar_thetas()
        self.get_point_positions()
        if do_stretch:
            self.apply_stretch_factor()

        # Now convert to cartesian coordinates
        self.x, self.y, self.z = polar_to_cartesian(
            self.r, self.theta, self.phi
            )
        if do_rotate:
            self.apply_random_rotation()


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
        n_regions_in_collars = np.empty(self.N_col, dtype=int)
        collar_areas = self.get_collar_areas()

        loop = enumerate(
            np.nditer(n_regions_in_collars, op_flags=["readwrite"])
        )

        discrepancy = 0

        for i, N_i in loop:
            ideal_n_reg = self.get_ideal_n_regions_in_collar(collar_areas[i])
            N_i[...] = int(round(ideal_n_reg + discrepancy))

            discrepancy += ideal_n_reg - N_i

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
        Sets the point positions in the centres of every region.

        Stores in self.theta, self.phi and also returns them.

        Equation 11,12.
        """
        N_tot   = self.n_regions_in_collars.sum() + 2

        self.theta  = np.empty(N_tot)
        self.phi    = np.empty(N_tot)

        # The cap particles are at the poles, listed at the end of these arrays.
        self.theta[-2]  = 0.0
        self.theta[-1]  = np.pi
        self.phi[-2]    = 0.0
        self.phi[-1]    = 0.0

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


    def apply_random_rotation(self):
        """
        Rotate the shell with three random Euler angles.
        """
        # Random Euler angles
        alpha   = np.random.rand() * 2*np.pi
        beta    = np.random.rand() * 2*np.pi
        gamma   = np.random.rand() * 2*np.pi
        A2_rot  = get_euler_rotation_matrix(alpha, beta, gamma)

        # Array of position vectors
        A2_pos  = np.array([self.x, self.y, self.z]).transpose()

        # Rotate each position vector
        for i in range(len(A2_pos)):
            A2_pos[i]   = np.dot(A2_rot, A2_pos[i])

        # Unpack positions
        self.x, self.y, self.z  = A2_pos.transpose()

        return


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
            A1_mat_prof: np.ndarray,
            verb=1
        ):
        """
        Generate nested spherical shells of particles to match radial profiles.

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

        @param verb | (opt.) int | verbosity to control printed output:
            0  :  Minimal
            1  :  Standard (default)
            2  :  Extra

        Outputs
        -------

        GenSphereIC.A1_r, .A1_x, .A1_y, .A1_z, .A1_m, .A1_rho, .A1_h, .A1_u,
            .A1_mat
        """
        self.N_picle_des    = N_picle_des
        self.A1_r_prof      = A1_r_prof
        self.A1_rho_prof    = A1_rho_prof
        self.A1_u_prof      = A1_u_prof
        self.A1_mat_prof    = A1_mat_prof

        self.N_prof     = len(self.A1_r_prof)
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
        self.A1_m   = []
        self.A1_r   = []
        self.A1_rho = []
        self.A1_h   = []
        self.A1_u   = []
        self.A1_mat = []
        self.A1_x   = []
        self.A1_y   = []
        self.A1_z   = []

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
        self.N_layer    = len(self.A1_r_bound)

        # Layer info
        if verb >= 1:
            if self.N_layer == 1:
                print(
                    "\nOne layer with outer radius %g      (material %d)" %
                    (self.A1_r_bound, self.A1_m_layer)
                    )
            else:
                print("\n%d layers with outer boundary radii:" % self.N_layer)
                for r_bound, mat in zip(self.A1_r_bound, self.A1_m_layer):
                    print("  %.3e      (material %d)" % (r_bound, mat))

        if verb >= 1:
            print("\nDividing the profile into shells...")

        # First (innermost) layer
        i_layer = 0
        if verb >= 1 and self.N_layer > 1:
            print("\n==== Layer %d ====" % (i_layer + 1))
        r_bound     = self.A1_r_bound[0]
        idx_bound   = self.A1_idx_bound[0]

        # Vary the particle mass until the particle shell boundary matches the
        # profile boundary

        # Start at the maximum allowed particle mass then decrease to fit
        self.m_picle    = self.m_picle_des * 1.01
        dm_picle_init   = 1e-3
        self.dm_picle   = dm_picle_init
        N_shell_init    = 0

        if verb >= 1:
            print("\nTweaking the particle mass to fix the boundaries...")
        if verb == 2:
            print("  Particle mass   Relative tweak ", end='')

        # This also sets the shell data: A1_idx_outer and A1_r_outer
        is_done = False
        while True:
            if verb == 2:
                print("\n  %.5e     %.1e " % (
                    self.m_picle, self.dm_picle
                    ), end='', flush=True)

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

            # Find the shells that fit in this layer
            N_shell = 0
            while True:
                # Calculate the shell width from the profile density relative to
                # the core radius and density
                rho = self.A1_rho_prof[idx_outer]
                dr  = self.dr_core * np.cbrt(self.rho_core / rho)

                # Find the profile radius just beyond this shell (r_outer + dr)
                idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)

                try:
                    r_outer = self.A1_r_prof[idx_outer]
                # Hit outer edge, stop
                except IndexError:
                    # Extend the final shell to include the tiny extra bit of
                    # this layer
                    if is_done:
                        A1_idx_outer[-1]    = idx_bound
                        A1_r_outer[-1]      = r_bound

                    break

                # Record the shell
                A1_idx_outer.append(idx_outer)
                A1_r_outer.append(r_outer)

                N_shell += 1

            if is_done:
                if verb == 2:
                    print("")
                break

            # Number of shells for the starting particle mass
            if N_shell_init == 0:
                N_shell_init    = N_shell

            # Want to reduce the particle mass until one more shell *just* fits

            # Not got another shell yet, so reduce the mass
            if N_shell == N_shell_init:
                self.m_picle    *= 1 - self.dm_picle

            # Got one more shell, but need it to *just* fit, so go back one step
            # and retry with smaller mass changes (repeat this twice!)
            elif self.dm_picle > dm_picle_init * 1e-2:
                if verb == 2:
                    print("  Reduce tweak", end='')

                self.m_picle    *= 1 + self.dm_picle
                self.dm_picle   *= 1e-1

            # Got one more shell and refined the mass so it just fits, so done!
            else:
                # Repeat one more time to extend the final shell to include the
                # tiny extra bit of this layer
                is_done = True

        if verb >= 1:
            print("Done particle mass tweaking!")
        if verb == 2:
            print("  from %.5e to %.5e" % (
                self.m_picle_des, self.m_picle
                ))

        i_layer += 1

        # Outer layer(s)
        while i_layer < self.N_layer:
            if verb >= 1:
                print("\n==== Layer %d ====" % (i_layer + 1))

            r_bound     = self.A1_r_bound[i_layer]
            idx_bound   = self.A1_idx_bound[i_layer]

            # Vary the number of particles in the first shell of this layer
            # until the particle shell boundary matches the profile boundary

            # First find the initial number of particles continuing from the
            # previous layer

            # Calculate the shell width from the profile density
            # relative to the core radius and density
            idx_inner   = self.A1_idx_bound[i_layer - 1] + 1
            rho         = self.A1_rho_prof[idx_inner]
            dr          = self.dr_core * np.cbrt(self.rho_core / rho)

            # Find the profile radius just beyond this shell (r_outer + dr)
            idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)

            # Shell mass and initial number of particles
            m_shell         = sum(self.A1_m_prof[idx_inner:idx_outer])
            N_shell_init    = int(round(m_shell / self.m_picle))
            print("N_shell_init = ", N_shell_init)  ###

            is_done = False
            while True:
                # Find the outer boundary radii of all shells
                A1_idx_outer_tmp    = []
                A1_r_outer_tmp      = []

                # # # # # # WILO

#                # Record shell boundary
#                A1_idx_outer_tmp.append(idx_outer)
#                A1_r_outer_tmp.append(self.dr_core)
#
#                # Find the shells that fit in this layer
#                N_shell = 0
#                while True:
#                    # Calculate the shell width from the profile density
#                    # relative to the core radius and density
#                    rho = self.A1_rho_prof[idx_outer]
#                    dr  = self.dr_core * np.cbrt(self.rho_core / rho)
#
#                    # Find the profile radius just beyond this shell (r_outer +
#                    # dr)
#                    idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)
#
#                    try:
#                        r_outer = self.A1_r_prof[idx_outer]
#                    # Hit outer edge, stop
#                    except IndexError:
#                        # Extend the final shell to include the tiny extra bit
#                        # of this layer
#                        if is_done:
#                            A1_idx_outer_tmp[-1]    = idx_bound
#                            A1_r_outer_tmp[-1]      = r_bound
#
#                        break
#
#                    # Record the shell
#                    A1_idx_outer_tmp.append(idx_outer)
#                    A1_r_outer_tmp.append(r_outer)
#
#                    N_shell += 1
#
#                if is_done:
#                    break
#
#                # Number of shells for the starting particle mass
#                if N_shell_init == 0:
#                    N_shell_init    = N_shell
#
#                # Want to reduce the particle mass until one more shell *just*
#                # fits
#
#                # Not got another shell yet, so reduce the mass
#                if N_shell == N_shell_init:
#                    self.m_picle    *= 1 - self.dm_picle
#
#                # Got one more shell, but need it to *just* fit, so go back one
#                # step and retry with smaller mass changes (repeat this twice!)
#                elif self.dm_picle > dm_picle_init * 1e-2:
#                    if verb == 2:
#                        print("  Reduce tweak", end='')
#
#                    self.m_picle    *= 1 + self.dm_picle
#                    self.dm_picle   *= 1e-1
#
#                # Got one more shell and refined the mass so it just fits, so
#                # done!
#                else:
#                    if verb == 2:
#                        print()
#
#                    # Repeat one more time to extend the final shell to include
#                    # the tiny extra bit of this layer
#                    is_done = True

            # # # # # #

            # Add these to the previous layer(s)' shells
            A1_idx_outer.append(A1_idx_outer_tmp)
            A1_r_outer.append(A1_r_outer_tmp)

            i_layer += 1

        if verb >= 1:
            print("\nDone profile division into shells!")

        if verb >= 1:
            print("\nFinding the values for the particles in each shell...")
        if verb == 2:
            header  = ("  Radius    Number   Mass      Density   Energy    "
                       "Material")
            print(header)

        # Set the particle values for each shell
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
            A1_mat_shell.append(self.A1_mat_prof[idx_inner])

            idx_inner   = idx_outer

            if verb == 2:
                print("  %.2e  %07d  %.2e  %.2e  %.2e  %d" % (
                    A1_r_shell[-1], A1_N_shell[-1], A1_m_picle_shell[-1],
                    A1_rho_shell[-1], A1_u_shell[-1], A1_mat_shell[-1]
                    ))

        if verb == 2:
            print(header)
        if verb >= 1:
            print("Done shell particle values!")

        # Estimate the smoothing lengths from the densities
        num_ngb     = 50
        kernel_edge = 2
        A1_h_shell  = (
            3/(4*np.pi) * num_ngb * np.array(A1_m_shell) / np.cbrt(A1_rho_shell)
            / kernel_edge
            )

        # Generate the particles in each shell
        if verb >= 1:
            print("\nArranging the particles in each shell...")

        for N, m, r, rho, h, u, mat in zip(
            A1_N_shell, A1_m_picle_shell, A1_r_shell, A1_rho_shell, A1_h_shell,
            A1_u_shell, A1_mat_shell
            ):
            self.generate_shell_particles(N, m, r, rho, h, u, mat)

        self.flatten_particle_arrays()

        if verb >= 1:
            print("Done particles!")

        self.N_picle    = len(self.A1_r)
        if verb >= 1:
            print("\nFinal number of particles = %d" % self.N_picle)

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

        Set A1_idx_bound, A1_r_bound, and A1_mat_layer.
        """
        A1_idx_bound    = np.where(np.diff(self.A1_mat_prof) == 1)[0]
        # Include the outer edge
        self.A1_idx_bound   = np.append(A1_idx_bound, self.N_prof - 1)
        self.A1_r_bound     = self.A1_r_prof[self.A1_idx_bound]
        self.A1_m_layer     = self.A1_mat_prof[self.A1_idx_bound]

        return

    def get_tetrahedron_points(self, r: float):
        """
        Return the positions of particles at the vertices of a tetrahedron with
        radius r.
        """
        # Radius scale
        r_scale = r / np.cbrt(3)
        # Tetrahedron vertex coordinates
        A1_x    = np.array([1, 1, -1, -1]) * r_scale
        A1_y    = np.array([1, -1, 1, -1]) * r_scale
        A1_z    = np.array([1, -1, -1, 1]) * r_scale

        return A1_x, A1_y, A1_z

    def generate_shell_particles(
        self, N: int, m: float, r: float, rho: float, h: float, u: float,
        mat: int
        ):
        """
        Make a single spherical shell of particles with the given properties.
        """
        # Append the data to the all-particle arrays
        self.A1_m.append([m] * N)
        self.A1_r.append([r] * N)
        self.A1_rho.append([rho] * N)
        self.A1_h.append([h] * N)
        self.A1_u.append([u] * N)
        self.A1_mat.append([mat] * N)

        # Make a tetrahedron for the central 4 particles
        if N == 4:
            A1_x, A1_y, A1_z    = self.get_tetrahedron_points(r)
            self.A1_x.append(A1_x)
            self.A1_y.append(A1_y)
            self.A1_z.append(A1_z)
        # Make an SEA shell otherwise
        else:
            shell = GenShell(N, r)
            self.A1_x.append(shell.x)
            self.A1_y.append(shell.y)
            self.A1_z.append(shell.z)

        return

    def flatten_particle_arrays(self):
        """
        Flatten the particle data arrays for output.
        """
        for array in ["r", "x", "y", "z", "m", "rho", "h", "u", "mat",]:
            exec("self.A1_%s = np.hstack(self.A1_%s)" % (array, array))

        return














