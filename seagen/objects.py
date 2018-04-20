"""
This file is part of SEAGen.
Copyright (C) 2018 Jacob Kegerreis (jacob.kegerreis@durham.ac.uk)
GNU General Public License http://www.gnu.org/licenses/

Jacob Kegerreis and Josh Borrow

Objects for SEAGen

This file includes:
    + GenShell, an object for generating individual spherical shells of
      particles using the SEA method.
    + GenSphereIC, an object for generating spherical initial conditions of
      particles in nested shells.

Notation:
    + Arrays of dimension * are explicitly labelled as A*_name
    + Particle is abbreviated as picle
    + Spherical polars: theta = zenith (colatitude), phi = azimuth (longitude)
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


        Inputs
        ------

        @param N | integer | the number of cells/particles to create.

        @param r | float | the radius of the shell.

        @param do_stretch | (opt.) bool | set False to not do the SEA method's
            latitude stretching (default: True).

        @param do_stretch | (opt.) bool | set True to randomly rotate the
            sphere of particles after their intial placement (default: True).


        Outputs
        -------

        Particle position arrays:
            GenShell.A1_x
            GenShell.A1_y
            GenShell.A1_z

        (Spherical polar coordinates are used internally but do not have the
        final rotation applied to them.)


        Notation
        ------

        theta: zenith (colatitude)
        phi: azimuth (longitude)
        """
        self.N      = N
        self.A1_r   = r * np.ones(N)

        # Derived properties
        self.A_reg  = 4 * np.pi / N

        # Start in spherical polar coordinates for the initial placement
        self.get_collar_areas()
        self.update_collar_colatitudes()
        self.get_point_positions()
        if do_stretch:
            self.apply_stretch_factor()

        # Now convert to cartesian coordinates for the rotation and output
        self.A1_x, self.A1_y, self.A1_z = polar_to_cartesian(
            self.A1_r, self.A1_theta, self.A1_phi
            )
        if do_rotate:
            self.apply_random_rotation()


    def get_cap_colatitude(self) -> float:
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

        theta_cap   = self.get_cap_colatitude()

        self.N_col  = int(round((np.pi - 2 * theta_cap)/(np.sqrt(self.A_reg))))

        return self.N_col


    def get_collar_colatitudes(self) -> np.ndarray:
        """
        Gets the top A1_theta of all of the collars, including the bottom cap's
        A1_theta, and stores them in self.A1_collar_theta as well as returning
        them.
        """
        self.get_number_of_collars()

        cap_height          = self.get_cap_colatitude()
        height_of_collar    = (np.pi - 2 * cap_height) / self.N_col

        # Allocate collars array
        self.A1_collar_theta    = np.arange(self.N_col + 1, dtype=float)
        # Collars have a fixed height initially
        self.A1_collar_theta    *= height_of_collar
        # Starting at the bottom of the top polar cap
        self.A1_collar_theta    += (cap_height)

        return self.A1_collar_theta


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
        sin2_theta_i        = np.sin(theta_i / 2)**2
        sin2_theta_i_m_o    = np.sin(theta_i_minus_one / 2)**2

        return 4 * np.pi * (sin2_theta_i - sin2_theta_i_m_o)


    def get_collar_areas(self) -> np.ndarray:
        """
        Gets the collar areas and stores them in self.A1_collar_area.
        """
        self.get_collar_colatitudes()

        self.A1_collar_area = np.empty(self.N_col)

        self.A1_collar_area[:]  = self.get_collar_area(
            self.A1_collar_theta[1:],
            self.A1_collar_theta[:-1]
        )

        return self.A1_collar_area


    def get_ideal_N_regions_in_collar(self, A_col: float) -> float:
        """
        Gets the ideal number of regions in a collar.

        Equation 7.
        """
        return A_col / self.A_reg


    def get_N_regions_in_collars(self) -> np.ndarray:
        """
        Gets the number of regions in each collar.

        Stores them in self.A1_N_reg_in_collar.

        Equation 8,9.
        """
        A1_N_reg_in_collar  = np.empty(self.N_col, dtype=int)
        collar_areas        = self.get_collar_areas()

        loop    = enumerate(
            np.nditer(A1_N_reg_in_collar, op_flags=["readwrite"])
        )

        discrepancy = 0

        for i, N_i in loop:
            N_reg_ideal = self.get_ideal_N_regions_in_collar(collar_areas[i])
            N_i[...]    = int(round(N_reg_ideal + discrepancy))

            discrepancy += N_reg_ideal - N_i

        self.A1_N_reg_in_collar = A1_N_reg_in_collar

        return self.A1_N_reg_in_collar


    def update_collar_colatitudes(self) -> np.ndarray:
        """
        After get_N_regions_in_collars, we must update the collar thetas due to
        the now integer numbers of regions in each collar instead of the ideal.

        Also returns self.A1_collar_theta.

        Equation 10.
        """
        # First we must get the cumulative number of regions in each collar,
        # including the top polar cap
        A1_N_reg_in_collar_cum  = np.cumsum(self.get_N_regions_in_collars()) + 1
        A1_N_reg_in_collar_cum  = np.append([1], A1_N_reg_in_collar_cum)

        self.A1_collar_theta    = 2 * np.arcsin(
            np.sqrt(A1_N_reg_in_collar_cum * self.A_reg / (4 * np.pi))
            )

        return self.A1_collar_theta


    def choose_longitude_offset(self,
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
        N_i_even            = abs((N_i % 2) - 1)
        N_i_minus_one_even  = abs((N_i_minus_one % 2) - 1)

        if N_i_even != N_i_minus_one_even:
            # Exclusive or
            return 0.5 * (N_i_even * d_phi_i +
                          N_i_minus_one_even * d_phi_i_minus_one)
        else:
            return 0.5 * min(d_phi_i, d_phi_i_minus_one)


    def get_point_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sets the point positions in the centres of every region.

        Stores in self.A1_theta, self.A1_phi and also returns them.

        Equation 11,12.
        """
        N_tot   = self.A1_N_reg_in_collar.sum() + 2

        self.A1_theta   = np.empty(N_tot)
        self.A1_phi     = np.empty(N_tot)

        # The cap particles are at the poles, listed at the end of these arrays.
        self.A1_theta[-2]   = 0.0
        self.A1_theta[-1]   = np.pi
        self.A1_phi[-2]     = 0.0
        self.A1_phi[-1]     = 0.0

        # All regions in a collar are at the same colatitude, theta.
        A1_theta        = np.zeros(self.N_col + 2)
        A1_theta[:-2]   = 0.5 * (
            self.A1_collar_theta[:-1] + self.A1_collar_theta[1:]
            )

        # Particles in each collar are equally spaced in longitude, phi,
        # and offset appropriately from the previous collar.
        A1_d_phi    = 2 * np.pi / self.A1_N_reg_in_collar
        A1_phi_0    = np.empty(self.N_col)

        loop = enumerate(
            np.nditer(A1_phi_0, op_flags=["writeonly"])
        )

        for i, phi_0_i in loop:
            # The first collar has no previous collar to rotate away from
            # so doesn't need an offset.
            if i == 0:
                phi_0_i = 0
            else:
                phi_0_i = self.choose_longitude_offset(
                    self.A1_N_reg_in_collar[i],
                    self.A1_N_reg_in_collar[i-1],
                    A1_d_phi[i],
                    A1_d_phi[i-1]
                )

                # Also add a random initial offset to ensure that successive
                # collars do not create lines of ~adjacent particles.
                # (Second paragraph following Equation 12.)
                m       = np.random.randint(0, self.A1_N_reg_in_collar[i-1])
                phi_0_i += (m * A1_d_phi[i-1])

        # Fill the position arrays.
        loop = enumerate(
            np.nditer(self.A1_N_reg_in_collar, op_flags=["readonly"])
        )

        N_regions_done  = 0
        for region, N_regions_in_collar in loop:
            N_regions_done_next = N_regions_in_collar + N_regions_done

            # Set A1_theta
            self.A1_theta[N_regions_done:N_regions_done_next] = A1_theta[region]

            # Set phi (Equation 12)
            j               = np.arange(N_regions_in_collar, dtype=float)
            A1_phi_collar   = A1_phi_0[region] + j * A1_d_phi[region]

            self.A1_phi[N_regions_done:N_regions_done_next] = A1_phi_collar

            N_regions_done  = N_regions_done_next

        self.A1_phi         %= 2 * np.pi
        self.A1_theta       %= np.pi
        self.A1_theta[-1]   = np.pi

        return self.A1_theta, self.A1_phi


    def apply_stretch_factor(self, a=0.2, b=2.0):
        """
        Apply the SEA stretch factor.

        Equation 13.
        """
        pi_over_2   = np.pi / 2
        inv_sqrtN   = 1 / np.sqrt(self.N)

        A1_prefactor    = (pi_over_2 - self.A1_theta) * a * inv_sqrtN

        A1_exp_factor   = - (
            (pi_over_2 - abs(pi_over_2 - self.A1_theta))
            / (np.pi * b * inv_sqrtN)
            )

        self.A1_theta   += (A1_prefactor * np.exp(A1_exp_factor))

        # Leave the cap points at the poles
        self.A1_theta[-2]   = 0.0
        self.A1_theta[-1]   = np.pi

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
        A2_pos  = np.array([self.A1_x, self.A1_y, self.A1_z]).transpose()

        # Rotate each position vector
        for i in range(len(A2_pos)):
            A2_pos[i]   = np.dot(A2_rot, A2_pos[i])

        # Unpack positions
        self.A1_x, self.A1_y, self.A1_z  = A2_pos.transpose()

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
            0  :  None
            1  :  Standard (default)
            2  :  Extra
            3  :  Debug

        Outputs
        -------

        Particle data arrays:
            GenSphereIC.A1_x, A1_y, A1_z, A1_r, A1_m, A1_rho, A1_h, A1_u, A1_mat
        """
        # ========
        # Setup
        # ========
        self.N_picle_des    = N_picle_des
        self.A1_r_prof      = A1_r_prof
        self.A1_rho_prof    = A1_rho_prof
        self.A1_u_prof      = A1_u_prof
        self.A1_mat_prof    = A1_mat_prof

        # Verbosity
        if verb >= 1:
            verb_options    = {
                0: "None", 1: "Standard", 2: "Extra", 3: "Debug"
                }
            print("\nVerbosity %d: %s printing" % (verb, verb_options[verb]))

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

        # Check profiles start from non-zero radius
        ...

        # Interpolate profiles if not dense enough in radius
        ...

        # Calculate the mass profile
        self.get_mass_profile()
        # Enclosed mass profile
        self.A1_m_enc_prof  = np.cumsum(self.A1_m_prof)
        self.m_tot          = self.A1_m_enc_prof[-1]
        self.m_picle_des    = self.m_tot / self.N_picle_des

        # Max allowed particle mass
        m_picle_max = self.m_picle_des * 1.01
        # Initial relative particle mass tweak
        dm_picle_init   = 1e-3
        # SPH Kernel: number of neighbours (approximate) and r/h at which the
        # kernel goes to zero
        num_ngb     = 50
        kernel_edge = 2

        # Find the radii of all material boundaries (including the outer edge)
        self.find_material_boundaries()
        self.N_layer    = len(self.A1_r_bound)

        if verb >= 1:
            print("\n%d layer(s):" % self.N_layer)
            print(
                "    Outer radius   Mass          Material"
                )
            for r_bound, idx_bound, mat in zip(
                self.A1_r_bound, self.A1_idx_bound, self.A1_m_layer
                ):
                print(
                    "    %5e   %.5e   %d" %
                    (r_bound, self.A1_m_enc_prof[idx_bound], mat)
                    )

            print("\n> Divide the profile into shells")

        # ================
        # First (innermost) layer
        # ================
        i_layer = 0
        if verb >= 1 and self.N_layer > 1:
            print("\n==== Layer %d ====" % (i_layer + 1))

        idx_bound   = self.A1_idx_bound[0]
        r_bound     = self.A1_r_bound[0]

        # ========
        # Vary the particle mass until the particle shell boundary matches the
        # profile boundary
        # ========
        # Start at the maximum allowed particle mass then decrease to fit
        self.m_picle    = m_picle_max
        self.dm_picle   = dm_picle_init
        N_shell_init    = 0

        if verb >= 1:
            print("\n> Tweak the particle mass to fix the outer boundary")
        if verb == 3:
            print("    Particle mass   Relative tweak ", end='')

        # This also sets the shell data: A1_idx_outer and A1_r_outer
        is_done = False
        while True:
            if verb == 3:
                # No endline so can add more on this line in the loop
                print(
                    "\n  %.5e     %.1e " % (self.m_picle, self.dm_picle),
                    end='', flush=True
                    )

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

            # ========
            # Find the shells that fit in this layer
            # ========
            N_shell = 0
            while True:
                # Calculate the shell width from the profile density relative to
                # the core radius and density
                rho = self.A1_rho_prof[idx_outer]
                dr  = self.dr_core * np.cbrt(self.rho_core / rho)

                # Find the profile radius just beyond this shell (r_outer + dr)
                idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)

                # Hit outer edge, stop
                if idx_outer >= idx_bound:
                    # Extend the final shell to include the tiny extra bit of
                    # this layer
                    if is_done:
                        A1_idx_outer[-1]    = idx_bound
                        A1_r_outer[-1]      = r_bound

                    break
                r_outer = self.A1_r_prof[idx_outer]

                # Record the shell
                A1_idx_outer.append(idx_outer)
                A1_r_outer.append(r_outer)

                N_shell += 1

            if is_done:
                if verb == 3:
                    print("")
                break

            # Number of shells for the starting particle mass
            if N_shell_init == 0:
                N_shell_init    = N_shell

            # ========
            # Reduce the particle mass until one more shell *just* fits
            # ========
            # Not got another shell yet, so reduce the mass
            if N_shell == N_shell_init:
                self.m_picle    *= 1 - self.dm_picle

            # Got one more shell, but need it to *just* fit, so go back one step
            # and retry with smaller mass changes (repeat this twice!)
            elif self.dm_picle > dm_picle_init * 1e-2:
                if verb == 3:
                    print("  Reduce tweak", end='')

                self.m_picle    *= 1 + self.dm_picle
                self.dm_picle   *= 1e-1

            # Got one more shell and refined the mass so it just fits, so done!
            else:
                # Repeat one more time to extend the final shell to include the
                # tiny extra bit of this layer
                is_done = True

        if verb >= 1:
            print("> Done particle mass tweaking!")
        if verb >= 2:
            print("    from %.5e to %.5e" % (
                self.m_picle_des, self.m_picle
                ))
        if verb >= 1:
            print("\n%d shells in layer %d" % (N_shell, i_layer + 1))

        i_layer += 1

        # ================
        # Outer layer(s)
        # ================
        while i_layer < self.N_layer:
            if verb >= 1:
                print("\n==== Layer %d ====" % (i_layer + 1))

            r_bound     = self.A1_r_bound[i_layer]
            idx_bound   = self.A1_idx_bound[i_layer]

            # ========
            # First find the initial number of particles continuing from the
            # previous layer
            # ========
            # Calculate the shell width from the profile density
            # relative to the core radius and density
            idx_inner   = self.A1_idx_bound[i_layer - 1]
            r_inner     = self.A1_r_bound[i_layer - 1]
            rho         = self.A1_rho_prof[idx_inner]
            dr          = self.dr_core * np.cbrt(self.rho_core / rho)

            # Find the profile radius just beyond this shell (r_outer + dr)
            idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)

            # Shell mass and initial number of particles
            m_shell         = sum(self.A1_m_prof[idx_inner:idx_outer])
            N_picle_shell   = int(round(m_shell / self.m_picle))
            N_picle_init    = N_picle_shell

            # ========
            # Vary the number of particles in the first shell of this layer
            # until the particle shell boundary matches the profile boundary
            # ========
            if verb >= 1:
                print("\n> Tweak the number of particles in the first shell "
                      "to fix the outer boundary")
            if verb == 3:
                print("    Number   1st shell width", end='')

            # Initialise
            N_shell_init    = 0
            dN_picle_shell  = 1
            is_done         = False
            while True:
                if verb == 3:
                    # No endline so can add more on this line in the loop
                    print("\n  %d      " % N_picle_shell, end='')

                # Find the outer boundary radii of all shells
                A1_idx_outer_tmp    = []
                A1_r_outer_tmp      = []

                # Set the starting dr by the shell that contains the mass of
                # N_picle_shell particles, instead of continuing to use dr_core
                idx_outer   = idx_inner + np.searchsorted(
                    self.A1_m_enc_prof[idx_inner:]
                    - self.A1_m_enc_prof[idx_inner],
                    N_picle_shell * self.m_picle
                    )
                r_outer     = self.A1_r_prof[idx_outer]
                self.dr_0   = r_outer - r_inner

                if verb == 3:
                    print("%.3e" % self.dr_0, end='', flush=True)

                # Mass-weighted mean density
                self.rho_0  = get_mass_weighted_mean(
                    self.A1_m_prof[idx_inner:idx_outer],
                    self.A1_rho_prof[idx_inner:idx_outer]
                    )

                # Record shell boundary
                A1_idx_outer_tmp.append(idx_outer)
                A1_r_outer_tmp.append(r_outer)

                # ========
                # Find the shells that fit in this layer
                # ========
                N_shell = 0
                while True:
                    # Calculate the shell width from the profile density
                    # relative to the first shell in this layer
                    rho = self.A1_rho_prof[idx_outer]
                    dr  = self.dr_0 * np.cbrt(self.rho_0 / rho)

                    # Find the profile radius just beyond this shell (r_outer +
                    # dr)
                    idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)

                    # Hit outer edge, stop
                    if idx_outer >= idx_bound:
                        # Extend the final shell to include the tiny extra bit
                        # of this layer
                        if is_done:
                            A1_idx_outer_tmp[-1]    = idx_bound
                            A1_r_outer_tmp[-1]      = r_bound

                        break
                    r_outer = self.A1_r_prof[idx_outer]

                    # Record the shell
                    A1_idx_outer_tmp.append(idx_outer)
                    A1_r_outer_tmp.append(r_outer)

                    N_shell += 1

                if is_done:
                    if verb == 3:
                        print()
                    break

                # Number of shells for the initial number of particles
                if N_shell_init == 0:
                    N_shell_init    = N_shell

                # ========
                # Change the number of particles in the first shell until either
                # one more shell just fits or just until this number of shells
                # just fits
                # ========
                # Got one more shell, so done!
                if N_shell == N_shell_init + 1:
                    # Repeat one more time to extend the final shell to include
                    # the tiny extra bit of this layer
                    is_done = True

                # Got one less shell, so go back one step then done!
                elif N_shell == N_shell_init - 1:
                    N_picle_shell   -= 1

                    # Repeat one more time to extend the final shell to include
                    # the tiny extra bit of this layer
                    is_done = True

                # Not yet done so vary the number of particles in the first
                # shell (i.e. try: N-1, N+1, N-2, N+2, ...)
                else:
                    N_picle_shell   += dN_picle_shell
                    dN_picle_shell  = (
                        -np.sign(dN_picle_shell) * (abs(dN_picle_shell) + 1)
                        )

            # Add these to the previous layer(s)' shells
            A1_idx_outer.append(A1_idx_outer_tmp)
            A1_r_outer.append(A1_r_outer_tmp)

            if verb >= 1:
                print("> Done first-shell particle number tweaking!")
            if verb >= 2:
                print("    from %d to %d" % (
                    N_picle_init, N_picle_shell
                    ))
            if verb >= 1:
                print("\n%d shells in layer %d" % (N_shell, i_layer + 1))

            i_layer += 1

        # Stack all layers' shells together
        A1_idx_outer    = np.hstack(A1_idx_outer)
        A1_r_outer      = np.hstack(A1_r_outer)

        if verb >= 1:
            print("\n> Done profile division into shells!")

        if verb >= 1:
            print("\n> Find the values for the particles in each shell")
        if verb >= 2:
            header  = ("    Radius    Number   Mass      Density   Energy    "
                       "Material")
            print(header)

        # ================
        # Set the particle values for each shell
        # ================
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
            r_mw    = get_mass_weighted_mean(
                A1_m_prof_shell, self.A1_r_prof[idx_inner:idx_outer]
                )
            A1_r_shell.append((r_half + r_mw) / 2)

            # Other properties
            A1_rho_shell.append(get_mass_weighted_mean(
                A1_m_prof_shell, self.A1_rho_prof[idx_inner:idx_outer]
                ))
            A1_u_shell.append(get_mass_weighted_mean(
                A1_m_prof_shell, self.A1_u_prof[idx_inner:idx_outer]
                ))
            A1_mat_shell.append(self.A1_mat_prof[idx_inner])

            idx_inner   = idx_outer

            if verb >= 2:
                print("    %.2e  %07d  %.2e  %.2e  %.2e  %d" % (
                    A1_r_shell[-1], A1_N_shell[-1], A1_m_picle_shell[-1],
                    A1_rho_shell[-1], A1_u_shell[-1], A1_mat_shell[-1]
                    ))

        if verb >= 2:
            print(header)
        if verb >= 1:
            print("> Done shell particle values!")

        # Estimate the smoothing lengths from the densities
        A1_h_shell  = (
            3/(4*np.pi) * num_ngb * np.array(A1_m_shell) / np.cbrt(A1_rho_shell)
            / kernel_edge
            )

        # ================
        # Generate the particles in each shell
        # ================
        if verb >= 1:
            print("\n> Arrange the particles in each shell")

        for N, m, r, rho, h, u, mat in zip(
            A1_N_shell, A1_m_picle_shell, A1_r_shell, A1_rho_shell, A1_h_shell,
            A1_u_shell, A1_mat_shell
            ):
            self.generate_shell_particles(N, m, r, rho, h, u, mat)

        self.flatten_particle_arrays()

        if verb >= 1:
            print("> Done particles!")

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
            self.A1_x.append(shell.A1_x)
            self.A1_y.append(shell.A1_y)
            self.A1_z.append(shell.A1_z)

        return


    def flatten_particle_arrays(self):
        """
        Flatten the particle data arrays for output.
        """
        for array in ["r", "x", "y", "z", "m", "rho", "h", "u", "mat",]:
            exec("self.A1_%s = np.hstack(self.A1_%s)" % (array, array))

        return

















