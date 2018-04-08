"""
Objects for SEAGen

Created by: Josh Borrow (joshua.borrow@durham.ac.uk)

This file includes:

    + GenShell, an object for generating individual spheres of particles
      using the SEA method

    + GenIC, an object for generating a whole _sphere_ of particles using
      the SEA method; this creates several GenShell objects.

Using these you are able to get the particle positions in spherical polar
co-ordinates; note that here:

    + r = [0, inf)
    + phi = [0, 2pi)
    + theta = [0, pi]

unlike the usual definition. This is chosen to be compatible with the paper.
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


class GenIC(object):
    """
    """
    def __init__(
            self,
            density: Callable,
            part_mass: float,
            r_range: Tuple[float]
        ):
        """
        Generates a whole set of particles to represent a filled-in sphere
        by creating several GenShell objects.

        Inputs
        ------

        @param density | callable | a function that describes density as a
                                    function of radius. Should take a single
                                    argument such that density = density(r).

        @param part_mass | float | particle mass.

        @param r_range | tuple[float] | a tuple describing the range of r
                                        values to create particles over. At
                                        the moment, the lower bound is ignored.


        Outputs
        -------

        GenIC.r, GenIC.theta, GenIC.phi. Note that here:

        + r = [0, inf)
        + phi = [0, 2pi)
        + theta = [0, pi]
        """
        self.density = density
        self.part_mass = part_mass
        self.r_range = r_range

        self.r = []
        self.theta = []
        self.phi = []

        # Do processing
        self.calculate_total_mass()
        self.calculate_parts_in_shells()
        self.create_all_shells()

        # Turn our list of arrays into a single long array.
        self.stack_arrays()


    def make_single_shell(self, N: int, r_inner: float, dr: float):
        """
        Make a single spherical shell (i.e. one row of positions).
        """
        this_shell = GenShell(N=N, r_inner=r_inner, dr=dr)
        this_shell.apply_stretch_factor()

        self.r.append(this_shell.r)
        self.theta.append(this_shell.theta)
        self.phi.append(this_shell.phi)

        return


    def mass_in_profile_shell(self, r: float) -> float:
        """
        The mass in an infinitesimal shell at profile radius r.
        (The density profile function multiplied by the jacobian.)
        """

        return 4 * np.pi * r * r * self.density(r)


    def calculate_total_mass(self) -> float:
        """
        Calculate the total mass using the density callable.

        Stores the total mass as self.total_mass.
        """

        m, _ = quad(self.mass_in_profile_shell, *self.r_range)

        self.total_mass = m

        return m


    def make_central_tetra(self):
        """
        Sets the central tetrahedron paritcle positions, based on particle mass
        and inner density.
        Creates:
            self.parts_in_shells
            self.shell_widths
            self.inner_radii.
        """

        inner_density = self.density(0)
        volume = self.part_mass / inner_density

        # We actually care about a _sphere_ that encloses the
        # tetrahedrons rather than the volume of the tetra itself.
        ## Need to find the radius that encloses 4*m_part for non-const density!
        radius = np.cbrt(3 * volume / (4 * np.pi))

        self.shell_widths = [radius]
        self.parts_in_shells = [4]
        self.inner_radii = [radius] ## Not zero?

        # Get intial particle positions.

        self.r.append([radius] * 4)
        self.theta.append(
            [2.186276, 0.955316, 0.955316, 2.186276]
        )
        self.phi.append(
            [2*np.pi-2.356194, 2.356194, 2*np.pi-0.785398, 0.785398]
        )

        return


    def calculate_parts_in_shells(self):
        """
        Calculate the (approximate) number of particles in each shell.

        Stores the numbers as self.parts_in_shells, and the shell widths in
        self.shell_widths. Stores inner radii as self.inner_radii.
        """
        n_parts = self.total_mass / self.part_mass

        self.make_central_tetra()

        prefactor = 4 * np.pi / 3

        while (self.inner_radii[-1] + self.shell_widths[-1]) < self.r_range[1]:
            dr_core = self.shell_widths[0]
            radius_core = self.inner_radii[-1] + dr_core
            self.inner_radii.append(radius_core)

            tot_parts = sum(self.parts_in_shells)
            density_core = self.part_mass * tot_parts / (prefactor * radius_core**3)
            density = self.density(radius_core)

            print(density, density_core)
            dr = dr_core * np.cbrt(density_core / density)

            ## Need to calculate the actual enclosed mass in the particle shell!
            mass_in_profile_shell, _ = quad(
                self.mass_in_profile_shell,
                radius_core,
                radius_core+dr
            )
            parts_in_shell = int(round(mass_in_profile_shell / self.part_mass))

            self.shell_widths.append(dr)
            self.parts_in_shells.append(parts_in_shell)

#            print(f"dr: {dr}, parts: {parts_in_shell}, inner: {radius_core}")


        return


    def create_all_shells(self):
        """
        Create all shells! Essentially loops over all of the parameters.
        """

        n_shells = len(self.shell_widths)

        for shell in tqdm(range(n_shells)):
            self.make_single_shell(
                N=self.parts_in_shells[shell],
                r_inner=self.inner_radii[shell],
                dr=self.shell_widths[shell]
            )

        return


    def stack_arrays(self):
        """
        Stack the r, theta, phi lists into one long array for output.
        """

        self.r = np.hstack(self.r)
        self.theta = np.hstack(self.theta)
        self.phi = np.hstack(self.phi)

        return


