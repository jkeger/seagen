"""
Objects for SEAGen
"""

import numpy as np


class GenSphere(object):
    """
    """
    def __init__(self, N: int, r: float, dr: float):
        """
        Generates a single sphere.

        Access the particle positions with:
            GenSphere.r
            GenSphere.theta
            GenSphere.phi.

        
        Inputs
        ------

        @param N | integer | the number of cells/particles to create.

        @param r | float | the inner radius of the shell

        @param dr | float | width of the shell.
        """

        self.N = N
        self.r = r
        self.dr = dr

        # Derived Properties
        self.A_reg = 4 * np.pi / N

        self.get_point_positions()
        

    def get_cap(self) -> float:
        """
        Gets the cap colatitude.

        Equation 3.
        """

        return 2 * np.asin(np.sqrt(self.N))


    def get_number_of_collars(self) -> float:
        """
        Gets the number of collars N_col.

        Equation 4.
        """

        theta_cap = self.get_cap()

        self.N_col = int((np.pi - 2 * theta_cap)/(np.sqrt(self.A_reg)))

        return self.N_col


    def get_collar_heights(self) -> np.ndarray:
        """
        Gets the theta of all of the collars, and stores them in
        self.collars, as well as returning them.
        """

        n_collars = self.get_number_of_collars()
        cap_height = self.get_cap()
        height_of_collar = (np.pi - 2 * cap_height) / n_collars

        # Allocate collars array
        self.collars = np.arange(n_collars)
        # Collars have a fixed height
        self.collars *= height_of_collar
        # Need to account for the cap being an 'offset'
        self.collars += (cap_height)

        return self.collars


    def get_area_of_collar(
            self,
            theta_i: float,
            theta_i_minus_one: float
        ) -> float:
        """
        Gets the area of a collar given the collar heights of itself and its
        neighbor.

        Equation 5.
        """
        
        s_t_i = np.sin(theta_i / 2)**2
        s_t_i_m_o = np.sin(theta_i_minus_one / 2)**2

        return 4 * np.pi * (s_t_i - s_t_i_m_o)


    def get_collar_areas(self) -> np.ndarray:
        """
        Gets the collar areas and stores them in self.collar_areas.
        """

        collar_heights = self.get_collar_heights()

        self.collar_areas = np.empty(self.N_col)
        
        # The cap has an area of A_reg
        self.collar_areas[0] = self.A_reg

        self.collar_areas[1:] = self.get_area_of_collar(
            collar_heights[1:],
            collar_heights[:-1]
        )

        return self.collar_areas


    def get_ideal_number_of_regions_in_collar(self, A: float) -> float:
        """
        Gets the ideal number of regions in a collar.

        Equation 7.
        """

        return A / self.A_reg


    def get_reigons_in_collars(self) -> np.ndarray:
        """
        Gets the number of regions in each collar.

        Stores them in self.n_regions_in_collars.

        Equation 8/9.
        """
        
        # Because of the discrepancy counter, we will just use a regular loop.

        self.n_regions_in_collars = np.empty(self.N_col, dtype=int)
        areas = get_collar_areas()

        loop = np.ndenumerate(
            np.nditer(self.n_regions_in_collars, opflags=["readwrite"])
        )
        
        discrepancy = 0

        for N_i, i in loop:
            ideal_number = get_ideal_number_of_regions_in_collar(areas[i])
            N_i = int(ideal_number + discrepancy)

            discrepancy += (N_i - ideal_number)


        return self.n_regions_in_collars


    def update_collar_heights(self) -> np.ndarray:
        """
        After get_regions_in_collars, we must update the collar heights due to
        there being slightly different numbers of regions in each collar than
        expected.

        Also returns self.collars.

        Equation 10.
        """

        # First we must get the cumulative sums of the particle numbers
        n_regions = self.get_regions_in_collars()

        # Unsure if this is correct; should we be including 'self' terms?
        summed = np.cumsum(n_regions)

        self.collars = 2 * np.asin(np.sqrt(summed * self.A_reg / (4 * np.pi)))

        return self.collars


    def choose_phi_0(self,
            N_i: float,
            N_i_plus_one: float
            d_phi_i: float,
            d_phi_i_plus_one: float,
        ) -> float:
        """
        Choose the value of phi0. This comes from the discussion paragraph
        after Equation 12.
        """

        N_i_even = (N_i % 2) - 1
        N_i_plus_one_even = (N_i_plus_one % 2) - 1

        if N_i_even != N_i_plus_one_even:
            # Exclusive or
            return 0.5 * (N_i_even * d_phi_i +
                          N_i_plus_one_even * d_phi_i_plus_one)
        else:
            return max(d_phi_i, d_phi_i_plus_one)


    def get_point_positions(self) -> np.ndarray, np.ndarray:
        """
        Gets the point positions (theta and phi) using the above
        calculated data. Also returns theta, phi.

        Stores in self.theta, self.phi

        Equation 11/12.
        """

        total_number_of_particles = self.n_regions_in_collars.sum()

        self.theta = np.empty(total_number_of_particles)
        self.phi = np.empty(total_number_of_particles)
    
        # All regions in a collar 'share' a theta.
        theta = 0.5 * (self.collars[:-1], self.collars[1:])


        # The 'phi' array is somewhat more complicated.

        d_phi = 2 * np.pi / self.n_regions_in_collars
        phi_0 = np.empty(self.N_col)

        loop = np.ndenumerate(
            np.nditer(phi_0_i, opflags=["writeonly"])
        )

        for phi_0_i, i in loop:
            try:
                phi_0_i = self.choose_phi_0(
                    self.n_regions_in_collars[i],
                    self.n_regions_in_collars[i-1],
                    d_phi[i],
                    d_phi[i-1]
                )

                # Also need to do the offset to ensure that successive collars
                # No not create overlaps.
                m = np.randint(0, self.n_regions_in_collars[i-1])
                phi_0_i += (m * d_phi[i-1])

            except IndexError:
                # This must be the first element, which is at the poles.
                # (i.e. phi can be anything...)
                phi_0_i = 0


        # We can now fill the arrays.
        total_number_covered = 0
        loop = np.ndenumerate(
            np.nditer(self.n_regions_in_collars, opflags=["readonly"])
        )

        for number_of_parts, region in loop:
            upper_bound = number_of_parts + total_number_covered

            # Update theta
            self.theta[total_number_covered:upper_bound] = theta[region]
            
            # Equation 12
            these_phi = phi_0[region] + np.arange(number_of_parts) * d_phi[region]
            self.phi[total_number_covered:upper_bound] = these_phi

            total_number_covered = upper_bound
        

        return self.theta, self.phi








    
