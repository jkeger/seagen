"""
Some miniature-SPH (that is _very slow_) to help with the initial conditions
generator.

Created by: Josh Borrow (joshua.borrow@gmail.com)
"""

import numpy as np

from numba import vectorize, float64
from scipy.optimize import root
from typing import Callable
from tqdm import tqdm

from .secant import run_iterations


@vectorize([float64(float64, float64)])
def cubic_spline(r, h):
    """
    The un-vectorized version of the cubic spline kernel, which is then vectorized
    by numba.
    
    Inputs
    ------
    
    @param r | float | distance between the two particles
    
    @param h | float | smoothing length of the particle i.
    
    
    Outputs
    -------
    
    @output W | float | the kernel at r/h.
    """
    
    sigma_3 = 1 / (np.pi * h * h * h)
    q = abs(r / h)
    
    if q <= 1.0:
        q2 = q * q
        W = 1.0 - 1.5 * q2 * (1.0 - 0.5 * q)
        W *= sigma_3
    elif q <= 2.0:
        two_minus_q = 2 - q
        two_minus_q_c = np.power(two_minus_q, 3)
        W = 0.25 * two_minus_q_c
        W *= sigma_3
    else:
        W = 0
        
    return W


class EquationOfState(object):
    """
    An equation of state object. This will generate an equation of state given
    normalisation factors, etc.
    """
    def __init__(self, gamma=5./3., entropy=1.):
        """
        Equation of state:
                P = A \rho^\gamma .
        
        Inputs
        ------
        
        @param gamma | float | the hydro_gamma for the equation of state.
        
        @param entropy | float | the fixed entropy; this acts as a normalisation.
        
        
        Outputs
        -------
        
        You can access the equation of state through the following functions:
        
        EquationOfState.get_pressure(density)
        EquationOfState.get_internal_energy(density)
        """
        
        self.gamma = gamma
        self.entropy = entropy
        
        return
        
    
    def get_pressure(self, density: np.ndarray) -> np.ndarray:
        """
        Get the pressure from the internal energy.
        
        Inputs
        ------
        
        @param density | np.ndarray | the SPH densities of the particles
        
        
        Outputs
        -------
        
        @output P | np.ndarray | the pressures of the particles.
        """
        
        return self.entropy * np.power(density, self.gamma)
    
    
    def get_internal_energy(self, density: np.ndarray) -> np.ndarray:
        """
        Gets the internal energy per unit mass.
        
        Inputs
        ------
        
        @param density | np.ndarray | the SPH densities of the particles
        
        
        Outputs
        -------
        
        @output u | np.ndarray | the internal energies of the particles.
        """
        
        g_minus_one = self.gamma - 1
        
        prefactor = A / g_minus_one
        
        density_power = np.power(density, g_minus_one)
        
        return prefactor * density_power
    

class SPHDataset(object):
    """
    An SPH dataset object. This will calculate the following properties:
    
    + Density
    + Smoothing Length
    + Internal Energy
    + Pressure
    
    for all of the particles, when given a kernel, particles, and an equation
    of state.
    """
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            m: np.ndarray,
            eta: float,
            h_init: float,
            kernel: Callable,
            eos: EquationOfState
        ):
        """
        Note that this is not a particularly memory-lean or quick SPH code.
        
        Inputs
        ------
        
        @param x, y, z | np.ndarray | x, y, z, positions of particles
        
        @param m | np.ndarray | the particle masses
        
        @param eta | float | kernel eta used to constrain smoothing lengths
   
        @param h_init | float | initial smoothing length for all particles
        
        @param kernel | callable | kernel function (r, h)
        
        @param eos | EquationOfState | equation of state object.
        
        
        Outputs
        -------
        
        SPHDataset.densities, SPHDataset.hsml, SPHDataset.u, SPHDataset.pressure.
        """
        
        self.x = x
        self.y = y
        self.z = z
        self.m = m
        
        self.eta = eta
        self.h_init = h_init
        self.kernel = kernel
        self.eos = eos
        
        self.hsml = np.ones_like(self.x) * self.h_init
        
        print("Running root-finding for h")
        self.update_h()
        
        return
    
    
    def calculate_r(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Calculates an n long array of r displacements.
        """
        
        dx = np.square(self.x - x)
        dy = np.square(self.y - y)
        dz = np.square(self.z - z)
        
        dr = np.sqrt(dx + dy + dz)
        
        del dx, dy, dz
        
        return dr
    
    
    def calculate_kernels(self, dr: np.ndarray, h: float) -> np.ndarray:
        """
        Calculate the kernel values. Returns them.
        """
        
        return self.kernel(dr, h)


    def calculate_density(self, x, y, z, h):
        """
        Calculate density for a single particle.
        """

        dr = self.calculate_r(x, y, z)

        kernels = self.calculate_kernels(dr, h)

        weighted = kernels * self.m

        return weighted.sum()

    
    def get_density(self, h: np.ndarray) -> np.ndarray:
        """
        Gets the density by summing over kernels.
        
        The smoothing lengths are passed as a parameter h for compatibility with the
        root finding algorithms.
        
        Stores it in self.densities and returns it.
        """

        self.densities = np.array([
            self.calculate_density(
                x=x,
                y=y,
                z=z,
                h=hsml
            ) for x, y, z, hsml in tqdm(
                zip(
                    self.x,
                    self.y,
                    self.z,
                    h
                )
            )
        ])
        
        return self.densities
    
    
    def constraint_equation(self, h: np.ndarray):
        """
        The constraint equation minimised to find h.
        
        Returns the difference between the expected value and the 'true' value of
        the smoothing length from the measured density.
        """
        
        density = self.get_density(h)
        
        expected_h = self.eta * np.cbrt(self.m/density)
        
        diff = expected_h - h
        
        return diff
    
    
    def update_h(self) -> np.ndarray:
        """
        Updates the smoothing lengths.
        """
        
        output = run_iterations(
            self.hsml,
            self.constraint_equation,
            0.1,
            30,
        )
        
        self.hsml = output
        
        return self.hsml
        
