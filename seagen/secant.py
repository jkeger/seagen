"""
Secant method implementation for root-finding in SPH.

Created by: Josh Borrow
"""

import numpy as np
from numba import vectorize, float64
from typing import Callable, Tuple


@vectorize([float64(float64, float64, float64, float64)])
def x_n(x_n_1, x_n_2, f_x_n_1, f_x_n_2):
    """
    Calculate the next iteration of x.

    Inputs
    ------

    @param x_n_1 | float | x_{n-1}
    
    @param x_n_2 | float | x_{n-2}

    @param f_x_n_1 | float | f(x_{n-1})
    
    @param f_x_n_2 | float | f(x_{n-2})

    
    Outputs
    -------

    @output x_n | float | x_n, the next iteration in the Secant method.
    """

    bottom = f_x_n_1 - f_x_n_2

    top = x_n_2 * f_x_n_1 - x_n_1 * f_x_n_2

    return top / bottom


def initial_conditions(
        x: np.ndarray,
        f: Callable,
        extra_args=None,
        tol=0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the initial conditions for the secant method.

    Tol tells you how far to go in x (0.1 corresponds to 10%) when choosing
    the initial positions.

    Inputs
    ------

    @param x | np.ndarray | The 'x-values' for x_{n-2}

    @param f | callable | The function over which we wish to iterate

    @param extra_args | dict | extra arguments for f

    @param tol | float | the tolerance to create the ICs within.
    """

    x_n_1 = x * (1 - tol)
    
    f_x_n_1 = f(x_n_1)
    f_x_n_2 = f(x)

    return x_n_1, x, f_x_n_1, f_x_n_2


def run_iterations(
        x: np.ndarray,
        f: Callable,
        tol: float,
        n_calls: int,
        extra_args=None
    ) -> np.ndarray:
    """
    Run the secant method iterations up to a tolerence of tol and with a
    maximum number of iterations ncalls.

    This is slightly modified to the usual situaiton as it assumes that none
    of the x values are correlated with each other.

    Inputs
    ------

    @param x | np.ndarray | the initial guess for x

    @param f | callable | the constraint equation

    @param tol | float | the tolerenace to calculate the ideal value of x to.
                         This tolerance corresponds to the mean difference
                         between the two most recent values of x.

    @param n_calls | int | the maximum number of iterations

    @param extra_args | dict | extra arguments to pass to f.


    Outputs
    -------

    @output x | np.ndarray | the reduced value of x.
    """

    if extra_args is None: extra_args = {}

    x_n_1, x_n_2, f_x_n_1, f_x_n_2 = initial_conditions(x, f, extra_args)
    n_iter = 1

    while (np.mean(x_n_1 - x_n_2) < tol) and (n_iter < n_calls):
        # Iteration step
        x = x_n(x_n_1, x_n_2, f_x_n_1, f_x_n_2)

        # Update values
        x_n_2 = x_n_1
        x_n_1 = x

        f_x_n_2 = f_x_n_1
        f_x_n_1 = f(x_n_1, **extra_args)

        n_iter += 1

    return x

