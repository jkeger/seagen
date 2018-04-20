"""
This file is part of SEAGen.
Copyright (C) 2018 Jacob Kegerreis (jacob.kegerreis@durham.ac.uk)
GNU General Public License http://www.gnu.org/licenses/

Jacob Kegerreis and Josh Borrow

Helper functions for SEAGen.

Created by: Josh Borrow (joshua.borrow@durham.ac.uk) and Jacob Kegerreis
"""

import numpy as np

from typing import Tuple


class InvalidCoordinate(Exception):
    pass


def check_valid_polar(
        r: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray
    ) -> bool:
    """
    Check if these are valid polar co-ordinates; i.e.

        + Radius r \in [0, inf)
        + Zenith theta \in [0, pi)
        + Azimuth phi \in [0, 2 pi]

    If not, it raises an InvalidCoordinate exception.
    """

    if not ((min(r) >= 0.) and (np.isfinite(r).all())):
        raise InvalidCoordinate(
            "Your r values are not bounded between 0 and infinity"
        )

    if not ((min(theta) >= 0.) and (max(theta) <= np.pi)):
        raise InvalidCoordinate(
            "Your theta values are not bounded from 0 to <= pi"
        )

    if not ((min(phi) >= 0.) and (max(phi) < 2 * np.pi)):
        raise InvalidCoordinate(
            "Your phi values are not bounded from 0 to < 2 pi"
        )

    return True


def polar_to_cartesian(
        r: float,
        theta:float,
        phi: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts spherical polar to cartesian, and checks that they are valid
    using check_valid_polar.

    theta: zenith (colatitude)
    phi: azimuth (longitude)
    """

    if check_valid_polar(r, theta, phi):
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)

        return x, y, z

    else:
        return None, None, None


def get_euler_rotation_matrix(
        alpha: float,
        beta: float,
        gamma: float
    ) -> np.ndarray:
    """
    Return the rotation matrix for three Euler angles, alpha, beta, and
    gamma. Returns a 3x3 matrix as a np.ndarray.
    """
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    sb = np.sin(beta)
    cb = np.cos(beta)
    sg = np.sin(gamma)
    cg = np.cos(gamma)

    return np.array([
        [cg*cb*ca - sg*sa,    cg*cb*sa + sg*ca, -cg*sb],
        [-sg*cb*ca - cg*sa,  -sg*cb*sa + cg*ca,  sg*sb],
        [      sb*ca,             sb*sa,           cb ]
    ])


def get_shell_mass(r_inner: float, r_outer: float, rho: float) -> float:
    """
    Calculate the mass of a uniform-density shell.
    """
    return 4/3*np.pi * rho * (r_outer**3 - r_inner**3)


def get_mass_weighted_mean(A1_mass: float, A1_value: float) -> float:
    """
    Calculate the mean of the value array weighted by the mass array.

    @jacob -- is this actually a mean? At the moment this is just a 
    weighted sum.
    """
    return np.sum(A1_mass * A1_value) / np.sum(A1_mass)










