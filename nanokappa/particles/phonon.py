# Copyright (C) 2024, Bruno Hartmann da Silva
# License: MIT

import numpy as np
from types import FunctionType
from nanokappa.particles import Particle
from nanokappa.utils import Constants as C


class Phonon(Particle):
    """Phonon particle class.

    Attributes
    ----------
    position : np.ndarray
        Position of the particle.
    velocity : np.ndarray
        Velocity of the particle.
    omega : float
        Angular frequency.
    occupation : float
        Occupation number.
    tau_function : function
        The function to be called giving the relation tau = f(T), where tau is the relaxation time and T is the temperature.
    """
    omega: float
    occupation: float
    tau_function: FunctionType

    def scatter(self, T: float):
        """Scatter the phonon according to the lifetime approximation.

        Parameters
        ----------
        T : float
            The temperature the phonon is subjected to.
        """
        n0 = self.occupation_function(T)
        tau = self.tau_function(T)
        self.occupation = n0 + (self.occupation - n0) * np.exp(-self.dt / tau)

    def occupation_function(self, T):
        """Bose-Einstein distribution.

        Parameters
        ----------
        T : float
            The temperature the phonon is subjected to.

        Returns
        -------
        n : float
            The phonon occupation at that temperature.
        """
        return 0.5 + 1 / (np.exp(C.hbar * self.omega / (C.kb * T)) - 1)
